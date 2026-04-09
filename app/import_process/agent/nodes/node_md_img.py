import os
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

# MinIO相关依赖
from minio import Minio
from minio.deleteobjects import DeleteObject

# 【核心改造1：移除原生OpenAI，导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
# LLM客户端工具类（核心复用，替换原生OpenAI调用）
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖（消息构造+异常捕获）
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具（统一使用）
from app.core.logger import logger
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

"""
  主要目标： 将md中图片进行单独处理，方便后去模型识别图片的含义！
  主要动作： 图片->文件服务器-> 图片网络地址    （上文100）图片（下文100）->视觉模型-> 图片总结  
           ---》 [图片的总结](网络图片地址) -> state ->  md_content == 新的内容（图片处理后的）|| md_path = 处理后的md的地址
  总结技术：
        minio
        视觉模型： 提示词 + 访问 
  总结步骤： 
     1. 校验并且获取本次操作的数据 
        参数： state  -> md_path md_content 
        响应： 1. 校验后的md_content  2.md路径对象  3. 获取图片的文件夹 images
     2. 识别md中使用过的图片，采取做下一步（进行图片总结）
        参数： 1. md_content 2. images图片的文件夹地址
        响应： [(图片名,图片地址,(上文,下文))]
     3. 进行图片内容的总结和处理 （视觉模型）
        参数： 第二次的响应 [(图片名,图片地址,(上文,下文))]   || md文件的名称（提示词中 md文件名就是存储图片images的文件名）
        响应： {图片名:总结,......}
     4. 上传图片minio以及更新md的内容 
        参数：minio_client || {图片名:总结,......} || [(图片名,图片地址,(上文,下文))] (minio) || md_content 旧 || md文件的名称（提示词中 md文件名就是存储图片images的文件名）
        响应：new_md_content
        state[md_content] = new_md_content
     5. 进行数据的最终处理和备份 
        参数：new_md_content , 原md地址 -》 xx.md -> xx_new.md  
        响应：新的md的地址 new_md_path 
        state[md_path] = new_new_md_path
    return state
"""

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    这部分定义了一个白名单（.jpg, .png 等）。当侦探看到一个文件时，先查验它到底是不是真正的图片。
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """
    提取内容
    检查文件到底在不在，把 Markdown 的文本内容读到内存里，并且定位存放图片的 images 文件夹在哪
    :param state:
    :return:
    """
    # 1. 获取md的地址 md_path
    md_file_path = state["md_path"]
    if not md_file_path:
        raise ValueError("md_path不能为空！")

    md_path_obj = Path(md_file_path)
    if not md_path_obj.exists():
        raise FileNotFoundError(f"md_path:{md_file_path} 文件不存在！")

    # 要读取md_content
    if not state['md_content']:
        # 没有，再读取！ 有，证明是pdf节点解析过来的，已经给md_content进行赋值了！
        with md_path_obj.open("r", encoding="utf-8") as f:
            # 将所有文字内容打包成一个大字符串赋值给md_content
            md_content = f.read()
        state['md_content'] = md_content
        """
        关于with的拆解步骤
        f = md_path_obj.open("r")  # 1. 找管理员借出古籍
        md_content = f.read()      # 2. 阅读并抄写里面的内容
        f.close()                  # 3. 必须手动把古籍还给管理员！
        """
    # Todo: AI加了下面三行，用于测试main_graph。
    else:
        # 如果 state 中已经有 md_content，直接使用
        md_content = state['md_content']
    # 图片文件夹obj
    # 注意：自己传入的md -》 你的图片文件夹也必须交 images
    images_dir_obj = md_path_obj.parent / "images"
    return md_content, md_path_obj, images_dir_obj


def find_image_in_md_content(md_content, image_file, context_length: int = 100):
    """
      拿着图片的名字，去茫茫的 Markdown 文本里用正则表达式搜索
      找到这个图片被插入的具体位置，并向前、向后各截取 100 个字符
    从md_content识别图片的上下文！
    约定上下文长度100
    :param md_content:
    :param image_file: cat.png
    :param context_length：默认截取长度
    :return:result：[(上文),(下文)]
    """
    """
    # 你好啊
    我很好，还有7行代码今天就结束了！小伙伴们坚持好！谢谢！
    哈哈
    哈
    嘿嘿
    【start】 ![二大爷](/xxx/xx/zhaoweifeng.jpgxxx)【end】
    啦啦啦啦
    巴巴爸爸
    ![二大爷](/xxx/xx/zhaoweifeng.jpgxxx)
    嘿嘿额
    file_name zhaoweifeng.jpg
    """
    # 定义正则表达式  .*  .*?
    # r -> 不要把它们当成普通的换行符或转义符给吞了
    # 给个模版（通缉令）
    pattern = re.compile(r"!\[.*?\]\(.*?" + image_file + ".*?\)")

    content = None
    # 查询符合位置
    # 拿着模版去md_content中找出现的位置
    # 注意：finditer不会找到一个就停下，此处写死只要第一个。
    items = list(pattern.finditer(md_content))
    if not items:
        return None
    if item := items[0]:
        start, end = item.span()  # span获取匹配对象的起始和终止的位置
        # 4. 以图片为中心，向前切 100 个字（作为上文），向后切 100 个字（作为下文）
        # max 和 min 是为了防止切到文章边界外报错
        pre_text = md_content[max(start - context_length, 0):start]  # 考虑前面有没有context_length 没有从0开始
        post_text = md_content[end:min(end + context_length, len(md_content))]  # 考虑后面有没有context_length 没有就到长度
        # 将上下文字符串打包成元组
        content = (pre_text, post_text)
    # 截取位置前后的内容
    if content:
        # logger.info(f"图片：{image_file} ,在{md_content[:100]}中使用了：{len(results)}次，截取第一个上下文：{results[0]}")
        logger.info(f"图片：{image_file} ,在{md_content[:100]}，截取第一个上下文：{content}")
        # 返回第一个上下文[("我看到", "在睡觉。")]
        return content


def step_2_scan_images(md_content: str, images_dir_obj: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    进行md中图片识别，并且截取图片对应的上下文环境
    把 images 文件夹里的图片一张张拿出来，调用上面的勘探方法。
    最后生成一份清单（包含图片名、路径、上下文）
    :param md_content:
    :param images_dir_obj:
    :return: [(图片名，图片地址，上下元组())]
    """
    # 1. 我们先创建一个目标集合
    targets = []
    # 2. 循环读取images中的所有图片，校验在md中是否使用，使用了就截取上下文
    # 列出指定文件夹里面所有的文件和子文件夹的名称。
    # os.listdir()->['cat.png', 'dog.jpg', 'readme.txt', '新文件夹']
    for image_file in os.listdir(images_dir_obj):
        # 遍历每个文件的名字
        # 过滤掉非图片文件（比如隐藏文件）
        if not is_supported_image(image_file):
            logger.warning(f"当前文件：{image_file},不是图片格式，无需处理！")
            continue
        # 是图片，我们就在md查询，看是否存在，存在，读取对应的上下文即可
        # （上，下文）
        # find_image_in_md_content返回值 -> 元组content = (pre_text, post_text)
        # content_data = content = (pre_text, post_text)
        content_data = find_image_in_md_content(md_content, image_file)
        if not content_data:
            logger.warning(f"图片：{image_file}没有在md内容使用！上下文为空！")
            continue
        # targets：[(图片名，图片地址，上下文元组(上文，下文))]
        # images_dir_obj = "/output/20260404/task_abc123/小米汽车SU7说明书/images"（字符串）
        # str(images_dir_obj / image_file) = Path(images_dir_obj) / "cat.png"
        targets.append((image_file, str(images_dir_obj / image_file), content_data))

    return targets


# def step_3_generate_img_summaries(targets, stem):
#     """
#     获取图片的内容描述！ 利用视觉模型！
#     :param targets: [(图片名.xxx,图片地址,(上文,下文))，(图片名.xxx,图片地址,(上文,下文))]
#     :param stem:  文件夹的名字  md名称 output / h180xxxx /  h180xxxx.md  | images
#     :return: {图片名.xx : 总结和描述 , 图片名.xx : 总结和描述 , 图片名.xx : 总结和描述 ,图片名.xx : 总结和描述....}
#     """
#     # 准备了一个空的字典 summaries，用来当做最后的“鉴定报告汇总表”
#     summaries = {} # 最终结果
#     # 循环每一张图片，向视觉模型进行请求，获取总结结果！
#     # 确保一个对类对象就行了！！！
#     # 名为 request_times 的队列，用来记录每次发请求的时间
#     request_times = deque()
#     for image_file,image_path, context in  targets:
#         # 解构 图片名 图片地址 (上,下)
#         # 1. 访问限速问题（我们模型的限速标准 1分钟 可以访问10  限制并发访问次数..）
#         apply_api_rate_limit(request_times, max_requests=9)
#         # 2. 向视觉模型发起请求
#         # 2.1 模型对象
#         vm_model = get_llm_client(model=lm_config.lv_model)
#         # 2.2 准备提示词
#         # 加载了提示词模版并格式化了提示词
#         # 此时 prompt 变成了一段极其具体的纯文本指令：
#         #"你是一个图片分析专家。当前图片出自文件夹【hak180产品安全手册】。上下文是：【操作机器前请注意，否则可能引起烫伤。】。请你总结一下这张图片画了什么。"
#         prompt = load_prompt("image_summary",root_folder=stem,image_content=context)
#
#         # import base64
#         with open(image_path, "rb") as f:
#             #AI 专家远在云端，你不能直接顺着网线把 D 盘里的图片硬塞给它。
#             # 所以，代码以只读二进制模式（"rb"）打开图片，然后用 base64 技术，将整张图片的像素数据强制编码成一串极其漫长的“乱码字符串”
#             # image_base64 = "iVBORw0KGgoAAAANSUhEUgAA..."（省略几万字）
#             image_base64 = base64.b64encode(f.read()).decode("utf-8")  # 字节转成字符
#
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             # 直接放图片的网络地址 "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
#                             # base64图片转后的字符串  jpg -> image/jpeg
#                             "url": f"data:image/jpeg;base64,{image_base64}"
#                         },
#                     },
#                     {
#                         "type": "text",
#                         "text": f"{prompt}"
#                     },
#                 ],
#             },
#         ]
#         # 2.3 执行获取总结
#         # 假设 AI 看完图片和“防烫伤”的上下文后，返回的原始文本是:
#         # response->"\n图中是一个红色的三角形警告标志，中间画着一双正在冒烟的手。 \n"
#         response = vm_model.invoke(messages)
#         # summary->"图中是一个红色的三角形警告标志，中间画着一双正在冒烟的手。"
#         summary = response.content.strip().replace("\n","")
#         # summaries 字典现在变成了：
#         # {
#         #     "warning.png": "图中是一个红色的三角形警告标志，中间画着一双正在冒烟的手。",
#         #     "machine_front.jpg": "产品正面外观图，展示了电源按钮和显示屏的位置。",
#         #     ...
#         # }
#         summaries[image_file] = summary
#         logger.info(f"图片：{image_file}，总结结果：{summary}")
#     logger.info(f"总结图片，获取结果：{summaries}")
#     return summaries


def step_3_generate_img_summaries(targets, stem):
    """
    使用 LangChain 流水线重构的图片总结函数
    """
    summaries = {}
    # 用于记录请求时间，防止被 API 封锁
    request_times = deque()

    # 【亮点 1：在循环外，提前搭建好一条固定的“处理流水线”】
    # 1. 制造“模具”：定义好包含图片和文字的结构化消息模板
    # 制造 LangChain 模具：定义好大模型接收的多模态消息格式（包含一张图 + 一段字）
    chat_template = ChatPromptTemplate.from_messages([
        ("user", [
            # 告诉 LangChain 这里有个坑位叫 img_base64，它是一张图片
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{img_base64}"}},
            # 告诉 LangChain 这里有个坑位叫 text_prompt，它是一段文字
            {"type": "text", "text": "{text_prompt}"}
        ])
    ])

    # 2. 获取你配置的视觉大模型客户端（如千问VL）
    vm_model = get_llm_client(model=lm_config.lv_model)

    # 3. 拼装传送带 (LCEL语法)： 模板拼装 -> 交给大模型 -> 自动提取纯文本
    # 组装神级流水线： 模具拼装 | 送入大模型 | 把大模型的胡言乱语剥离成纯文本
    chain = chat_template | vm_model | StrOutputParser()

    # 开始批量处理物证
    # targets：[(图片名，图片地址，上下文元组(上文，下文))]
    for image_file, image_path, context in targets:
        apply_api_rate_limit(request_times, max_requests=9) # 限速控制

        # 准备材料 1：将图片转为 base64 乱码
        # 将本地的二进制图片，强行转化为大模型能看懂的 base64 超长乱码字符串
        # image_path = "/output/20260404/task_abc123/小米汽车SU7说明书/images/cat.png"（字符串）
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        # 准备材料 2：继续使用你们优雅的 load_prompt 加载外部文字话术
        # 加载提示词：告诉大模型“结合以下上下文，总结这张图”
        # content = (pre_text, post_text)
        # root_folder=stem中的stem为：md_path_obj.stem -> 小米汽车SU7说明书
        # md_path_obj = Path("/output/20260404/task_abc123/小米汽车SU7说明书/小米汽车SU7说明书.md")
        # 即：text_prompt -> 这是“{小米汽车SU7说明书}”文件中的一张图片，图片上文部分为“{image_content[0]}”，
        # 下文部分为“{image_content[1]}”，请用中文简要总结这张图片的内容，用于 Markdown 图片标题，控制在50字以内。
        text_prompt = load_prompt("image_summary", root_folder=stem, image_content=context)
        # 一键触发流水线，把变量塞进模具！
        summary = chain.invoke({
            "img_base64": image_base64,
            "text_prompt": text_prompt
        })

        # 此时得到的 summary 已经是干净的字符串了，只需要简单去个回车
        # summary 的真实面目只是一个 String ->
           # summary = "图中展示了小米SU7车门侧面的半隐藏式门把手结构..."
        # 去掉回车换行，并存入字典:summaries
        # 格式为 {"su7.jpg": "图中展示了小米SU7车门侧面的半隐藏式门把手结构..."}
        summary = summary.strip().replace("\n", "")
        summaries[image_file] = summary
        logger.info(f"图片：{image_file}，总结结果：{summary}")

    logger.info(f"总结图片，获取结果：{summaries}")
    return summaries

def step_4_upload_images_and_replace_md(summaries, targets, md_content, stem):
    """
    将我们图片传递到minio服务器
    替换原md中的图片和描述
    :param summaries:  图片名 ： 描述
    :param targets:  （图片名，原地址，（上，下））
    :param md_content: 原md内容
    :param stem: 文件名
    :return: 新md
    """
    # 理解minio存储结果： 桶 / upload-images / 文件夹名字 / 图片对象.jpg
    minio_client = get_minio_client()
    # ------------------- 动作1：清理旧资产 ------------------
    # 1.  删除minio中的对应文件的图片
    # 1.1 获取要删除的对象
    # Object object_name
    # 注意：{minio_config.minio_img_dir[1:]}  一定要去掉一个 /
    """
    总管家带着案卷代号 "hak180产品安全手册"，
    去 MinIO 仓库里找对应的专属货架（prefix 路径）。
    把货架上旧的 warning.png 统统打包到 delete_object_list 垃圾袋里，
    直接一把火烧掉（remove_objects）。打扫出一个干干净净的新货架
    """
    object_list = minio_client.list_objects(minio_config.bucket_name,
                              prefix= f"{minio_config.minio_img_dir[1:]}/{stem}",
                              recursive=True)
    # 都有一个对象的名
    delete_object_list = [DeleteObject(obj.object_name) for obj in object_list]
    # 需要的DeleteObject
    # 1.2 调用方法进行删除即可
    errors = minio_client.remove_objects(minio_config.bucket_name,delete_object_list)
    for errors in errors:
        logger.error(f"删除对象失败：{errors}")
    logger.info(f"已经完成{stem}下的对象清空，本次删除了：{len(delete_object_list)}个对象！！！")

    # ------------------ - 动作2：上传图片 --------------
    # 2. 上传图片到minio服务器
    # 记录图片上传结果的字典
    images_url = {}
    # targets:[（图片名，原地址，（上，下））,（图片名，原地址，（上，下））]
    """
    [
        ("su7_door.jpg", "D:/.../su7_door.jpg", (...)), 
        ("su7_front.jpg", "D:/.../su7_front.jpg", (...))
    ]
    """
    for image_file,image_path, _ in targets:
        try:
            # 把本地图片传到云端桶里
            minio_client.fput_object(
                bucket_name= minio_config.bucket_name,
                object_name= f"{minio_config.minio_img_dir}/{stem}/{image_file}", # 传入minio 桶后面的命名  xx.png  xx/xxx/xx.png
                file_path= image_path,
                content_type="image/jpeg"
            )
            # 拼接出可通过公网访问的 URL
            # 图片地址 = 协议 + 端点 + 桶名 + 对象名  http://47.94.86.115:9000/ 桶名 / 对象名
            """
                images_url = {
                    "su7_door.jpg": "http://minio.../小米汽车SU7说明书/su7_door.jpg",
                    "su7_front.jpg": "http://minio.../小米汽车SU7说明书/su7_front.jpg"
                }
            """
            images_url[image_file] = f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_file}"
            logger.info(f"完成图片{image_file}上传，访问地址为：{images_url[image_file]}")
        except Exception as e:
            logger.error(f"上传图片失败：{image_file}，失败原因：{e}")

    # ---------- 动作3：偷天换日（替换 Markdown 中的图片内容） ---------
    # 有了AI总结（summaries），也有了云端地址（images_url），代码需要把它们“合二为一”
      # image_file：图片名
      # summary：图片描述总结
    # summaries  -> {image_file：summary}
    # images_url -> {image_file：url}
    image_infos = {}
    # 把总结文字和云端 URL 揉合在一起
    for image_file, summary in summaries.items():
        # 如果这个图片既有总结，又成功上传到了云端（url := images_url.get(image_file)为真）
        # 就把它们打包成一个元组(描述, 网址)
        if url := images_url.get(image_file):
            image_infos[image_file] = (summary,url)
    logger.info(f"图片处理的汇总结果:{image_infos}")
    # 最终汇总image_infos： {图片名 : (描述 , url地址)}
    """
    image_infos = {
        "su7_door.jpg": (
            "小米SU7半隐藏式门把手特写",         # 来自 summaries
            "http://minio.../su7_door.jpg"   # 来自 images_url
        ),
        "su7_front.jpg": (
            "水滴形矩阵LED大灯展示", 
            "http://minio.../su7_front.jpg"
        )
    }
    """

    if image_infos:
        """
        xxxx
        xxx  ![xx](图片地址/image_file) -> ![summary](minio的url)
        xxxx
        """

        """
        rep.sub(新文本, 旧长文) 就像是 Word 里的“全部替换”功能。它扫描整篇 md_content：
        发现了旧代码：![原本的废话](/images/warning.png)。
        咔嚓一下！把它生生抠掉。
        换成：![图中是一个红色的三角形警告标志...](http://192.168.1.100:9000/.../warning.png)
        """
        for image_file, (summary, url) in image_infos.items():
            # 使用正则扫描出旧的标签
            # ![](/xxx/xx/image_file) -> ![无所谓](无所谓image_file无所谓)
            rep = re.compile(r"!\[.*?\]\(.*?"+image_file+".*?\)")
            # 把它替换成包含AI总结和云端地址的新标签：
            # ![AI总结](公网URL)
            md_content = rep.sub(f"![{summary}]({url})", md_content)
        logger.info(f"已经完成md内容的替换，新的内容为:{md_content}")
    # 把图片的内容替换完毕之后，在原md中全部替换新的图片内容
    return md_content


def step_5_replace_md_and_save(new_md_content, md_path_obj):
    """
    完成新的md的磁盘本分，并且返回老地址！
    新的命名  xxx_new.md
    :param new_md_content: -> 承接第四步的额返回值md_content
    :param md_path_obj: 老地址
    :return: 新地址
    """
    # 系统在生成新文件时，为了不破坏原始文件（保留底稿以防万一），决定新建一个文件。
    # 这里的 os.path.splitext() 就像是一把“精准的手术刀”。
    # 它专门用来把文件路径里的**“名字”和“后缀名（扩展名）”**一刀切开
    # md_path_obj = Path("/output/20260404/task_abc123/小米汽车SU7说明书/小米汽车SU7说明书.md")
    # new_md_path_str = Path("/output/20260404/task_abc123/小米汽车SU7说明书/小米汽车SU7说明书_new.md")
    # 给文件改名，加上 _new 后缀，以防覆盖原稿
        # 当代码执行 os.path.splitext(md_path_obj)[0] + "_new.md" 时，
        # 发生了一个隐式的降级操作： os.path 系列的老牌函数在处理 Path 对象时，会提取它的文本路径。
        # 加上后面的字符串拼接 + "_new.md"，生成的结果就彻底变成了一个普通的字符串（比如 "D:/.../小米汽车SU7说明书_new.md"）
    new_md_path_str = os.path.splitext(md_path_obj)[0] + "_new.md"
    #当以 "w" 模式打开一个文件时，如果这个文件不存在，系统会立刻凭空创建一个空白文件；
    # 如果已经存在，系统会极其无情地把里面原来的内容全部清空
    # 将包含云端链接和AI总结的全新文本，写入新文件
    with open(new_md_path_str, "w", encoding="utf-8") as f:
        # 就像是打印机一样，把内存里那段已经排版完美、带有 MinIO 链接和 AI 总结文本，刻录进硬盘
        f.write(new_md_content)
    logger.info(f"已经完成了新内容的写入，新的地址为:{new_md_path_str}")
    return new_md_path_str




def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    # 1. 自动获取当前工位名称："node_md_img"
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    # 前端进度条点亮黄灯，显示正在处理
    add_running_task(state['task_id'], function_name)

    # 2. 呼叫step1，校验，并提取长文本、文件路径和图片文件夹路径
    #         参数： state  -> md_path md_content
    #         响应： 1.校验后的md_content  2.md路径对象  3. 获取图片的文件夹 images
    md_content, md_path_obj, images_dir_obj = step_1_get_content(state)
    # 如果压根就没有 images 文件夹（比如这篇文档全是纯文字，根本没配图）-> 提前下班
    # 如果说明书没有图，就不需要再去唤醒耗时又费钱的视觉大模型了
    if not images_dir_obj.exists():
        logger.info(f">>> [{function_name}]没有图片，直接返回 state ！")
        return state

    # 3. 识别md中使用过的图片 -> 把所有图片的上下文全给我揪出来！
    # targets->[(图片名,图片地址,(上文,下文 = 100))，(图片名,图片地址,(上文,下文 = 100))，(图片名,图片地址,(上文,下文 = 100))]
    targets = step_2_scan_images(md_content, images_dir_obj)
    #     参数： 1. md_content 2. images图片的文件夹地址
    #     响应： targets = [(图片名,图片地址,(上文,下文))]

    # 4. 进行图片内容的总结和处理 （视觉模型）
    # 参数： 1.第二次的响应 [(图片名,图片地址,(上文,下文))] 2.md文件的名称（提示词中 md文件名就是存储图片images的文件名）
    # 响应： summaries = {图片名:总结,图片名:总结,图片名:总结}
    summaries = step_3_generate_img_summaries(targets, md_path_obj.stem)

    # 5. 上传图片到minio同时替换md中的图片内容为（描述 + url地址）
    #         参数：1.{图片名:总结,......} 2.[(图片名,图片地址,(上文,下文))] (minio) 3.md_content 旧 4.md文件的名称
    #         响应：new_md_content
    new_md_content = step_4_upload_images_and_replace_md(summaries, targets, md_content, md_path_obj.stem)

    # 6. 新的md内容替换和保存修改装 -> _new
    #  参数：new_md_content , 原md地址 -> xx.md -> xx_new.md
    #  响应：新的md的地址 new_md_path
    new_md_file_path = step_5_replace_md_and_save(new_md_content, md_path_obj)
    #  md_path -> 新的地址：new_md_file_path
    #  md_content -> 新的内容 ：new_md_content
    # new_md_file_path（step5返回值） = "/output/20260404/task_abc123/小米汽车SU7说明书/小米汽车SU7说明书_new.md"（字符串）
    # new_md_content（step4返回值）：原md中图片的内容替换完毕后的，图片的新内容
    state["md_path"] = new_md_file_path
    state["md_content"] = new_md_content
    logger.info(f">>> [{function_name}]开始结束了！现在的状态为：{state}")
    add_done_task(state['task_id'], function_name)
    return state


if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")