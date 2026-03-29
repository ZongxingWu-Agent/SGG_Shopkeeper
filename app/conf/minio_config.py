# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv
# import dotenv
# 提前加载.env配置文件（确保os.getenv能获取到MinIO相关配置）
# dotenv.load_dotenv()
load_dotenv()
"""
核心功能
1、定义配置结构
    使用 Python dataclass 定义 MinIOConfig 类
    规范 MinIO 连接所需的所有参数
2、从环境变量读取配置
    通过 load_dotenv() 加载项目根目录的 .env 文件
    使用 os.getenv() 获取 MinIO 相关的环境变量
3、集中管理 MinIO 连接信息
    将分散的配置项统一到一个配置对象中
    供 minio_utils.py 等其他模块调用
"""

# 定义MinIO对象存储服务配置（与LLMConfig风格一致，字段对应.env配置项）
@dataclass
class MinIOConfig:
    endpoint: str    # MinIO服务地址（含http/https和端口）
    access_key: str  # MinIO访问密钥（对应MINIO_ACCESS_KEY）
    secret_key: str  # MinIO秘钥（对应MINIO_SECRET_KEY）
    bucket_name: str # MinIO默认存储桶名（知识库文件专用）
    minio_img_dir: str #Minio存储图片的文件夹
    minio_secure: bool # 是否使用ssl加密 http 还是 https


# 实例化MinIO配置对象，自动从.env读取配置并绑定
minio_config = MinIOConfig(
    endpoint=os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    bucket_name=os.getenv("MINIO_BUCKET_NAME"),
    minio_img_dir=os.getenv("MINIO_IMG_DIR"),
    minio_secure=os.getenv("MINIO_SECURE") == "True"
)