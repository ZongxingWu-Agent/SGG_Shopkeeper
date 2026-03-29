# 导入Python内置模块
import os
import json
# 导入MinIO官方Python SDK核心类
from minio import Minio
# 项目内部配置与日志
from app.conf.minio_config import minio_config
from app.core.logger import logger
"""
核心作用
1、初始化 MinIO 客户端
    从配置文件 minio_config 读取连接信息（endpoint、access_key、secret_key 等）
    创建全局 MinIO 客户端对象 minio_client
2、自动管理存储桶（Bucket）
    检查配置的存储桶是否存在
    不存在时自动创建
    避免重复创建
3、配置访问策略
    设置存储桶为公网只读模式
    允许匿名用户通过 URL 直接访问文件
    适合用于公开资源、文档等的分发场景
    4、提供客户端获取接口
    get_minio_client() 函数供项目其他地方调用已初始化的客户端
5、使用场景
    从项目结构看，这个 MinIO 客户端主要用于：
    存储导入的文档（PDF、手册等）
    存储处理后的 Markdown 文件
    存储图片等资源
    作为对象存储服务，配合 Milvus 向量数据库、MongoDB 历史记录等一起使用
"""
# 全局MinIO客户端对象，初始化后供全项目调用
minio_client = None

try:
    # 初始化MinIO客户端实例
    minio_client = Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=False  # 内网/本地部署用HTTP，公网部署需改为True并配置SSL
    )
    bucket_name = minio_config.bucket_name

    # 检查存储桶是否存在，不存在则自动创建
    if not minio_client.bucket_exists(bucket_name):
        logger.info(f"MinIO存储桶[{bucket_name}]不存在，开始创建")
        minio_client.make_bucket(bucket_name)
        logger.info(f"MinIO存储桶[{bucket_name}]创建成功")
    else:
        logger.info(f"MinIO存储桶[{bucket_name}]已存在，无需重复创建")

    # 配置存储桶公网只读策略：允许匿名用户通过URL直接访问桶内文件
    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"AWS": ["*"]},  # *表示所有匿名用户（S3兼容标识）
            "Action": ["s3:GetObject"],   # 仅授权文件获取/访问操作
            "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
        }]
    }
    minio_client.set_bucket_policy(bucket_name, json.dumps(bucket_policy))
    logger.info(f"MinIO存储桶[{bucket_name}]已配置公网只读策略，支持匿名URL访问")

except Exception as e:
    # 捕获初始化异常，记录错误日志并置空客户端
    logger.error(f"MinIO客户端初始化失败，错误信息：{str(e)}", exc_info=True)
    minio_client = None


def get_minio_client():
    """
    获取全局初始化的MinIO客户端实例
    :return: 已初始化的Minio对象 / None（初始化失败时）
    """
    return minio_client