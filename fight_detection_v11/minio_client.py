import io
import logging
from datetime import datetime, timedelta, timezone
from minio.deleteobjects import DeleteObject
from minio import Minio
from config import config
logger = logging.getLogger(__name__)

class MinioUploader:
    def __init__(self):
        # 初始化 MinIO 客户端
        self.client = Minio(
            endpoint=config.minio_endpoint,
            access_key=config.minio_access_key,
            secret_key=config.minio_secret_key,
            secure=config.minio_secure
        )
        self.bucket = config.minio_bucket
        self._ensure_bucket()

    def _ensure_bucket(self):
        """确保指定的存储桶存在，如果不存在则自动创建。"""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"成功创建 MinIO 存储桶: {self.bucket}")
        except Exception as e:
            logger.error(f"MinIO 存储桶检查/创建失败: {e}")

    def upload_image(self, image_bytes: bytes, filename: str):
        """异步上传图片 (此方法由线程池或后台任务调用执行)"""
        try:
            self.client.put_object(
                bucket_name=self.bucket,
                object_name=filename,
                data=io.BytesIO(image_bytes),
                length=len(image_bytes),
                content_type="image/jpeg"
            )
            # logger.debug(f"成功将 {filename} 上传至 MinIO")
        except Exception as e:
            logger.error(f"上传 {filename} 到 MinIO 失败: {e}")

    def cleanup_old_images(self, days_ago=1):
        """扫描并删除存储桶内指定天数（默认1天）之前的旧照片"""
        now = datetime.now(timezone.utc)
        threshold = now - timedelta(days=days_ago)
        
        logger.info(f"--- 开启 MinIO 存储空间自动化清理计划(阈值: {days_ago}天前) ---")
        
        try:
            # 1. 递归扫描获取所有对象信息
            objects = self.client.list_objects(self.bucket, recursive=True)
            
            # 2. 筛选出超过时间阈值的对象
            to_delete = []
            for obj in objects:
                # 某些 MinIO 版本下 last_modified 为 UTC 时间
                if obj.last_modified < threshold:
                    to_delete.append(DeleteObject(obj.object_name))
            
            # 3. 如果有待删除项，执行原子性批量删除
            if to_delete:
                errors = self.client.remove_objects(self.bucket, to_delete)
                # remove_objects 只有在删除出错时才会返回生成器
                for error in errors:
                    logger.error(f"清理旧图像时遇到错误: {error}")
                logger.info(f"清理完成：成功收回 {len(to_delete)} 张旧照片的物理存储空间。")
            else:
                logger.info("清理检查完毕：没有发现任何过期的过期旧照片。")
                
        except Exception as e:
            logger.error(f"执行 MinIO 定时清理任务时由于异常被迫崩溃: {e}")

# 全局 MinIO 上传实例
minio_uploader = MinioUploader()
