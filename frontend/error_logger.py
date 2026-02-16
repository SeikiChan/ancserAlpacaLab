"""
Error Logger for Streamlit App
自动捕获错误并写入日志文件
"""
import logging
import sys
import os
from datetime import datetime
import warnings

def setup_logging():
    """设置日志系统"""
    # 创建 logs 目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件名：dashboard_YYYY-MM-DD.log
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"dashboard_{today}.log")

    # 配置 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)  # 同时输出到终端
        ]
    )

    # 捕获所有 warnings
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger('py.warnings')
    warnings_logger.setLevel(logging.WARNING)

    # 重定向 stderr 到 logger (捕获 pandas 等库的警告)
    class LoggerWriter:
        def __init__(self, level):
            self.level = level

        def write(self, message):
            if message.strip():
                self.level(message.strip())

        def flush(self):
            pass

    # sys.stderr = LoggerWriter(logging.warning)

    logging.info("=" * 60)
    logging.info(f"Dashboard started at {datetime.now()}")
    logging.info("=" * 60)

    return log_file

def log_error(error: Exception, context: str = ""):
    """记录错误到日志"""
    import traceback
    error_msg = f"ERROR in {context}: {str(error)}\n{traceback.format_exc()}"
    logging.error(error_msg)

def log_warning(message: str):
    """记录警告到日志"""
    logging.warning(message)

def log_info(message: str):
    """记录信息到日志"""
    logging.info(message)
