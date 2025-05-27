import logging
import os

def setup_logging():
    """独立日志配置模块"""
    log_path = os.environ.get("LOG_PATH", "/tmp/2api.log")
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get("LOG_FORMAT", 
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    stream_handler = logging.StreamHandler()
    
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        handlers=[stream_handler, file_handler]
    )
    return logging.getLogger('2api')

logger = setup_logging()