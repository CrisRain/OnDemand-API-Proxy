import threading
import time
from datetime import datetime, timedelta
from functools import wraps
from utils import logger
import config

class RateLimiter:
    """请求速率限制器 (基于token/IP)"""
    def __init__(self, limit_per_minute=None): # 允许传入参数，但优先配置
        # 优先从配置读取，如果未配置或传入了明确值，则使用该值
        # 配置项: "rate_limit"
        configured_limit = config.get_config_value("rate_limit", default=30) # 从配置读取，默认30次/分钟
        self.limit = limit_per_minute if limit_per_minute is not None else configured_limit
        self.window_size = config.get_config_value("rate_limit_window_seconds", default=60) # 从配置读取，默认60秒
        self.requests = {}  # {identifier: [timestamp1, timestamp2, ...]}
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> bool:
        """
        检查标识符请求是否允许
        
        参数:
            identifier: 唯一标识 (token/IP)
            
        返回:
            bool: 允许则True，否则False
        """
        with self.lock:
            now = time.time()
            if identifier not in self.requests:
                self.requests[identifier] = []
            
            # 清理过期请求
            self.requests[identifier] = [t for t in self.requests[identifier] if now - t < self.window_size]
            
            # 检查请求数是否超限
            if len(self.requests[identifier]) >= self.limit:
                return False
            
            # 记录当前请求
            self.requests[identifier].append(now)
            return True

from typing import Dict # 确保 Dict 已导入，如果其他方式未导入类型提示

# 由于 client_sessions 相关逻辑已移除，以下会话清理功能不再需要。
# def _cleanup_user_sessions(...)
# def session_cleanup(...)
# def start_cleanup_thread(...)
# _cleanup_thread_started = False
# _cleanup_thread_lock = threading.Lock()