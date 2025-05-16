import threading
import time
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify
from utils import logger
import config

class RateLimiter:
    """速率限制器，限制每个IP的请求频率"""
    def __init__(self, limit_per_minute):
        self.limit = limit_per_minute
        self.window_size = 60  # 窗口大小（秒）
        self.requests = {}  # {ip: [timestamp1, timestamp2, ...]}
        self.lock = threading.Lock()
    
    def is_allowed(self, ip: str) -> bool:
        with self.lock:
            now = time.time()
            if ip not in self.requests:
                self.requests[ip] = []
            
            # 移除窗口外的请求
            self.requests[ip] = [t for t in self.requests[ip] if now - t < self.window_size]
            
            # 检查是否超过限制
            if len(self.requests[ip]) >= self.limit:
                return False
            
            # 添加新请求
            self.requests[ip].append(now)
            return True

def session_cleanup():
    """定期清理过期的会话"""
    # 获取配置实例
    config_instance = config.config_instance
    
    with config_instance.client_sessions_lock:
        current_time = datetime.now()
        expired_keys = []
        for client_id, session_data in config_instance.client_sessions.items():
            last_time = session_data["last_time"]
            if current_time - last_time > timedelta(minutes=config_instance.get('session_timeout_minutes')):
                expired_keys.append(client_id)
        
        for key in expired_keys:
            del config_instance.client_sessions[key]
        
        if expired_keys:
            logger.info(f"已清理 {len(expired_keys)} 个过期会话")

def start_cleanup_thread():
    """启动定期清理线程"""
    def cleanup_worker():
        while True:
            time.sleep(config.get_config_value('session_timeout_minutes') * 60 / 2)  # 每半个超时周期清理一次
            try:
                session_cleanup()
            except Exception as e:
                logger.error(f"会话清理出错: {e}")
    
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    logger.info("会话清理线程已启动")