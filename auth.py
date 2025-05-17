import threading
import time
from datetime import datetime, timedelta
from functools import wraps
# from flask import request, jsonify # 移除冗余导入
from utils import logger
import config

class RateLimiter:
    """请求速率限制器 (基于token/IP)"""
    def __init__(self, limit_per_minute=None): # 允许传入参数，但优先配置
        # 优先从配置读取，如果未配置或传入了明确值，则使用该值
        # 配置项: "rate_limit"
        configured_limit = config.get_config_value("rate_limit", default=60) # 默认60次/分钟
        self.limit = limit_per_minute if limit_per_minute is not None else configured_limit
        self.window_size = 60  # 窗口大小（秒）
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

def session_cleanup():
    """定期清理过期会话"""
    # 获取配置
    config_instance = config.config_instance
    
    with config_instance.client_sessions_lock:
        current_time = datetime.now()
        total_expired = 0
        
        # 遍历用户
        for user_id in list(config_instance.client_sessions.keys()):
            user_sessions = config_instance.client_sessions[user_id]
            expired_accounts = []
            
            # 遍历账户会话
            for account_email, session_data in user_sessions.items():
                last_time = session_data["last_time"]
                if current_time - last_time > timedelta(minutes=config_instance.get('session_timeout_minutes')):
                    expired_accounts.append(account_email)
                    # 记录过期会话信息 (上下文/IP)
                    context_info = session_data.get("context", "无上下文")
                    ip_info = session_data.get("ip", "无IP")
                    # 上下文预览(前30字符)，防日志过长
                    context_preview = context_info[:30] + "..." if len(context_info) > 30 else context_info
                    logger.debug(f"过期会话: 用户={user_id[:8]}..., 账户={account_email}, 上下文={context_preview}, IP={ip_info}")
            
            # 删除过期账户会话
            for account_email in expired_accounts:
                del user_sessions[account_email]
                total_expired += 1
            
            # 若用户无会话，则删除
            if not user_sessions:
                del config_instance.client_sessions[user_id]
        
        if total_expired:
            logger.info(f"已清理 {total_expired} 个过期会话")

_cleanup_thread_started = False
_cleanup_thread_lock = threading.Lock()

def start_cleanup_thread():
    """启动会话定期清理线程 (幂等)"""
    global _cleanup_thread_started
    with _cleanup_thread_lock:
        if _cleanup_thread_started:
            logger.debug("会话清理线程已运行，跳过此次启动。")
            return
        
        def cleanup_worker():
            while True:
                # 循环内获取最新配置，防动态更新
                try:
                    timeout_minutes = config.get_config_value('session_timeout_minutes', default=30) # 默认值
                    sleep_interval = timeout_minutes * 60 / 2
                    if sleep_interval <= 0: # 防无效休眠间隔
                        logger.warning(f"无效会话清理休眠间隔: {sleep_interval}s, 用默认15分钟。")
                        sleep_interval = 15 * 60
                    time.sleep(sleep_interval)
                    session_cleanup()
                except Exception as e:
                    logger.error(f"会话清理线程异常: {e}", exc_info=True) # 添加 exc_info=True 获取更详细的堆栈

        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True, name="SessionCleanupThread")
        cleanup_thread.start()
        _cleanup_thread_started = True
        logger.info("会话清理线程启动成功。")