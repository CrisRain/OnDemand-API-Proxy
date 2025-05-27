import time
import logging
import functools
import requests
from abc import ABC, abstractmethod
from typing import Callable, Any, Dict, Optional, Type, Union, TypeVar, cast
from datetime import datetime # <--- 移动到这里

# 导入配置模块
import config

# 类型变量定义
T = TypeVar('T')

class RetryStrategy(ABC):
    """重试策略的抽象基类"""
    
    @abstractmethod
    def should_retry(self, exception: Exception, retry_count: int, max_retries: int) -> bool:
        """
        判断是否应该重试
        
        Args:
            exception: 捕获的异常
            retry_count: 当前重试次数
            max_retries: 最大重试次数
            
        Returns:
            bool: 是否应该重试
        """
        pass
    
    @abstractmethod
    def get_retry_delay(self, retry_count: int, base_delay: int) -> float:
        """
        计算重试延迟时间
        
        Args:
            retry_count: 当前重试次数
            base_delay: 基础延迟时间（秒）
            
        Returns:
            float: 重试延迟时间（秒）
        """
        pass
    
    @abstractmethod
    def log_retry_attempt(self, logger: logging.Logger, exception: Exception, 
                         retry_count: int, max_retries: int, delay: float) -> None:
        """
        记录重试尝试
        
        Args:
            logger: 日志记录器
            exception: 捕获的异常
            retry_count: 当前重试次数
            max_retries: 最大重试次数
            delay: 重试延迟时间
        """
        pass
    
    @abstractmethod
    def on_retry(self, exception: Exception, retry_count: int) -> None:
        """
        重试前的回调函数，可以执行额外操作
        
        Args:
            exception: 捕获的异常
            retry_count: 当前重试次数
        """
        pass


class ExponentialBackoffStrategy(RetryStrategy):
    """指数退避重试策略，适用于连接错误"""
    
    def should_retry(self, exception: Exception, retry_count: int, max_retries: int) -> bool:
        return (isinstance(exception, requests.exceptions.ConnectionError) and 
                retry_count < max_retries)
    
    def get_retry_delay(self, retry_count: int, base_delay: int) -> float:
        # 指数退避: base_delay * 2^(retry_count)
        return base_delay * (2 ** retry_count)
    
    def log_retry_attempt(self, logger: logging.Logger, exception: Exception,
                         retry_count: int, max_retries: int, delay: float) -> None:
        # 检查logger是否为函数对象（如client._log）
        if callable(logger) and not isinstance(logger, logging.Logger):
            # 如果是函数，直接调用它
            logger(f"连接错误，{delay:.1f}秒后重试 ({retry_count}/{max_retries}): {exception}", "WARNING")
        else:
            # 如果是Logger对象，调用warning方法
            logger.warning(f"连接错误，{delay:.1f}秒后重试 ({retry_count}/{max_retries}): {exception}")
    
    def on_retry(self, exception: Exception, retry_count: int) -> None:
        # 连接错误不需要额外操作
        pass


class LinearBackoffStrategy(RetryStrategy):
    """线性退避重试策略，适用于超时错误"""
    
    def should_retry(self, exception: Exception, retry_count: int, max_retries: int) -> bool:
        return (isinstance(exception, requests.exceptions.Timeout) and 
                retry_count < max_retries)
    
    def get_retry_delay(self, retry_count: int, base_delay: int) -> float:
        # 线性退避: base_delay * retry_count
        return base_delay * retry_count
    
    def log_retry_attempt(self, logger: logging.Logger, exception: Exception,
                         retry_count: int, max_retries: int, delay: float) -> None:
        # 检查logger是否为函数对象（如client._log）
        if callable(logger) and not isinstance(logger, logging.Logger):
            # 如果是函数，直接调用它
            logger(f"请求超时，{delay:.1f}秒后重试 ({retry_count}/{max_retries}): {exception}", "WARNING")
        else:
            # 如果是Logger对象，调用warning方法
            logger.warning(f"请求超时，{delay:.1f}秒后重试 ({retry_count}/{max_retries}): {exception}")
    
    def on_retry(self, exception: Exception, retry_count: int) -> None:
        # 超时错误不需要额外操作
        pass


class ServerErrorStrategy(RetryStrategy):
    """服务器错误重试策略，适用于5xx错误"""
    
    def should_retry(self, exception: Exception, retry_count: int, max_retries: int) -> bool:
        if not isinstance(exception, requests.exceptions.HTTPError):
            return False
        
        response = getattr(exception, 'response', None)
        if response is None:
            return False
        
        return (500 <= response.status_code < 600 and retry_count < max_retries)
    
    def get_retry_delay(self, retry_count: int, base_delay: int) -> float:
        # 线性退避: base_delay * retry_count
        return base_delay * retry_count
    
    def log_retry_attempt(self, logger: logging.Logger, exception: Exception,
                         retry_count: int, max_retries: int, delay: float) -> None:
        response = getattr(exception, 'response', None)
        status_code = response.status_code if response else 'unknown'
        # 检查logger是否为函数对象（如client._log）
        if callable(logger) and not isinstance(logger, logging.Logger):
            # 如果是函数，直接调用它
            logger(f"服务器错误 {status_code}，{delay:.1f}秒后重试 ({retry_count}/{max_retries})", "WARNING")
        else:
            # 如果是Logger对象，调用warning方法
            logger.warning(f"服务器错误 {status_code}，{delay:.1f}秒后重试 ({retry_count}/{max_retries})")
    
    def on_retry(self, exception: Exception, retry_count: int) -> None:
        # 服务器错误不需要额外操作
        pass


class RateLimitStrategy(RetryStrategy):
    """速率限制重试策略，适用于429错误，包括账号切换逻辑和延迟重试"""
    
    def __init__(self, client=None):
        """
        初始化速率限制重试策略
        
        Args:
            client: API客户端实例，用于切换账号
        """
        self.client = client
        self.consecutive_429_count = 0  # 连续429错误计数器
    
    def should_retry(self, exception: Exception, retry_count: int, max_retries: int) -> bool:
        if not isinstance(exception, requests.exceptions.HTTPError):
            return False
        
        response = getattr(exception, 'response', None)
        if response is None:
            return False
        
        is_rate_limit = response.status_code == 429
        if is_rate_limit:
            self.consecutive_429_count += 1
        else:
            self.consecutive_429_count = 0  # 重置计数器
            
        return is_rate_limit
    
    def get_retry_delay(self, retry_count: int, base_delay: int) -> float:
        # 根据用户反馈，429错误时不需要延迟，立即重试
        return 0
    
    def log_retry_attempt(self, logger: logging.Logger, exception: Exception,
                         retry_count: int, max_retries: int, delay: float) -> None:
        # 检查logger是否为函数对象（如client._log）
        message = ""
        if self.consecutive_429_count > 1:
            message = f"连续第{self.consecutive_429_count}次速率限制错误，尝试立即重试"
        else:
            message = "速率限制错误，尝试切换账号"
            
        if callable(logger) and not isinstance(logger, logging.Logger):
            # 如果是函数，直接调用它
            logger(message, "WARNING")
        else:
            # 如果是Logger对象，调用warning方法
            logger.warning(message)
    
    def on_retry(self, exception: Exception, retry_count: int) -> None:
        # 新增: 获取关联信息
        user_identifier = getattr(self.client, '_associated_user_identifier', None)
        request_ip = getattr(self.client, '_associated_request_ip', None) # request_ip 可能在某些情况下需要

        # 只有在首次429错误或账号池中有多个账号时才切换账号
        if self.consecutive_429_count == 1 or (self.consecutive_429_count > 0 and self.consecutive_429_count % 3 == 0):
            if self.client and hasattr(self.client, 'email'):
                # 记录当前账号进入冷却期
                current_email = self.client.email # 这是切换前的 email
                config.set_account_cooldown(current_email)
                
                # 获取新账号
                new_email, new_password = config.get_next_ondemand_account_details()
                if new_email:
                    # 更新客户端信息
                    self.client.email = new_email # 这是切换后的 email
                    self.client.password = new_password
                    self.client.token = ""
                    self.client.refresh_token = ""
                    self.client.session_id = ""  # 重置会话ID，确保创建新会话
                    
                    # 尝试使用新账号登录并创建会话
                    try:
                        # 获取当前请求的上下文哈希，以便在切换账号后重新登录和创建会话时使用
                        current_context_hash = getattr(self.client, '_current_request_context_hash', None)
                        
                        self.client.sign_in(context=current_context_hash)
                        if self.client.create_session(external_context=current_context_hash):
                            # 如果成功登录并创建会话，记录日志并设置标志位
                            if hasattr(self.client, '_log'):
                                self.client._log(f"成功切换到账号 {new_email} 并使用上下文哈希 '{current_context_hash}' 重新登录和创建新会话。", "INFO")
                            # 设置标志位，通知调用方下次需要发送完整历史
                            setattr(self.client, '_new_session_requires_full_history', True)
                            if hasattr(self.client, '_log'):
                                self.client._log(f"已设置 _new_session_requires_full_history = True，下次查询应发送完整历史。", "INFO")
                        else:
                            # 会话创建失败，记录错误
                            if hasattr(self.client, '_log'):
                                self.client._log(f"切换到账号 {new_email} 后，创建新会话失败。", "WARNING")
                                # 确保在这种情况下不设置需要完整历史的标志，因为会话本身就没成功
                                setattr(self.client, '_new_session_requires_full_history', False)


                        # --- 新增: 更新 client_sessions ---
                        if not user_identifier:
                            if hasattr(self.client, '_log'):
                                self.client._log("RateLimitStrategy: _associated_user_identifier not found on client. Cannot update client_sessions.", "ERROR")
                            # 即使没有 user_identifier，账号切换和会话创建也已发生，只是无法更新全局会话池
                        else:
                            old_email_in_strategy = current_email # 切换前的 email
                            new_email_in_strategy = self.client.email # 切换后的 email (即 new_email)

                            with config.config_instance.client_sessions_lock:
                                if user_identifier in config.config_instance.client_sessions:
                                    user_specific_sessions = config.config_instance.client_sessions[user_identifier]

                                    # 1. 移除旧 email 的条目 (如果存在)
                                    #    我们只移除那些 client 实例确实是当前 self.client 的条目，
                                    #    或者更简单地，如果旧 email 存在，就移除它，因为 user_identifier
                                    #    现在应该通过 new_email 使用这个（已被修改的）client 实例。
                                    if old_email_in_strategy in user_specific_sessions:
                                        # 检查 client 实例是否匹配可能不可靠，因为 client 内部状态已变。
                                        # 直接删除旧 email 的条目，因为这个 user_identifier + client 组合现在用新 email。
                                        del user_specific_sessions[old_email_in_strategy]
                                        if hasattr(self.client, '_log'):
                                            self.client._log(f"RateLimitStrategy: Removed session for old email '{old_email_in_strategy}' for user '{user_identifier}'.", "INFO")
                                    
                                    # 2. 添加/更新新 email 的条目
                                    #    确保它指向当前这个已被修改的 self.client 实例
                                    #    并重置 active_context_hash。
                                    #    IP 地址应来自 self.client._associated_request_ip 或 routes.py 中设置的值。
                                    #    由于 routes.py 在创建/分配会话时已将 IP 存入 client_sessions，
                                    #    这里我们主要关注 client 实例和 active_context_hash。
                                    #    如果 request_ip 在 self.client 中可用，则使用它，否则尝试保留已有的。
                                    ip_to_use = request_ip if request_ip else user_specific_sessions.get(new_email_in_strategy, {}).get("ip", "unknown_ip_in_retry_update")
                                    
                                    # 从 client 实例获取原始请求的上下文哈希
                                    # 这个哈希应该由 routes.py 在调用 send_query 之前设置到 client 实例上
                                    active_hash_for_new_session = getattr(self.client, '_current_request_context_hash', None)

                                    user_specific_sessions[new_email_in_strategy] = {
                                        "client": self.client, # 关键: 指向当前更新了 email/session_id 的 client 实例
                                        "active_context_hash": active_hash_for_new_session, # 使用来自 client 实例的哈希
                                        "last_time": datetime.now(), # 更新时间
                                        "ip": ip_to_use
                                    }
                                    log_message_hash_part = f"set to '{active_hash_for_new_session}' (from client instance's _current_request_context_hash)" if active_hash_for_new_session is not None else "set to None (_current_request_context_hash not found on client instance)"
                                    if hasattr(self.client, '_log'):
                                        self.client._log(f"RateLimitStrategy: Updated/added session for new email '{new_email_in_strategy}' for user '{user_identifier}'. active_context_hash {log_message_hash_part}.", "INFO")
                                else:
                                    if hasattr(self.client, '_log'):
                                        self.client._log(f"RateLimitStrategy: User '{user_identifier}' not found in client_sessions during update attempt.", "WARNING")
                        # --- 更新 client_sessions 结束 ---

                    except Exception as e:
                        # 登录或创建会话失败，记录错误但不抛出异常
                        # 让后续的重试机制处理
                        if hasattr(self.client, '_log'):
                            self.client._log(f"切换到账号 {new_email} 后登录或创建会话失败: {e}", "WARNING")
                            # 此处不应更新 client_sessions，因为新账号的会话未成功建立


class RetryHandler:
    """重试处理器，管理多个重试策略"""
    
    def __init__(self, client=None, logger=None):
        """
        初始化重试处理器
        
        Args:
            client: API客户端实例，用于切换账号
            logger: 日志记录器或日志函数
        """
        self.client = client
        # 如果logger是None，使用默认logger
        # 如果logger是函数或Logger对象，直接使用
        self.logger = logger or logging.getLogger(__name__)
        self.strategies = [
            ExponentialBackoffStrategy(),
            LinearBackoffStrategy(),
            ServerErrorStrategy(),
            RateLimitStrategy(client)
        ]
    
    def retry_operation(self, operation: Callable[..., T], *args, **kwargs) -> T:
        """
        使用重试策略执行操作
        
        Args:
            operation: 要执行的操作
            *args: 操作的位置参数
            **kwargs: 操作的关键字参数
            
        Returns:
            操作的结果
            
        Raises:
            Exception: 如果所有重试都失败，则抛出最后一个异常
        """
        max_retries = config.get_config_value('max_retries')
        base_delay = config.get_config_value('retry_delay')
        retry_count = 0
        last_exception = None
        
        while True:
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                # 查找适用的重试策略
                strategy = next((s for s in self.strategies if s.should_retry(e, retry_count, max_retries)), None)
                
                if strategy:
                    retry_count += 1
                    delay = strategy.get_retry_delay(retry_count, base_delay)
                    strategy.log_retry_attempt(self.logger, e, retry_count, max_retries, delay)
                    strategy.on_retry(e, retry_count)
                    
                    if delay > 0:
                        time.sleep(delay)
                else:
                    # 没有适用的重试策略，或者已达到最大重试次数
                    raise


def with_retry(max_retries: Optional[int] = None, retry_delay: Optional[int] = None):
    """
    重试装饰器，用于装饰需要重试的方法
    
    Args:
        max_retries: 最大重试次数，如果为None则使用配置值
        retry_delay: 基础重试延迟，如果为None则使用配置值
        
    Returns:
        装饰后的函数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 获取配置值
            _max_retries = max_retries or config.get_config_value('max_retries')
            _retry_delay = retry_delay or config.get_config_value('retry_delay')
            
            # 创建重试处理器
            handler = RetryHandler(client=self, logger=getattr(self, '_log', None))
            
            # 定义要重试的操作
            def operation():
                return func(self, *args, **kwargs)
            
            # 执行操作并处理重试
            return handler.retry_operation(operation)
        
        return wrapper
    
    return decorator