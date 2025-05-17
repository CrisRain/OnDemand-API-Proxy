import os
import json
import time
from collections import defaultdict
import threading
from typing import Dict, List, Any, Optional, Union, get_type_hints
from datetime import datetime, timedelta
from utils import logger, load_config


class Config:
    """配置管理类，用于存储和管理所有配置"""
    
    # 默认配置值
    _defaults = {
        "session_timeout_minutes": 3600,  # 会话不活动超时时间（分钟）- 增加以减少创建新会话的频率
        "max_retries": 5,  # 默认重试次数 - 增加以处理更多错误
        "retry_delay": 3,  # 默认重试延迟（秒）- 增加以减少请求频率
        "request_timeout": 45,  # 默认请求超时（秒）- 增加以允许更长的处理时间
        "stream_timeout": 180,  # 流式请求的默认超时（秒）- 增加以允许更长的处理时间
        "rate_limit": 30,  # 默认速率限制（每分钟请求数）- 减少以避免触发API速率限制
        "account_cooldown_seconds": 300,  # 账户冷却期（秒）- 在遇到429错误后暂时不使用该账户
        "debug_mode": False,  # 调试模式
        "api_access_token": "sk-2api-ondemand-access-token-2025",  # API访问认证Token
        "stats_file_path": "stats_data.json",  # 统计数据文件路径
        "stats_backup_path": "stats_data_backup.json",  # 统计数据备份文件路径
        "stats_save_interval": 300,  # 每5分钟保存一次统计数据
        "max_history_items": 1000,  # 最多保存的历史记录数量
        "default_endpoint_id": "predefined-claude-3.7-sonnet"  # 备用/默认端点 ID
    }
    
    # 模型名称映射：OpenAI 模型名 -> on-demand.io endpointId
    _model_mapping = {
        "gpt-3.5-turbo": "predefined-openai-gpto3-mini",
        "gpto3-mini": "predefined-openai-gpto3-mini",
        "gpt-4o": "predefined-openai-gpt4o",
        "gpt-4o-mini": "predefined-openai-gpt4o-mini",
        "gpt-4-turbo": "predefined-openai-gpt4.1",  # gpt-4.1 的别名
        "gpt-4.1": "predefined-openai-gpt4.1",
        "gpt-4.1-mini": "predefined-openai-gpt4.1-mini",
        "gpt-4.1-nano": "predefined-openai-gpt4.1-nano",
        "deepseek-v3": "predefined-deepseek-v3",
        "deepseek-r1": "predefined-deepseek-r1",
        "claude-3.5-sonnet": "predefined-claude-3.5-sonnet",
        "claude-3.7-sonnet": "predefined-claude-3.7-sonnet",
        "claude-3-opus": "predefined-claude-3-opus",
        "claude-3-haiku": "predefined-claude-3-haiku",
        "gemini-1.5-pro": "predefined-gemini-2.0-flash",
        "gemini-2.0-flash": "predefined-gemini-2.0-flash",
        # 根据需要添加更多映射
    }
    
    def __init__(self):
        """初始化配置对象"""
        # 从默认值初始化配置
        self._config = self._defaults.copy()
        
        # 用量统计
        self.usage_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "model_usage": defaultdict(int),  # 模型使用次数
            "account_usage": defaultdict(int),  # 账户使用次数
            "daily_usage": defaultdict(int),  # 每日使用次数
            "hourly_usage": defaultdict(int),  # 每小时使用次数
            "request_history": [],  # 请求历史记录
            "total_prompt_tokens": 0,  # 总提示tokens
            "total_completion_tokens": 0,  # 总完成tokens
            "total_tokens": 0,  # 总tokens
            "model_tokens": defaultdict(int),  # 每个模型的tokens使用量
            "daily_tokens": defaultdict(int),  # 每日tokens使用量
            "hourly_tokens": defaultdict(int),  # 每小时tokens使用量
            "last_saved": datetime.now().isoformat()  # 最后保存时间
        }
        
        # 线程锁
        self.usage_stats_lock = threading.Lock()  # 用于线程安全的统计数据访问
        self.account_index_lock = threading.Lock()  # 用于线程安全的账户选择
        self.client_sessions_lock = threading.Lock()  # 用于线程安全的会话管理
        
        # 当前账户索引（用于创建新客户端会话时的轮询选择）
        self.current_account_index = 0
        
        # 内存中存储每个客户端的会话和最后交互时间
        # 格式: {用户标识符: {账户邮箱: {"client": OnDemandAPIClient实例, "last_time": datetime对象}}}
        # 这样确保不同用户的会话是隔离的，每个用户只能访问自己的会话
        self.client_sessions = {}
        
        # 账户信息
        self.accounts = []
        
        # 账户冷却期记录 - 存储因速率限制而暂时不使用的账户
        # 格式: {账户邮箱: 冷却期结束时间(datetime对象)}
        self.account_cooldowns = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        self._config[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """批量更新配置值"""
        self._config.update(config_dict)
    
    def get_model_endpoint(self, model_name: str) -> str:
        """获取模型对应的端点ID"""
        return self._model_mapping.get(model_name, self.get("default_endpoint_id"))
    
    def load_from_file(self) -> bool:
        """从配置文件加载配置"""
        try:
            # utils.load_config() 当前不接受 file_path 参数，因此移除
            config_data = load_config()
            if config_data:
                # 更新配置
                for key, value in config_data.items():
                    if key != "accounts":  # 账户信息单独处理
                        self.set(key, value)
                
                # 处理账户信息
                if "accounts" in config_data:
                    self.accounts = config_data["accounts"]
                
                logger.info("已从配置文件加载配置")
                return True
            return False
        except Exception as e:
            logger.error(f"加载配置文件时出错: {e}")
            return False
    
    def load_from_env(self) -> None:
        """从环境变量加载配置"""
        # 从环境变量加载账户信息
        if not self.accounts:
            accounts_env = os.getenv("ONDEMAND_ACCOUNTS", "")
            if accounts_env:
                try:
                    self.accounts = json.loads(accounts_env).get('accounts', [])
                    logger.info("已从环境变量加载账户信息")
                except json.JSONDecodeError:
                    logger.error("解码 ONDEMAND_ACCOUNTS 环境变量失败")
        
        # 从环境变量加载其他设置
        env_mappings = {
            "session_timeout_minutes": "SESSION_TIMEOUT_MINUTES",
            "max_retries": "MAX_RETRIES",
            "retry_delay": "RETRY_DELAY",
            "request_timeout": "REQUEST_TIMEOUT",
            "stream_timeout": "STREAM_TIMEOUT",
            "rate_limit": "RATE_LIMIT",
            "debug_mode": "DEBUG_MODE",
            "api_access_token": "API_ACCESS_TOKEN"
        }
        
        for config_key, env_key in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                # 根据默认值的类型进行转换
                default_value = self.get(config_key)
                if isinstance(default_value, bool):
                    self.set(config_key, env_value.lower() == 'true')
                elif isinstance(default_value, int):
                    self.set(config_key, int(env_value))
                elif isinstance(default_value, float):
                    self.set(config_key, float(env_value))
                else:
                    self.set(config_key, env_value)

    def save_stats_to_file(self):
        """将统计数据保存到文件中"""
        try:
            with self.usage_stats_lock:
                # 创建统计数据的副本
                stats_copy = {
                    "total_requests": self.usage_stats["total_requests"],
                    "successful_requests": self.usage_stats["successful_requests"],
                    "failed_requests": self.usage_stats["failed_requests"],
                    "model_usage": dict(self.usage_stats["model_usage"]),
                    "account_usage": dict(self.usage_stats["account_usage"]),
                    "daily_usage": dict(self.usage_stats["daily_usage"]),
                    "hourly_usage": dict(self.usage_stats["hourly_usage"]),
                    "request_history": list(self.usage_stats["request_history"]),
                    "total_prompt_tokens": self.usage_stats["total_prompt_tokens"],
                    "total_completion_tokens": self.usage_stats["total_completion_tokens"],
                    "total_tokens": self.usage_stats["total_tokens"],
                    "model_tokens": dict(self.usage_stats["model_tokens"]),
                    "daily_tokens": dict(self.usage_stats["daily_tokens"]),
                    "hourly_tokens": dict(self.usage_stats["hourly_tokens"]),
                    "last_saved": datetime.now().isoformat()
                }
                
                stats_file_path = self.get("stats_file_path")
                stats_backup_path = self.get("stats_backup_path")
                
                # 先保存到备份文件，然后重命名，避免写入过程中的文件损坏
                with open(stats_backup_path, 'w', encoding='utf-8') as f:
                    json.dump(stats_copy, f, ensure_ascii=False, indent=2)
                
                # 如果主文件存在，先删除它
                if os.path.exists(stats_file_path):
                    os.remove(stats_file_path)
                
                # 将备份文件重命名为主文件
                os.rename(stats_backup_path, stats_file_path)
                
                logger.info(f"统计数据已保存到 {stats_file_path}")
                self.usage_stats["last_saved"] = datetime.now().isoformat()
        except Exception as e:
            logger.error(f"保存统计数据时出错: {e}")

    def load_stats_from_file(self):
        """从文件中加载统计数据"""
        try:
            stats_file_path = self.get("stats_file_path")
            if os.path.exists(stats_file_path):
                with open(stats_file_path, 'r', encoding='utf-8') as f:
                    saved_stats = json.load(f)
                
                with self.usage_stats_lock:
                    # 更新基本计数器
                    self.usage_stats["total_requests"] = saved_stats.get("total_requests", 0)
                    self.usage_stats["successful_requests"] = saved_stats.get("successful_requests", 0)
                    self.usage_stats["failed_requests"] = saved_stats.get("failed_requests", 0)
                    self.usage_stats["total_prompt_tokens"] = saved_stats.get("total_prompt_tokens", 0)
                    self.usage_stats["total_completion_tokens"] = saved_stats.get("total_completion_tokens", 0)
                    self.usage_stats["total_tokens"] = saved_stats.get("total_tokens", 0)
                    
                    # 更新字典类型的统计数据
                    for model, count in saved_stats.get("model_usage", {}).items():
                        self.usage_stats["model_usage"][model] = count
                    
                    for account, count in saved_stats.get("account_usage", {}).items():
                        self.usage_stats["account_usage"][account] = count
                    
                    for day, count in saved_stats.get("daily_usage", {}).items():
                        self.usage_stats["daily_usage"][day] = count
                    
                    for hour, count in saved_stats.get("hourly_usage", {}).items():
                        self.usage_stats["hourly_usage"][hour] = count
                    
                    for model, tokens in saved_stats.get("model_tokens", {}).items():
                        self.usage_stats["model_tokens"][model] = tokens
                    
                    for day, tokens in saved_stats.get("daily_tokens", {}).items():
                        self.usage_stats["daily_tokens"][day] = tokens
                    
                    for hour, tokens in saved_stats.get("hourly_tokens", {}).items():
                        self.usage_stats["hourly_tokens"][hour] = tokens
                    
                    # 更新请求历史
                    self.usage_stats["request_history"] = saved_stats.get("request_history", [])
                    
                    # 限制历史记录数量
                    max_history_items = self.get("max_history_items")
                    if len(self.usage_stats["request_history"]) > max_history_items:
                        self.usage_stats["request_history"] = self.usage_stats["request_history"][-max_history_items:]
                
                logger.info(f"已从 {stats_file_path} 加载统计数据")
                return True
            else:
                logger.info(f"未找到统计数据文件 {stats_file_path}，将使用默认值")
                return False
        except Exception as e:
            logger.error(f"加载统计数据时出错: {e}")
            return False

    def start_stats_save_thread(self):
        """启动定期保存统计数据的线程"""
        def save_stats_periodically():
            while True:
                time.sleep(self.get("stats_save_interval"))
                self.save_stats_to_file()
        
        save_thread = threading.Thread(target=save_stats_periodically, daemon=True)
        save_thread.start()
        logger.info(f"统计数据保存线程已启动，每 {self.get('stats_save_interval')} 秒保存一次")

    def init(self):
        """初始化配置，从配置文件或环境变量加载设置"""
        # 从配置文件加载配置
        self.load_from_file()
        
        # 从环境变量加载配置
        self.load_from_env()
        
        # 验证账户信息
        if not self.accounts:
            error_msg = "在 config.json 或环境变量 ONDEMAND_ACCOUNTS 中未找到账户信息"
            logger.critical(error_msg)
            # 不抛出异常，而是继续运行
            logger.warning("将继续运行，但没有账户信息，可能会导致功能受限")
        
        logger.info("已加载API访问Token")
        
        # 加载之前保存的统计数据
        self.load_stats_from_file()
        
        # 启动定期保存统计数据的线程
        self.start_stats_save_thread()

    def get_next_ondemand_account_details(self):
        """获取下一个 OnDemand 账户的邮箱和密码，用于轮询。
        会跳过处于冷却期的账户。"""
        with self.account_index_lock:
            current_time = datetime.now()
            
            # 清理过期的冷却记录
            expired_cooldowns = [email for email, end_time in self.account_cooldowns.items()
                               if end_time < current_time]
            for email in expired_cooldowns:
                del self.account_cooldowns[email]
                logger.info(f"账户 {email} 的冷却期已结束，现在可用")
            
            # 尝试最多len(self.accounts)次，以找到一个不在冷却期的账户
            for _ in range(len(self.accounts)):
                account_details = self.accounts[self.current_account_index]
                email = account_details.get('email')
                
                # 更新索引到下一个账户，为下次调用做准备
                self.current_account_index = (self.current_account_index + 1) % len(self.accounts)
                
                # 检查账户是否在冷却期
                if email in self.account_cooldowns:
                    cooldown_end = self.account_cooldowns[email]
                    remaining_seconds = (cooldown_end - current_time).total_seconds()
                    logger.warning(f"账户 {email} 仍在冷却期中，还剩 {remaining_seconds:.1f} 秒")
                    continue  # 尝试下一个账户
                
                # 找到一个可用账户
                logger.info(f"[系统] 新会话将使用账户: {email}")
                return email, account_details.get('password')
            
            # 如果所有账户都在冷却期，使用第一个账户（即使它在冷却期）
            logger.warning("所有账户都在冷却期！使用第一个账户，尽管它可能会触发速率限制")
            account_details = self.accounts[0]
            return account_details.get('email'), account_details.get('password')


# 创建全局配置实例
config_instance = Config()

def init_config():
    """初始化配置的兼容函数，用于向后兼容"""
    config_instance.init()


def get_config_value(name: str, default: Any = None) -> Any:
    """
    获取当前配置变量的最新值。
    推荐外部通过 config.get_config_value('变量名') 获取配置。
    对于 accounts, model_mapping, usage_stats, client_sessions，请使用新增的专用getter函数。
    """
    return config_instance.get(name, default)

# 新增的类型安全的getter函数
def get_accounts() -> List[Dict[str, str]]:
    """获取账户信息列表"""
    return config_instance.accounts

def get_model_mapping() -> Dict[str, str]:
    """获取模型名称到端点ID的映射"""
    return config_instance._model_mapping

def get_usage_stats() -> Dict[str, Any]:
    """获取用量统计数据"""
    return config_instance.usage_stats

def get_client_sessions() -> Dict[str, Any]:
    """获取客户端会话信息"""
    return config_instance.client_sessions

def get_next_ondemand_account_details():
    """获取下一个账户的兼容函数"""
    return config_instance.get_next_ondemand_account_details()

def set_account_cooldown(email, cooldown_seconds=None):
    """设置账户冷却期
    
    Args:
        email: 账户邮箱
        cooldown_seconds: 冷却时间（秒），如果为None则使用默认配置
    """
    if cooldown_seconds is None:
        cooldown_seconds = config_instance.get('account_cooldown_seconds')
    
    cooldown_end = datetime.now() + timedelta(seconds=cooldown_seconds)
    with config_instance.account_index_lock:  # 使用相同的锁保护冷却期字典
        config_instance.account_cooldowns[email] = cooldown_end
        logger.warning(f"账户 {email} 已设置冷却期 {cooldown_seconds} 秒，将于 {cooldown_end.strftime('%Y-%m-%d %H:%M:%S')} 结束")


# ⚠️ 警告：为保证配置动态更新，请勿使用 from config import XXX，只使用 import config 并通过 config.get_config_value('变量名') 获取配置。
# 这样可确保配置值始终是最新的。
# (｡•ᴗ-)ﾉﾞ 你的聪明小助手温馨提示~