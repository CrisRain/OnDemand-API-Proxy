import json
import os
import time
import tiktoken
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from logging_config import logger
from config import get_config_value  # 导入配置获取函数
# from config import get_config_value # 导入配置获取函数 - 已移除循环导入



def mask_email(email: str) -> str:
    """隐藏邮箱中间部分，保护隐私"""
    if not email or '@' not in email:
        return "无效邮箱"
    
    parts = email.split('@')
    username = parts[0]
    domain = parts[1]
    
    if len(username) <= 3:
        masked_username = username[0] + '*' * (len(username) - 1)
    else:
        masked_username = username[0] + '*' * (len(username) - 2) + username[-1]
    
    return f"{masked_username}@{domain}"

def generate_request_id() -> str:
    """生成唯一的请求ID"""
    return f"chatcmpl-{os.urandom(16).hex()}"

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    计算文本的token数量

    Args:
        text: 要计算token数量的文本
        model: 模型名称，默认为gpt-3.5-turbo
        
    Returns:
        int: token数量
    """
    # 类型保护，防止text为None或非字符串类型
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    try:
        # 根据模型名称获取编码器
        if "gpt-4" in model:
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        elif "claude" in model:
            # Claude模型使用cl100k_base编码器
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # 默认使用cl100k_base编码器
            encoding = tiktoken.get_encoding("cl100k_base")
        
        # 计算token数量
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        logger.error(f"计算token数量时出错: {e}")
        # 如果出错，使用简单的估算方法（每4个字符约为1个token）
        return len(text) // 4

def count_message_tokens(messages: list, model: str = "gpt-3.5-turbo") -> Tuple[int, int, int]:
    """
    计算OpenAI格式消息列表的token数量
    
    Args:
        messages: OpenAI格式的消息列表
        model: 模型名称，默认为gpt-3.5-turbo
        
    Returns:
        Tuple[int, int, int]: (提示tokens数, 完成tokens数, 总tokens数)
    """
    # 类型保护，防止messages为None或非列表类型
    if messages is None:
        messages = []
    elif not isinstance(messages, list):
        logger.warning(f"count_message_tokens 收到非列表类型的消息: {type(messages)}")
        messages = []
    
    prompt_tokens = 0
    completion_tokens = 0
    
    try:
        # 计算提示tokens
        for message in messages:
            # 确保message是字典类型
            if not isinstance(message, dict):
                logger.warning(f"跳过非字典类型的消息: {type(message)}")
                continue
                
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role and content:
                # 每条消息的基本token开销
                prompt_tokens += 4  # 每条消息的基本开销
                
                # 角色名称的token
                prompt_tokens += 1  # 角色名称的开销
                
                # 内容的token
                prompt_tokens += count_tokens(content, model)
                
                # 如果是assistant角色，计算完成tokens
                if role == 'assistant':
                    completion_tokens += count_tokens(content, model)
        
        # 消息结束的token
        prompt_tokens += 2  # 消息结束的开销
        
        # 计算总tokens
        total_tokens = prompt_tokens + completion_tokens
        
        return prompt_tokens, completion_tokens, total_tokens
    except Exception as e:
        logger.error(f"计算消息token数量时出错: {e}")
        # 返回安全的默认值
        return 0, 0, 0