import json
import time
import uuid
import html
import hashlib # Added import
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import request, Response, stream_with_context, jsonify, render_template, redirect, url_for, flash
from datetime import datetime

from utils import logger, generate_request_id, count_tokens, count_message_tokens
import config
from auth import RateLimiter
from client import OnDemandAPIClient
from datetime import timedelta

# 初始化速率限制器
# rate_limiter 将在 config_instance 定义后初始化

# 获取配置实例
config_instance = config.config_instance
rate_limiter = RateLimiter(config_instance.get('rate_limit_per_minute', 60))  # 从配置读取，默认为60

# 模型价格配置将从 config_instance 获取
# 默认价格也将从 config_instance 获取

def format_datetime(timestamp):
    """将ISO格式时间戳格式化为更易读的格式"""
    if not timestamp or timestamp == "从未保存":
        return timestamp
    
    try:
        # 处理ISO格式时间戳
        if 'T' in timestamp:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        # 处理已经是格式化字符串的情况
        return timestamp
    except Exception:
        return timestamp

def format_number(value):
    """根据数值大小自动转换单位"""
    if value is None or value == '-':
        return '-'
    
    try:
        value = float(value)
        if value >= 1000000000000:  # 万亿 (T)
            return f"{value/1000000000000:.2f}T"
        elif value >= 1000000000:  # 十亿 (G)
            return f"{value/1000000000:.2f}G"
        elif value >= 1000000:  # 百万 (M)
            return f"{value/1000000:.2f}M"
        elif value >= 1000:  # 千 (K)
            return f"{value/1000:.2f}K"
        elif value == 0:  # 零
            return "0"
        elif abs(value) < 0.01:  # 非常小的数值，使用科学计数法
            return f"{value:.2e}"
        else:
            return f"{value:.0f}" if value == int(value) else f"{value:.2f}"
    except (ValueError, TypeError):
        return str(value)

def format_duration(ms):
    """将毫秒格式化为更易读的格式"""
    if ms is None or ms == '-':
        return '-'
    
    try:
        ms = float(ms)  # 使用float而不是int，以支持小数
        if ms >= 86400000:  # 超过1天 (24*60*60*1000)
            return f"{ms/86400000:.2f}天"
        elif ms >= 3600000:  # 超过1小时 (60*60*1000)
            return f"{ms/3600000:.2f}小时"
        elif ms >= 60000:  # 超过1分钟 (60*1000)
            return f"{ms/60000:.2f}分钟"
        elif ms >= 1000:  # 超过1秒
            return f"{ms/1000:.2f}秒"
        else:
            return f"{ms:.0f}" if ms == int(ms) else f"{ms:.2f}毫秒"
    except (ValueError, TypeError):
        return str(ms)

def _update_usage_statistics(
    config_inst,
    request_id: str,
    requested_model_name: str,
    account_email: Optional[str],
    is_success: bool,
    duration_ms: int,
    is_stream: bool,
    prompt_tokens_val: int,
    completion_tokens_val: int,
    total_tokens_val: int,
    prompt_length: Optional[int] = None,
    completion_length: Optional[int] = None,
    error_message: Optional[str] = None,
    used_actual_tokens_for_history: bool = False
):
    """更新使用统计与请求历史的辅助函数。"""
    with config_inst.usage_stats_lock:
        config_inst.usage_stats["total_requests"] += 1
        
        current_email_for_stats = account_email if account_email else "unknown_account"

        if is_success:
            config_inst.usage_stats["successful_requests"] += 1
            config_inst.usage_stats["model_usage"].setdefault(requested_model_name, 0)
            config_inst.usage_stats["model_usage"][requested_model_name] += 1
            
            config_inst.usage_stats["account_usage"].setdefault(current_email_for_stats, 0)
            config_inst.usage_stats["account_usage"][current_email_for_stats] += 1

            config_inst.usage_stats["total_prompt_tokens"] += prompt_tokens_val
            config_inst.usage_stats["total_completion_tokens"] += completion_tokens_val
            config_inst.usage_stats["total_tokens"] += total_tokens_val
            config_inst.usage_stats["model_tokens"].setdefault(requested_model_name, 0)
            config_inst.usage_stats["model_tokens"][requested_model_name] += total_tokens_val

            today = datetime.now().strftime("%Y-%m-%d")
            hour = datetime.now().strftime("%Y-%m-%d %H:00")
            
            config_inst.usage_stats["daily_usage"].setdefault(today, 0)
            config_inst.usage_stats["daily_usage"][today] += 1
            
            config_inst.usage_stats["hourly_usage"].setdefault(hour, 0)
            config_inst.usage_stats["hourly_usage"][hour] += 1
            
            config_inst.usage_stats["daily_tokens"].setdefault(today, 0)
            config_inst.usage_stats["daily_tokens"][today] += total_tokens_val
            
            config_inst.usage_stats["hourly_tokens"].setdefault(hour, 0)
            config_inst.usage_stats["hourly_tokens"][hour] += total_tokens_val
        else:
            config_inst.usage_stats["failed_requests"] += 1

        history_entry = {
            "id": request_id,
            "timestamp": datetime.now().isoformat(),
            "model": requested_model_name,
            "account": current_email_for_stats,
            "success": is_success,
            "duration_ms": duration_ms,
            "stream": is_stream,
        }

        if is_success:
            if prompt_length is not None:
                history_entry["prompt_length"] = prompt_length
            if completion_length is not None:
                history_entry["completion_length"] = completion_length
            
            if is_stream:
                if used_actual_tokens_for_history:
                    history_entry["prompt_tokens"] = prompt_tokens_val
                    history_entry["completion_tokens"] = completion_tokens_val
                    history_entry["total_tokens"] = total_tokens_val
                else:
                    history_entry["prompt_tokens"] = prompt_tokens_val
                    history_entry["estimated_completion_tokens"] = completion_tokens_val
                    history_entry["estimated_total_tokens"] = total_tokens_val
            else: 
                history_entry["prompt_tokens"] = prompt_tokens_val
                history_entry["completion_tokens"] = completion_tokens_val
                history_entry["total_tokens"] = total_tokens_val
        else: 
            if error_message:
                history_entry["error"] = error_message
            if prompt_tokens_val > 0:
                 history_entry["prompt_tokens_attempted"] = prompt_tokens_val

        config_inst.usage_stats["request_history"].append(history_entry)
        max_history_items = config_inst.get('max_history_items', 1000)
        if len(config_inst.usage_stats["request_history"]) > max_history_items:
            config_inst.usage_stats["request_history"] = \
                config_inst.usage_stats["request_history"][-max_history_items:]

def _generate_hash_for_full_history(full_messages_list: List[Dict[str, str]], req_id: str) -> Optional[str]:
    """
    Generates a SHA256 hash from a list of messages, considering all messages.
    """
    if not full_messages_list:
        logger.debug(f"[{req_id}] (_generate_hash_for_full_history) No messages to hash.")
        return None
    try:
        # Ensure consistent serialization for hashing
        # Context meaning is only in role and content
        simplified_history = [{"role": msg.get("role"), "content": msg.get("content")} for msg in full_messages_list]
        serialized_history = json.dumps(simplified_history, sort_keys=True)
        return hashlib.sha256(serialized_history.encode('utf-8')).hexdigest()
    except (TypeError, ValueError) as e:
        logger.error(f"[{req_id}] (_generate_hash_for_full_history) Failed to serialize full history messages for hashing: {e}")
        return None

def _update_client_context_hash_after_reply(
    original_request_messages: List[Dict[str, str]],
    assistant_reply_content: str,
    request_id: str,
    user_identifier: str, # Corresponds to 'token' in chat_completions
    email_for_stats: Optional[str],
    current_ondemand_client_instance: Optional[OnDemandAPIClient],
    config_inst: config.Config,
    logger_instance # Pass logger directly
):
    """
    Helper to update the client's active_context_hash after a successful reply
    using the full conversation history up to the assistant's reply.
    """
    if not assistant_reply_content or not email_for_stats or not current_ondemand_client_instance:
        logger_instance.debug(f"[{request_id}] 更新客户端上下文哈希的条件不足（回复内容 '{bool(assistant_reply_content)}', 邮箱 '{email_for_stats}', 客户端实例 '{bool(current_ondemand_client_instance)}'），跳过。")
        return

    assistant_message = {"role": "assistant", "content": assistant_reply_content}
    # original_request_messages should be the messages list as it was when the request came in.
    full_history_up_to_assistant_reply = original_request_messages + [assistant_message]
    
    next_active_context_hash = _generate_hash_for_full_history(full_history_up_to_assistant_reply, request_id)
    
    if next_active_context_hash:
        with config_inst.client_sessions_lock:
            if user_identifier in config_inst.client_sessions and \
               email_for_stats in config_inst.client_sessions[user_identifier]:
                
                session_data_to_update = config_inst.client_sessions[user_identifier][email_for_stats]
                client_in_session = session_data_to_update.get("client")

                if client_in_session == current_ondemand_client_instance:
                    old_hash = session_data_to_update.get("active_context_hash")
                    session_data_to_update["active_context_hash"] = next_active_context_hash
                    session_data_to_update["last_time"] = datetime.now()
                    logger_instance.info(f"[{request_id}] 客户端 (账户: {email_for_stats}) 的 active_context_hash 已从 '{old_hash}' 更新为 '{next_active_context_hash}' 以反映对话进展。")
                else:
                    logger_instance.warning(f"[{request_id}] 尝试更新哈希时，发现 email_for_stats '{email_for_stats}' 对应的存储客户端与当前使用的 ondemand_client 不一致。跳过更新。")
            else:
                logger_instance.warning(f"[{request_id}] 尝试更新哈希时，在 client_sessions 中未找到用户 '{user_identifier}' 或账户 '{email_for_stats}'。跳过更新。")
    else:
        logger_instance.warning(f"[{request_id}] 未能为下一次交互生成新的 active_context_hash (基于回复 '{bool(assistant_reply_content)}'). 客户端的哈希未更新。")
 
def _get_context_key_from_messages(messages: List[Dict[str, str]], req_id: str) -> Optional[str]:
    """
    从末次用户消息前的消息列表生成上下文哈希密钥。
    """
    if not messages:
        logger.debug(f"[{req_id}] 无消息可供生成上下文密钥。")
        return None

    last_user_msg_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get('role') == 'user':
            last_user_msg_idx = i
            break
    
    # 若无用户消息或用户消息为首条，则无先前历史可生成上下文密钥。
    if last_user_msg_idx <= 0:
        logger.debug(f"[{req_id}] 无先前历史可生成上下文密钥 (last_user_msg_idx: {last_user_msg_idx})。")
        return None
    
    historical_messages = messages[:last_user_msg_idx]
    if not historical_messages: # 应由 last_user_msg_idx <= 0 捕获，此处为额外保障
        logger.debug(f"[{req_id}] 上下文密钥的历史消息列表为空。")
        return None

    try:
        # 确保哈希序列化的一致性
        # 上下文意义仅关注角色和内容
        simplified_history = [{"role": msg.get("role"), "content": msg.get("content")} for msg in historical_messages]
        serialized_history = json.dumps(simplified_history, sort_keys=True)
        return hashlib.sha256(serialized_history.encode('utf-8')).hexdigest()
    except (TypeError, ValueError) as e:
        logger.error(f"[{req_id}] 序列化历史消息以生成上下文密钥失败: {e}")
        return None

def register_routes(app):
    """注册所有路由到Flask应用"""
    
    # 注册自定义过滤器
    app.jinja_env.filters['format_datetime'] = format_datetime
    app.jinja_env.filters['format_number'] = format_number
    app.jinja_env.filters['format_duration'] = format_duration
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """健康检查端点，返回服务状态"""
        return {"status": "ok", "message": "2API服务运行正常"}, 200
    
    @app.route('/v1/models', methods=['GET'])
    def list_models():
        """以 OpenAI 格式返回可用模型列表。"""
        data = []
        # 获取当前时间戳，用于 'created' 字段
        created_time = int(time.time())
        model_mapping = config_instance._model_mapping
        for openai_name in model_mapping.keys():  # 仅列出已映射的模型
            data.append({
                "id": openai_name,
                "object": "model",
                "created": created_time,
                "owned_by": "on-demand.io"  # 或根据模型来源填写 "openai", "anthropic" 等
            })
        return {"object": "list", "data": data}
    
    @app.route('/v1/chat/completions', methods=['POST'])
    def chat_completions():
        """处理聊天补全请求，兼容 OpenAI 格式。"""
        request_id = generate_request_id()  # 生成唯一的请求 ID
        client_ip = request.remote_addr  # 获取客户端 IP 地址，仅用于日志记录
        logger.info(f"[{request_id}] 收到来自 IP: {client_ip} 的 /v1/chat/completions 请求")
        
        # 验证访问令牌
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.warning(f"[{request_id}] 未提供认证令牌或格式错误")
            return {"error": {"message": "缺少有效的认证令牌", "type": "auth_error", "code": "missing_token"}}, 401
        
        # 获取API访问令牌
        api_access_token = config_instance.get('api_access_token')
        token = auth_header[7:]  # 去掉 'Bearer ' 前缀
        if token != api_access_token:
            logger.warning(f"[{request_id}] 提供了无效的认证令牌")
            return {"error": {"message": "无效的认证令牌", "type": "auth_error", "code": "invalid_token"}}, 401

        # 检查速率限制 - 使用token而不是IP进行限制
        if not rate_limiter.is_allowed(token):
            logger.warning(f"[{request_id}] 用户 {token[:8]}... 超过速率限制")
            return {"error": {"message": "请求频率过高，请稍后再试", "type": "rate_limit_error", "code": "rate_limit_exceeded"}}, 429

        openai_data = request.get_json()
        if not openai_data:
            logger.error(f"[{request_id}] 请求体不是有效的JSON")
            return {"error": {"message": "请求体必须是 JSON。", "type": "invalid_request_error", "code": None}}, 400
        
        if app.config.get('DEBUG_MODE', False):
            logger.debug(f"[{request_id}] OpenAI 请求数据: {json.dumps(openai_data, indent=2, ensure_ascii=False)}")

        # 从 OpenAI 请求中提取参数
        # Capture the initial messages from the request for later use in rolling hash update
        initial_messages_from_request: List[Dict[str, str]] = openai_data.get('messages', [])
        messages: List[Dict[str, str]] = initial_messages_from_request # Keep 'messages' for existing logic
        stream_requested: bool = openai_data.get('stream', False)
        # 如果请求中没有指定模型，则使用映射表中的一个默认模型，或者最终的 DEFAULT_ENDPOINT_ID
        model_mapping = config_instance._model_mapping
        default_endpoint_id = config_instance.get('default_endpoint_id')
        requested_model_name: str = openai_data.get('model', list(model_mapping.keys())[0] if model_mapping else default_endpoint_id)
        
        # 从请求中获取参数，如果未提供则为 None
        temperature: Optional[float] = openai_data.get('temperature')
        max_tokens: Optional[int] = openai_data.get('max_tokens')
        top_p: Optional[float] = openai_data.get('top_p')
        frequency_penalty: Optional[float] = openai_data.get('frequency_penalty')
        presence_penalty: Optional[float] = openai_data.get('presence_penalty')

        if not messages:
            logger.error(f"[{request_id}] 缺少 'messages' 字段")
            return {"error": {"message": "缺少 'messages' 字段。", "type": "invalid_request_error", "code": "missing_messages"}}, 400

        # 为 on-demand.io 构建查询
        # on-demand.io 通常接受单个查询字符串，上下文由其会话管理。
        # 我们将发送最新的用户查询，可选地以系统提示为前缀。
        # --- 上下文感知会话管理与查询构建 (v2) ---

        # 1. 提取消息组件与上下文密钥
        request_context_hash = _get_context_key_from_messages(messages, request_id)
        logger.info(f"[{request_id}] 请求上下文哈希值: {request_context_hash}")

        historical_messages = [] # 末次用户消息前的消息，用于新上下文的密钥及查询构建
        if request_context_hash:
            last_user_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get('role') == 'user': last_user_idx = i; break
            # 此检查逻辑应与 _get_context_key_from_messages 中 historical_messages 的形成逻辑一致
            if last_user_idx > 0: historical_messages = messages[:last_user_idx]
        
        current_system_prompts_contents = [msg['content'] for msg in messages if msg.get('role') == 'system' and msg.get('content')]
        system_prompt_combined = "\n".join(current_system_prompts_contents)
        
        current_user_messages_contents = [msg['content'] for msg in messages if msg.get('role') == 'user' and msg.get('content')]
        current_user_query = current_user_messages_contents[-1] if current_user_messages_contents else ""

        if not current_user_query: # 此检查至关重要
            logger.error(f"[{request_id}] 'messages' 中未找到有效的 'user' 角色的消息内容。")
            # 记录调试消息
            logger.debug(f"[{request_id}] 接收到的消息: {json.dumps(messages, ensure_ascii=False)}")
            return {"error": {"message": "'messages' 中未找到有效的 'user' 角色的消息内容。", "type": "invalid_request_error", "code": "no_user_message"}}, 400
 
        user_identifier = token
        # 记录请求开始时间，确保在所有路径中 duration_ms 可用
        request_start_time = time.time()
        ondemand_client = None
        email_for_stats = None # 此为 OnDemandAPIClient 所用账户的邮箱
        is_newly_assigned_context = False # 若选定客户端首次处理此 context_hash 则为 True
 
        with config_instance.client_sessions_lock:
            current_time = datetime.now()
            if user_identifier not in config_instance.client_sessions:
                config_instance.client_sessions[user_identifier] = {}
            user_sessions_for_id = config_instance.client_sessions[user_identifier]

            # 阶段 1: 查找已服务此 context_hash 的客户端
            if request_context_hash: # 仅当历史记录中存在有效上下文哈希时才尝试匹配
                for acc_email_p1, session_data_p1 in user_sessions_for_id.items():
                    client_p1 = session_data_p1.get("client")
                    # 基本客户端有效性检查
                    if client_p1 and client_p1.token and client_p1.session_id and \
                       session_data_p1.get("active_context_hash") == request_context_hash:
                        
                        ondemand_client = client_p1
                        email_for_stats = acc_email_p1
                        session_data_p1["last_time"] = current_time
                        session_data_p1["ip"] = client_ip # Update IP as well
                        is_newly_assigned_context = False # 关键：此上下文已由此客户端服务
                        logger.info(f"[{request_id}] 上下文精确匹配。复用账户 {email_for_stats} 的客户端 (上下文哈希: {request_context_hash})。")
                        break # 已找到客户端
            
            # 阶段 2: 若无精确上下文匹配，则必须创建新客户端
            if not ondemand_client:
                logger.info(f"[{request_id}] 未找到精确上下文匹配 (上下文哈希: {request_context_hash or 'None'})。尝试获取/创建新客户端会话。")
                # 原重分配逻辑已移除，直接进入创建新客户端流程
                
                # 无现有客户端可重分配，须获取新账户或创建会话 (此处的逻辑现在是主要路径)
                # logger.info(f"[{request_id}] 无可重分配的客户端。尝试获取/创建新客户端会话。") # 此日志可移除或调整
                MAX_ACCOUNT_ATTEMPTS = 3 # 应为配置值
                for attempt in range(MAX_ACCOUNT_ATTEMPTS):
                        new_ondemand_email, new_ondemand_password = config.get_next_ondemand_account_details()
                        if not new_ondemand_email:
                            logger.error(f"[{request_id}] 尝试 {attempt+1} 次后，配置中无可用 OnDemand 账户。")
                            # 若所有尝试失败，则在循环外处理错误响应
                            break

                        email_for_stats = new_ondemand_email # 本次尝试暂设值
                        
                        # 检查 user_identifier 是否已对 new_ondemand_email 存在会话
                        # 若 get_next_ondemand_account_details 在小账户池循环，可能发生此情况
                        if new_ondemand_email in user_sessions_for_id:
                            existing_session_data_for_new_email = user_sessions_for_id[new_ondemand_email]
                            ondemand_client = existing_session_data_for_new_email["client"]
                            logger.info(f"[{request_id}] 账户 {new_ondemand_email} 已关联用户。尝试为新上下文重验证/分配 (尝试 {attempt+1})。")
                            # 确保有效后分配上下文
                            if not ondemand_client.token or not ondemand_client.session_id:
                                if not ondemand_client.sign_in() or not ondemand_client.create_session():
                                    logger.warning(f"[{request_id}] 为 {new_ondemand_email} 重验证会话失败: {ondemand_client.last_error}")
                                    ondemand_client = None; email_for_stats = None; continue # 尝试下一账户
                        else: # user_identifier 的全新账户，创建客户端
                            logger.info(f"[{request_id}] 为新账户 {new_ondemand_email} 创建新客户端 (尝试 {attempt+1})。")
                            client_id_for_log = f"{client_ip}-{new_ondemand_email.split('@')[0]}"
                            ondemand_client = OnDemandAPIClient(new_ondemand_email, new_ondemand_password, client_id=client_id_for_log)
                            if not ondemand_client.sign_in() or not ondemand_client.create_session():
                                logger.error(f"[{request_id}] 为 {new_ondemand_email} 初始化新客户端失败: {ondemand_client.last_error}")
                                ondemand_client = None; email_for_stats = None; continue # Try next account
                        
                        # 若客户端成功获取/验证
                        user_sessions_for_id[new_ondemand_email] = {
                            "client": ondemand_client, "last_time": current_time, "ip": client_ip,
                            "active_context_hash": request_context_hash # 分配至当前上下文
                        }
                        is_newly_assigned_context = True # 此客户端现服务于此(可能为新)上下文
                        logger.info(f"[{request_id}] 已为上下文哈希 {request_context_hash} 成功设置客户端 (账户: {email_for_stats})。")
                        break # 跳出账户尝试循环，客户端就绪
                
                if not ondemand_client: # 获取/创建客户端尝试均失败
                    logger.error(f"[{request_id}] 尝试 {MAX_ACCOUNT_ATTEMPTS} 次后获取/创建客户端失败。")
                    prompt_tok_val_err, _, _ = count_message_tokens(messages, requested_model_name)
                    _update_usage_statistics(
                        config_inst=config_instance, request_id=request_id, requested_model_name=requested_model_name,
                        account_email=email_for_stats, # 可能为最后尝试的邮箱或None
                        is_success=False, duration_ms=int((time.time() - request_start_time) * 1000), # request_start_time 可能未定义
                        is_stream=stream_requested, prompt_tokens_val=prompt_tok_val_err or 0,
                        completion_tokens_val=0, total_tokens_val=prompt_tok_val_err or 0,
                        error_message="多次尝试后获取/创建客户端会话失败。"
                    )
                    return {"error": {"message": "当前无法与 OnDemand 服务建立会话。", "type": "api_error", "code": "ondemand_session_unavailable"}}, 503

        # --- 会话管理结束 ---

        # 4. 基于上下文分配构建 final_query_to_ondemand
        final_query_to_ondemand = ""
        query_parts = []

        if is_newly_assigned_context and request_context_hash: # 若为新上下文且有历史，则发送历史记录
            logger.info(f"[{request_id}] 客户端 (账户: {email_for_stats}) 新分配至上下文 '{request_context_hash}'。使用历史消息构建查询。")
            # historical_messages 已在开始时提取
            formatted_historical_parts = []
            for msg in historical_messages:
                role = msg.get('role', 'unknown').capitalize()
                content = msg.get('content', '')
                if content: formatted_historical_parts.append(f"{role}: {content}")
            
            if formatted_historical_parts: query_parts.append("\n".join(formatted_historical_parts))
        elif not request_context_hash: # 无历史记录，某种意义上是“新”的，但无过往内容可发送
             logger.info(f"[{request_id}] 此请求无历史上下文哈希。客户端 (账户: {email_for_stats})。")
        else: # 非 newly_assigned_context 且 request_context_hash 存在 (找到精确匹配)
            logger.info(f"[{request_id}] 客户端 (账户: {email_for_stats}) 已服务上下文 '{request_context_hash}'。仅用当前消息构建查询。")

        # 始终添加当前系统提示(若有)和用户查询
        if system_prompt_combined: query_parts.append(f"System: {system_prompt_combined}")
        if current_user_query:
            query_parts.append(f"User: {current_user_query}")
        else: # 应在初始检查时捕获
            logger.error(f"[{request_id}] 严重错误: 最终查询构建时 current_user_query 为空！")
            # 若初始检查稳健，理想情况下不应发生此情况。
            # 若发生，则表明消息解析或逻辑流存在问题。
            # 为安全起见，若 API 要求，确保查询非空。
            if not query_parts: query_parts.append(" ") # 占位符

        final_query_to_ondemand = "\n\n".join(filter(None, query_parts))
        if not final_query_to_ondemand.strip():
            logger.warning(f"[{request_id}] 构建的查询为空或全为空格。发送占位符查询。")
            final_query_to_ondemand = " " # 安全占位符，以防 OnDemand API 要求非空。
        
        logger.info(f"[{request_id}] 构建的 OnDemand 查询 : {final_query_to_ondemand[:1000]}...")

        # 根据请求的模型名称获取 on-demand.io 的 endpoint_id
        endpoint_id = model_mapping.get(requested_model_name, default_endpoint_id)
        if requested_model_name not in model_mapping:
            logger.warning(f"[{request_id}] 模型 '{requested_model_name}' 不在映射表中, 将使用默认端点 '{default_endpoint_id}'.")

        # 构建模型配置，只包含用户明确提供的参数
        model_configs = {}
        
        # 构建模型配置，只包含用户明确提供的参数 (值为None的参数不会被包含)
        if temperature is not None:
            model_configs["temperature"] = temperature
        if max_tokens is not None:
            model_configs["maxTokens"] = max_tokens
        if top_p is not None:
            model_configs["topP"] = top_p
        if frequency_penalty is not None:
            model_configs["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            model_configs["presence_penalty"] = presence_penalty
            
        logger.info(f"[{request_id}] 构建的模型配置: {json.dumps(model_configs, ensure_ascii=False)}")
 
        # request_start_time 已移至会话管理之前
        
        # 使用特定于此 IP 的客户端实例向 OnDemand API 发送查询
        ondemand_result = ondemand_client.send_query(final_query_to_ondemand, endpoint_id=endpoint_id,
                                                     stream=stream_requested, model_configs_input=model_configs)
            
        # 处理响应
        if stream_requested:
            # 流式响应
            def generate_openai_stream(captured_initial_request_messages: List[Dict[str, str]]):
                full_assistant_reply_parts = [] # For aggregating streamed reply
                stream_response_obj = ondemand_result.get("response_obj")
                if not stream_response_obj:  # 确保 response_obj 存在
                    # 计算token数量（仅提示部分，因为流式响应无法准确计算完成tokens）
                    prompt_tokens, _, _ = count_message_tokens(messages, requested_model_name)
                    # 确保prompt_tokens不为None
                    if prompt_tokens is None:
                        prompt_tokens = 0
                    # 错误情况下，完成tokens为0
                    estimated_completion_tokens = 0
                    # 错误情况下，总tokens等于提示tokens
                    estimated_total_tokens = prompt_tokens
                    
                    error_json = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": requested_model_name,
                        "choices": [{"delta": {"content": "[流错误：未获取到响应对象]"}, "index": 0, "finish_reason": "error"}],
                        "usage": {  # 添加token统计信息
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": estimated_completion_tokens,
                            "total_tokens": estimated_total_tokens
                        }
                    }
                    yield f"data: {json.dumps(error_json, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                logger.info(f"[{request_id}] 开始流式传输 OpenAI 格式的响应。")
                # 初始化token计数变量
                actual_input_tokens = None
                actual_output_tokens = None
                actual_total_tokens = None
                
                try:
                    for line in stream_response_obj.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith("data:"):
                                json_str = decoded_line[len("data:"):].strip()
                                if json_str == "[DONE]":  # 这是 on-demand.io 的结束标记
                                    break  # 我们将在循环外发送 OpenAI 的 [DONE]
                                try:
                                    event_data = json.loads(json_str)
                                    event_type = event_data.get("eventType", "")
                                    
                                    # 处理内容块
                                    if event_type == "fulfillment":
                                        content_chunk = event_data.get("answer", "")
                                        if content_chunk is not None:  # 确保 content_chunk 不是 None
                                            full_assistant_reply_parts.append(content_chunk) # Aggregate
                                            openai_chunk = {
                                                "id": request_id,
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": requested_model_name,
                                                "choices": [
                                                    {
                                                        "delta": {"content": content_chunk},
                                                        "index": 0,
                                                        "finish_reason": None  # 流式传输过程中 finish_reason 为 None
                                                    }
                                                ]
                                            }
                                            yield f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"
                                    
                                    # 从metrics事件中提取准确的token计数
                                    elif event_type == "metricsLog":
                                        public_metrics = event_data.get("publicMetrics", {})
                                        if public_metrics:
                                            # 确保获取到的token计数是整数，避免None值
                                            actual_input_tokens = public_metrics.get("inputTokens", 0)
                                            if actual_input_tokens is None:
                                                actual_input_tokens = 0
                                                
                                            actual_output_tokens = public_metrics.get("outputTokens", 0)
                                            if actual_output_tokens is None:
                                                actual_output_tokens = 0
                                                
                                            actual_total_tokens = public_metrics.get("totalTokens", 0)
                                            if actual_total_tokens is None:
                                                actual_total_tokens = 0
                                                
                                            logger.info(f"[{request_id}] 从metricsLog获取到准确的token计数: 输入={actual_input_tokens}, 输出={actual_output_tokens}, 总计={actual_total_tokens}")
                                            
                                except json.JSONDecodeError:
                                    logger.warning(f"[{request_id}] 流式传输中 JSONDecodeError: {json_str}")
                                    continue  # 跳过无法解析的行
                    
                    # 如果没有从metrics中获取到准确的token计数，则使用估算方法
                    if actual_input_tokens == 0 or actual_output_tokens == 0 or actual_total_tokens == 0:
                        logger.warning(f"[{request_id}] 未从metricsLog获取到有效的token计数，使用估算方法")
                        prompt_tokens, _, _ = count_message_tokens(messages, requested_model_name)
                        # 确保prompt_tokens不为None
                        if prompt_tokens is None:
                            prompt_tokens = 0
                        estimated_completion_tokens = max(1, prompt_tokens // 2)  # 确保至少为1
                        estimated_total_tokens = prompt_tokens + estimated_completion_tokens
                    else:
                        # 使用从metrics中获取的准确token计数
                        prompt_tokens = actual_input_tokens
                        estimated_completion_tokens = actual_output_tokens
                        estimated_total_tokens = actual_total_tokens
                    
                    # 循环结束后，发送 OpenAI 流的终止块
                    final_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": requested_model_name,
                        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],  # 标准的结束方式
                        "usage": {  # 添加token统计信息
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": estimated_completion_tokens,
                            "total_tokens": estimated_total_tokens
                        }
                    }
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"  # OpenAI 流的最终结束标记
                    logger.info(f"[{request_id}] 完成 OpenAI 格式响应的流式传输。")
                    
                    full_streamed_reply = "".join(full_assistant_reply_parts)

                    # 更新使用统计
                    request_duration_val = int((time.time() - request_start_time) * 1000)
                    final_prompt_tokens_for_stats = actual_input_tokens if actual_input_tokens is not None and actual_input_tokens > 0 else prompt_tokens
                    final_completion_tokens_for_stats = actual_output_tokens if actual_output_tokens is not None and actual_output_tokens > 0 else estimated_completion_tokens
                    final_total_tokens_for_stats = actual_total_tokens if actual_total_tokens is not None and actual_total_tokens > 0 else estimated_total_tokens
                    used_actual_for_history = actual_input_tokens is not None and actual_input_tokens > 0

                    _update_usage_statistics(
                        config_inst=config_instance,
                        request_id=request_id,
                        requested_model_name=requested_model_name,
                        account_email=email_for_stats,
                        is_success=True,
                        duration_ms=request_duration_val,
                        is_stream=True,
                        prompt_tokens_val=final_prompt_tokens_for_stats,
                        completion_tokens_val=final_completion_tokens_for_stats,
                        total_tokens_val=final_total_tokens_for_stats,
                        prompt_length=len(final_query_to_ondemand),
                        used_actual_tokens_for_history=used_actual_for_history
                    )

                    # 更新客户端的 active_context_hash 以反映对话进展
                    _update_client_context_hash_after_reply(
                        original_request_messages=captured_initial_request_messages,
                        assistant_reply_content=full_streamed_reply,
                        request_id=request_id,
                        user_identifier=token, # user_identifier is token
                        email_for_stats=email_for_stats,
                        current_ondemand_client_instance=ondemand_client,
                        config_inst=config_instance,
                        logger_instance=logger
                    )
                except Exception as e:  # 捕获流处理过程中的任何异常
                    logger.error(f"[{request_id}] 流式传输过程中发生错误: {e}")
                    # 在流错误的情况下，不更新 active_context_hash，因为它可能基于不完整的对话
                    # 计算token数量（仅提示部分，因为流式响应无法准确计算完成tokens）
                    prompt_tokens, _, _ = count_message_tokens(messages, requested_model_name)
                    # 确保prompt_tokens不为None
                    if prompt_tokens is None:
                        prompt_tokens = 0
                    # 错误情况下，完成tokens为0
                    estimated_completion_tokens = 0
                    # 错误情况下，总tokens等于提示tokens
                    estimated_total_tokens = prompt_tokens
                    
                    error_json = {  # 发送一个错误块
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": requested_model_name,
                        "choices": [{"delta": {"content": f"[流处理异常: {str(e)}]"}, "index": 0, "finish_reason": "error"}],
                        "usage": {  # 添加token统计信息
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": estimated_completion_tokens,
                            "total_tokens": estimated_total_tokens
                        }
                    }
                    yield f"data: {json.dumps(error_json, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    
                    # 更新使用统计 - 失败的流式请求
                    request_duration_val = int((time.time() - request_start_time) * 1000)
                    _update_usage_statistics(
                        config_inst=config_instance,
                        request_id=request_id,
                        requested_model_name=requested_model_name,
                        account_email=email_for_stats, # email_for_stats 可能为 None
                        is_success=False,
                        duration_ms=request_duration_val,
                        is_stream=True,
                        prompt_tokens_val=prompt_tokens if prompt_tokens is not None else 0,
                        completion_tokens_val=0,
                        total_tokens_val=prompt_tokens if prompt_tokens is not None else 0,
                        error_message=str(e)
                    )
                finally:
                    if stream_response_obj:  # 确保关闭响应对象
                        stream_response_obj.close()
 
            return Response(stream_with_context(generate_openai_stream(initial_messages_from_request)), content_type='text/event-stream; charset=utf-8')
        else:
            # 非流式响应
            final_content = ondemand_result.get("content", "")
            
            # 计算token数量
            prompt_tokens, completion_tokens, total_tokens = count_message_tokens(messages, requested_model_name)
            completion_tokens_actual = count_tokens(final_content, requested_model_name)
            total_tokens_actual = prompt_tokens + completion_tokens_actual
            
            openai_response = {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": requested_model_name,
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": final_content
                        },
                        "finish_reason": "stop",  # 假设成功完成则为 "stop"
                        "index": 0
                    }
                ],
                "usage": {  # 计算token数量
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens_actual,
                    "total_tokens": total_tokens_actual
                }
            }
            logger.info(f"[{request_id}] 发送非流式 OpenAI 格式的响应。")
            
            # 更新使用统计 - 非流式成功请求
            request_duration_val = int((time.time() - request_start_time) * 1000)
            _update_usage_statistics(
                config_inst=config_instance,
                request_id=request_id,
                requested_model_name=requested_model_name,
                account_email=email_for_stats,
                is_success=True,
                duration_ms=request_duration_val,
                is_stream=False,
                prompt_tokens_val=prompt_tokens,
                completion_tokens_val=completion_tokens_actual,
                total_tokens_val=total_tokens_actual,
                prompt_length=len(final_query_to_ondemand),
                completion_length=len(final_content) if final_content else 0,
                used_actual_tokens_for_history=True
            )

            # 更新客户端的 active_context_hash 以反映对话进展
            _update_client_context_hash_after_reply(
                original_request_messages=initial_messages_from_request,
                assistant_reply_content=final_content,
                request_id=request_id,
                user_identifier=token, # user_identifier is token
                email_for_stats=email_for_stats,
                current_ondemand_client_instance=ondemand_client,
                config_inst=config_instance,
                logger_instance=logger
            )
            
            return openai_response
    
    @app.route('/', methods=['GET'])
    def show_stats():
        """显示用量统计信息的HTML页面"""
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        current_date = current_time.strftime('%Y-%m-%d')
        
        with config_instance.usage_stats_lock:
            # 复制基础统计数据
            total_requests = config_instance.usage_stats["total_requests"]
            successful_requests = config_instance.usage_stats["successful_requests"]
            failed_requests = config_instance.usage_stats["failed_requests"]
            total_prompt_tokens = config_instance.usage_stats["total_prompt_tokens"]
            total_completion_tokens = config_instance.usage_stats["total_completion_tokens"]
            total_tokens = config_instance.usage_stats["total_tokens"]
            
            # 计算成功率（整数百分比）
            success_rate = int((successful_requests / total_requests * 100) if total_requests > 0 else 0)
            
            # 计算平均响应时间
            successful_history = [req for req in config_instance.usage_stats["request_history"] if req.get('success', False)]
            total_duration = sum(req.get('duration_ms', 0) for req in successful_history)
            avg_duration = (total_duration / successful_requests) if successful_requests > 0 else 0
            
            # 计算最快响应时间
            min_duration = min([req.get('duration_ms', float('inf')) for req in successful_history]) if successful_history else 0
            
            # 计算今日请求数和增长率
            today_requests = config_instance.usage_stats["daily_usage"].get(current_date, 0)
            # 确保不会出现除以零或None值的情况
            if total_requests is None or today_requests is None:
                growth_rate = 0
            elif total_requests == today_requests or (total_requests - today_requests) <= 0:
                growth_rate = 100  # 如果所有请求都是今天的，增长率为100%
            else:
                growth_rate = (today_requests / (total_requests - today_requests) * 100)
            
            # 计算估算成本 - 使用模型价格配置
            total_cost = 0.0
            model_costs = {}  # 存储每个模型的成本
            
            # 获取请求历史中的token使用情况
            for req in successful_history:
                model_name = req.get('model', '')
                # 从配置获取模型价格
                all_model_prices = config_instance.get('model_prices', {})
                default_model_price = config_instance.get('default_model_price', {'input': 0.50 / 1000000, 'output': 2.00 / 1000000}) # 提供备用默认值
                model_price = all_model_prices.get(model_name, default_model_price)
                
                # 获取输入和输出token数量
                input_tokens = req.get('prompt_tokens', 0)
                
                # 根据是否有准确的completion_tokens字段决定使用哪个字段
                if 'completion_tokens' in req:
                    output_tokens = req.get('completion_tokens', 0)
                else:
                    output_tokens = req.get('estimated_completion_tokens', 0)
                
                # 计算此次请求的成本
                request_cost = (input_tokens * model_price['input']) + (output_tokens * model_price['output'])
                total_cost += request_cost
                
                # 累加到模型成本中
                if model_name not in model_costs:
                    model_costs[model_name] = 0
                model_costs[model_name] += request_cost
            
            # 计算平均成本
            avg_cost = (total_cost / successful_requests) if successful_requests > 0 else 0
            
            # 获取最常用模型
            model_usage = dict(config_instance.usage_stats["model_usage"])
            top_models = sorted(model_usage.items(), key=lambda x: x[1], reverse=True)
            top_model = top_models[0] if top_models else None
            
            # 构建完整的统计数据字典
            stats = {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": success_rate,
                "avg_duration": avg_duration,
                "min_duration": min_duration,
                "today_requests": today_requests,
                "growth_rate": growth_rate,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "avg_cost": avg_cost,
                "model_usage": model_usage,
                "model_costs": model_costs,  # 添加每个模型的成本
                "top_model": top_model,
                "model_tokens": dict(config_instance.usage_stats["model_tokens"]),
                "account_usage": dict(config_instance.usage_stats["account_usage"]),
                "daily_usage": dict(sorted(config_instance.usage_stats["daily_usage"].items(), reverse=True)[:30]),  # 最近30天
                "hourly_usage": dict(sorted(config_instance.usage_stats["hourly_usage"].items(), reverse=True)[:48]),  # 最近48小时
                "request_history": list(config_instance.usage_stats["request_history"][:50]),
                "daily_tokens": dict(sorted(config_instance.usage_stats["daily_tokens"].items(), reverse=True)[:30]),  # 最近30天
                "hourly_tokens": dict(sorted(config_instance.usage_stats["hourly_tokens"].items(), reverse=True)[:48]),  # 最近48小时
                "last_saved": config_instance.usage_stats.get("last_saved", "从未保存")
            }
        
        # 使用render_template渲染模板
        return render_template('stats.html', stats=stats, current_time=current_time_str)
    
    @app.route('/save_stats', methods=['POST'])
    def save_stats():
        """手动保存统计数据"""
        try:
            config_instance.save_stats_to_file()
            logger.info("统计数据已手动保存")
            return redirect(url_for('show_stats'))
        except Exception as e:
            logger.error(f"手动保存统计数据时出错: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500