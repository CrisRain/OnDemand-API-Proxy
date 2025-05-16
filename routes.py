import json
import time
import uuid
import html
from datetime import datetime
from typing import Dict, List, Any
from flask import request, Response, stream_with_context, jsonify, render_template, redirect, url_for, flash
from datetime import datetime

from utils import logger, generate_request_id, count_tokens, count_message_tokens
import config
from auth import RateLimiter
from client import OnDemandAPIClient

# 初始化速率限制器
rate_limiter = RateLimiter(60)  # 默认每分钟60个请求

# 获取配置实例
config_instance = config.config_instance

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
        if value >= 1000000000:
            return f"{value/1000000000:.1f}G"
        elif value >= 1000000:
            return f"{value/1000000:.1f}M"
        elif value >= 1000:
            return f"{value/1000:.1f}K"
        else:
            return f"{value:.0f}" if value == int(value) else f"{value:.1f}"
    except (ValueError, TypeError):
        return str(value)

def format_duration(ms):
    """将毫秒格式化为更易读的格式"""
    if ms is None or ms == '-':
        return '-'
    
    try:
        ms = int(ms)
        if ms >= 60000:  # 超过1分钟
            return f"{ms/60000:.1f}分钟"
        elif ms >= 1000:  # 超过1秒
            return f"{ms/1000:.1f}秒"
        else:
            return f"{ms}毫秒"
    except (ValueError, TypeError):
        return str(ms)

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
        client_ip = request.remote_addr  # 获取客户端 IP 地址
        logger.info(f"[{request_id}] 收到来自 IP: {client_ip} 的 /v1/chat/completions 请求")
        
        # 验证访问令牌
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.warning(f"[{request_id}] 未提供认证令牌或格式错误")
            return {"error": {"message": "缺少有效的认证令牌", "type": "auth_error", "code": "missing_token"}}, 401
        
        # 获取API访问令牌
        api_access_token = config_instance.get('api_access_token')
        print(f"[{request_id}] API_ACCESS_TOKEN: {api_access_token}")

        token = auth_header[7:]  # 去掉 'Bearer ' 前缀
        if token != api_access_token:
            logger.warning(f"[{request_id}] 提供了无效的认证令牌")
            return {"error": {"message": "无效的认证令牌", "type": "auth_error", "code": "invalid_token"}}, 401

        # 检查速率限制
        if not rate_limiter.is_allowed(client_ip):
            logger.warning(f"[{request_id}] 客户端 {client_ip} 超过速率限制")
            return {"error": {"message": "请求频率过高，请稍后再试", "type": "rate_limit_error", "code": "rate_limit_exceeded"}}, 429

        openai_data = request.get_json()
        if not openai_data:
            logger.error(f"[{request_id}] 请求体不是有效的JSON")
            return {"error": {"message": "请求体必须是 JSON。", "type": "invalid_request_error", "code": None}}, 400
        
        if app.config.get('DEBUG_MODE', False):
            logger.debug(f"[{request_id}] OpenAI 请求数据: {json.dumps(openai_data, indent=2, ensure_ascii=False)}")

        # 为每次请求获取新的账户并创建新的客户端会话
        with config_instance.client_sessions_lock:  # 线程安全地访问会话
            current_time = datetime.now()
            ondemand_client = None

            logger.info(f"[{request_id}] 为每次请求切换账户，创建新的客户端和会话。")
            email, password = config.get_next_ondemand_account_details()
            # 为日志创建一个更具描述性的 client_id
            client_identifier_for_log = f"{client_ip}-{email.split('@')[0]}"
            ondemand_client = OnDemandAPIClient(email, password, client_id=client_identifier_for_log)
            
            # 尝试登录并创建会话
            if not ondemand_client.sign_in() or not ondemand_client.create_session():
                error_msg = f"为 {email} 初始化 OnDemand 客户端失败: {ondemand_client.last_error or '未知的初始化错误'}"
                logger.error(f"[{request_id}] 错误: {error_msg}")
                return {"error": {"message": error_msg, "type": "api_error", "code": "ondemand_init_failed"}}, 500
            
            config_instance.client_sessions[client_ip] = {"client": ondemand_client, "last_time": current_time}
            logger.info(f"[{request_id}] 已为 {client_ip} 创建并存储新会话")
            
            # 更新最后交互时间
            config_instance.client_sessions[client_ip]["last_time"] = current_time

        # 从 OpenAI 请求中提取参数
        messages: List[Dict[str, str]] = openai_data.get('messages', [])
        stream_requested: bool = openai_data.get('stream', False)
        # 如果请求中没有指定模型，则使用映射表中的一个默认模型，或者最终的 DEFAULT_ENDPOINT_ID
        model_mapping = config_instance._model_mapping
        default_endpoint_id = config_instance.get('default_endpoint_id')
        requested_model_name: str = openai_data.get('model', list(model_mapping.keys())[0] if model_mapping else default_endpoint_id)
        temperature: float = openai_data.get('temperature', 0.7)
        max_tokens: int = openai_data.get('max_tokens', 4000)
        top_p: float = openai_data.get('top_p', 1.0)
        frequency_penalty: float = openai_data.get('frequency_penalty', 0.0)
        presence_penalty: float = openai_data.get('presence_penalty', 0.0)

        if not messages:
            logger.error(f"[{request_id}] 缺少 'messages' 字段")
            return {"error": {"message": "缺少 'messages' 字段。", "type": "invalid_request_error", "code": "missing_messages"}}, 400

        # 为 on-demand.io 构建查询
        # on-demand.io 通常接受单个查询字符串，上下文由其会话管理。
        # 我们将发送最新的用户查询，可选地以系统提示为前缀。
        system_prompt = ""
        user_query = ""
        conversation_history = []
        
        # 从 messages 数组中提取内容，考虑多轮对话
        for msg in messages:
            role = msg.get('role', '').lower()
            content = msg.get('content', '')
            if not content:  # 跳过空内容的消息
                continue
            if role == 'system':
                system_prompt = content  # 获取最后一个 system 角色的内容
            elif role == 'user':
                conversation_history.append(f"User: {content}")
                user_query = content  # 获取最后一个 user 角色的内容
            elif role == 'assistant':
                conversation_history.append(f"Assistant: {content}")

        if not user_query:  # 必须有一个用户查询
            logger.error(f"[{request_id}] 'messages' 中未找到 'user' 角色的消息")
            return {"error": {"message": "'messages' 中未找到 'user' 角色的消息。", "type": "invalid_request_error", "code": "no_user_message"}}, 400

        # 组合查询，考虑历史对话
        final_query_parts = []
        if system_prompt:
            final_query_parts.append(f"System instruction: {system_prompt}")  # 可以自定义这个前缀
        
        if conversation_history:
            # 添加对话历史，但确保最后一个用户查询在最后
            for entry in conversation_history[:-1]:  # 排除最后一个用户查询
                final_query_parts.append(entry)
        
        final_query_parts.append(f"User: {user_query}")  # 确保用户查询在最后

        final_query_to_ondemand = "\n\n".join(final_query_parts)  # 用换行符分隔不同部分
        
        logger.info(f"[{request_id}] 构建的 OnDemand 查询 (前200字符): {final_query_to_ondemand[:200]}...")

        # 根据请求的模型名称获取 on-demand.io 的 endpoint_id
        endpoint_id = model_mapping.get(requested_model_name, default_endpoint_id)
        if requested_model_name not in model_mapping:
            logger.warning(f"[{request_id}] 模型 '{requested_model_name}' 不在映射表中, 将使用默认端点 '{default_endpoint_id}'.")

        # 构建模型配置
        model_configs = {
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
            "frequencyPenalty": frequency_penalty,
            "presencePenalty": presence_penalty
        }

        # 记录请求开始时间
        request_start_time = time.time()
        
        # 使用特定于此 IP 的客户端实例向 OnDemand API 发送查询
        ondemand_result = ondemand_client.send_query(final_query_to_ondemand, endpoint_id=endpoint_id,
                                                     stream=stream_requested, model_configs=model_configs)

        if "error" in ondemand_result:
            error_msg = f"OnDemand API 错误: {ondemand_result['error']}"
            logger.error(f"[{request_id}] 错误: {error_msg}")
            # 如果错误表明是会话/认证问题，可能需要清除此客户端的会话
            if ondemand_client.last_error and \
               ("sign-in failed" in ondemand_client.last_error or \
                "Token refresh and re-login failed" in ondemand_client.last_error or \
                "Query failed: Session creation failed" in ondemand_client.last_error):  # 增加了会话创建失败的检查
                logger.warning(f"[{request_id}] 由于持续的认证/会话错误，正在清除 IP {client_ip} 的会话。")
                with config_instance.client_sessions_lock:
                    if client_ip in config_instance.client_sessions:
                        del config_instance.client_sessions[client_ip]
            # 记录失败的请求
            with config_instance.usage_stats_lock:
                config_instance.usage_stats["total_requests"] += 1
                config_instance.usage_stats["failed_requests"] += 1
                # 记录请求历史
                config_instance.usage_stats["request_history"].append({
                    "id": request_id,
                    "timestamp": datetime.now().isoformat(),
                    "model": requested_model_name,
                    "account": email,
                    "success": False,
                    "error": error_msg,
                    "duration_ms": int((time.time() - request_start_time) * 1000)
                })
                # 限制历史记录数量
                max_history_items = config_instance.get('max_history_items')
                if len(config_instance.usage_stats["request_history"]) > max_history_items:
                    config_instance.usage_stats["request_history"] = config_instance.usage_stats["request_history"][-max_history_items:]
                    
            return {"error": {"message": error_msg, "type": "api_error", "code": "ondemand_query_failed"}}, 502  # 502 Bad Gateway 表示上游服务问题

        # 处理响应
        if stream_requested:
            # 流式响应
            def generate_openai_stream():
                stream_response_obj = ondemand_result.get("response_obj")
                if not stream_response_obj:  # 确保 response_obj 存在
                    # 计算token数量（仅提示部分，因为流式响应无法准确计算完成tokens）
                    prompt_tokens, _, _ = count_message_tokens(messages, requested_model_name)
                    # 错误情况下，完成tokens为0
                    estimated_completion_tokens = 0
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
                                    if event_data.get("eventType", "") == "fulfillment":
                                        content_chunk = event_data.get("answer", "")
                                        if content_chunk is not None:  # 确保 content_chunk 不是 None
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
                                except json.JSONDecodeError:
                                    logger.warning(f"[{request_id}] 流式传输中 JSONDecodeError: {json_str}")
                                    continue  # 跳过无法解析的行
                    
                    # 计算token数量（仅提示部分，因为流式响应无法准确计算完成tokens）
                    prompt_tokens, _, _ = count_message_tokens(messages, requested_model_name)
                    # 估算完成tokens（流式响应无法准确计算，使用提示tokens的一半作为估算）
                    estimated_completion_tokens = prompt_tokens // 2
                    estimated_total_tokens = prompt_tokens + estimated_completion_tokens
                    
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
                    
                    # Token统计已移至最终块发送前
                    
                    # 记录成功的流式请求
                    request_duration = time.time() - request_start_time
                    with config_instance.usage_stats_lock:
                        config_instance.usage_stats["total_requests"] += 1
                        config_instance.usage_stats["successful_requests"] += 1
                        config_instance.usage_stats["model_usage"][requested_model_name] += 1
                        config_instance.usage_stats["account_usage"][email] += 1
                        
                        # 记录token使用情况（估算值）
                        config_instance.usage_stats["total_prompt_tokens"] += prompt_tokens
                        config_instance.usage_stats["total_completion_tokens"] += estimated_completion_tokens
                        config_instance.usage_stats["total_tokens"] += estimated_total_tokens
                        config_instance.usage_stats["model_tokens"][requested_model_name] += estimated_total_tokens
                        
                        # 记录每日和每小时使用情况
                        today = datetime.now().strftime("%Y-%m-%d")
                        hour = datetime.now().strftime("%Y-%m-%d %H:00")
                        config_instance.usage_stats["daily_usage"][today] += 1
                        config_instance.usage_stats["hourly_usage"][hour] += 1
                        config_instance.usage_stats["daily_tokens"][today] += estimated_total_tokens
                        config_instance.usage_stats["hourly_tokens"][hour] += estimated_total_tokens
                        
                        # 记录请求历史
                        config_instance.usage_stats["request_history"].append({
                            "id": request_id,
                            "timestamp": datetime.now().isoformat(),
                            "model": requested_model_name,
                            "account": email,
                            "success": True,
                            "duration_ms": int(request_duration * 1000),
                            "prompt_length": len(final_query_to_ondemand),
                            "prompt_tokens": prompt_tokens,
                            "estimated_completion_tokens": estimated_completion_tokens,
                            "estimated_total_tokens": estimated_total_tokens,
                            "stream": True
                        })
                        # 限制历史记录数量
                        max_history_items = config_instance.get('max_history_items')
                        if len(config_instance.usage_stats["request_history"]) > max_history_items:
                            config_instance.usage_stats["request_history"] = config_instance.usage_stats["request_history"][-max_history_items:]
                except Exception as e:  # 捕获流处理过程中的任何异常
                    logger.error(f"[{request_id}] 流式传输过程中发生错误: {e}")
                    
                    # 计算token数量（仅提示部分，因为流式响应无法准确计算完成tokens）
                    prompt_tokens, _, _ = count_message_tokens(messages, requested_model_name)
                    # 估算完成tokens（流式响应无法准确计算，使用提示tokens的一半作为估算）
                    estimated_completion_tokens = 0  # 错误情况下，完成tokens为0
                    estimated_total_tokens = prompt_tokens  # 错误情况下，总tokens等于提示tokens
                    
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
                    
                    # 记录失败的流式请求
                    with config_instance.usage_stats_lock:
                        config_instance.usage_stats["total_requests"] += 1
                        config_instance.usage_stats["failed_requests"] += 1
                        # 记录请求历史
                        config_instance.usage_stats["request_history"].append({
                            "id": request_id,
                            "timestamp": datetime.now().isoformat(),
                            "model": requested_model_name,
                            "account": email,
                            "success": False,
                            "error": str(e),
                            "duration_ms": int((time.time() - request_start_time) * 1000),
                            "stream": True
                        })
                        # 限制历史记录数量
                        max_history_items = config_instance.get('max_history_items')
                        if len(config_instance.usage_stats["request_history"]) > max_history_items:
                            config_instance.usage_stats["request_history"] = config_instance.usage_stats["request_history"][-max_history_items:]
                finally:
                    if stream_response_obj:  # 确保关闭响应对象
                        stream_response_obj.close()

            return Response(stream_with_context(generate_openai_stream()), content_type='text/event-stream; charset=utf-8')
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
            
            # 记录成功的请求
            request_duration = time.time() - request_start_time
            with config_instance.usage_stats_lock:
                config_instance.usage_stats["total_requests"] += 1
                config_instance.usage_stats["successful_requests"] += 1
                config_instance.usage_stats["model_usage"][requested_model_name] += 1
                config_instance.usage_stats["account_usage"][email] += 1
                
                # 记录token使用情况
                config_instance.usage_stats["total_prompt_tokens"] += prompt_tokens
                config_instance.usage_stats["total_completion_tokens"] += completion_tokens_actual
                config_instance.usage_stats["total_tokens"] += total_tokens_actual
                config_instance.usage_stats["model_tokens"][requested_model_name] += total_tokens_actual
                
                # 记录每日和每小时使用情况
                today = datetime.now().strftime("%Y-%m-%d")
                hour = datetime.now().strftime("%Y-%m-%d %H:00")
                config_instance.usage_stats["daily_usage"][today] += 1
                config_instance.usage_stats["hourly_usage"][hour] += 1
                config_instance.usage_stats["daily_tokens"][today] += total_tokens_actual
                config_instance.usage_stats["hourly_tokens"][hour] += total_tokens_actual
                
                # 记录请求历史
                config_instance.usage_stats["request_history"].append({
                    "id": request_id,
                    "timestamp": datetime.now().isoformat(),
                    "model": requested_model_name,
                    "account": email,
                    "success": True,
                    "duration_ms": int(request_duration * 1000),
                    "prompt_length": len(final_query_to_ondemand),
                    "completion_length": len(final_content) if final_content else 0,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens_actual,
                    "total_tokens": total_tokens_actual
                })
                # 限制历史记录数量
                max_history_items = config_instance.get('max_history_items')
                if len(config_instance.usage_stats["request_history"]) > max_history_items:
                    config_instance.usage_stats["request_history"] = config_instance.usage_stats["request_history"][-max_history_items:]
            
            return openai_response
    
    @app.route('/', methods=['GET'])
    def show_stats():
        """显示用量统计信息的HTML页面"""
        with config_instance.usage_stats_lock:
            # 复制统计数据以避免在渲染过程中发生变化
            stats = {
                "total_requests": config_instance.usage_stats["total_requests"],
                "successful_requests": config_instance.usage_stats["successful_requests"],
                "failed_requests": config_instance.usage_stats["failed_requests"],
                "model_usage": dict(config_instance.usage_stats["model_usage"]),
                "account_usage": dict(config_instance.usage_stats["account_usage"]),
                "daily_usage": dict(sorted(config_instance.usage_stats["daily_usage"].items(), reverse=True)[:30]),  # 最近30天
                "hourly_usage": dict(sorted(config_instance.usage_stats["hourly_usage"].items(), reverse=True)[:48]),  # 最近48小时
                "request_history": list(config_instance.usage_stats["request_history"][:50]),  # 最近50条记录
                # Token统计
                "total_prompt_tokens": config_instance.usage_stats["total_prompt_tokens"],
                "total_completion_tokens": config_instance.usage_stats["total_completion_tokens"],
                "total_tokens": config_instance.usage_stats["total_tokens"],
                "model_tokens": dict(config_instance.usage_stats["model_tokens"]),
                "daily_tokens": dict(sorted(config_instance.usage_stats["daily_tokens"].items(), reverse=True)[:30]),  # 最近30天
                "hourly_tokens": dict(sorted(config_instance.usage_stats["hourly_tokens"].items(), reverse=True)[:48]),  # 最近48小时
                "last_saved": config_instance.usage_stats.get("last_saved", "从未保存")
            }
        
        # 使用render_template渲染模板
        return render_template('stats.html', stats=stats, current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
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