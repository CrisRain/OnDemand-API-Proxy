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

        # 移除了request_history相关代码

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
        request_id = generate_request_id()
        request_start_time = time.time()
        
        # 验证访问令牌
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return {"error": {"message": "缺少有效的认证令牌", "type": "auth_error", "code": "missing_token"}}, 401
        
        token = auth_header[7:]  # 去掉 'Bearer ' 前缀
        if token != config_instance.get('api_access_token'):
            return {"error": {"message": "无效的认证令牌", "type": "auth_error", "code": "invalid_token"}}, 401

        # 检查速率限制
        if not rate_limiter.is_allowed(token):
            return {"error": {"message": "请求频率过高，请稍后再试", "type": "rate_limit_error", "code": "rate_limit_exceeded"}}, 429

        # 解析请求数据
        openai_data = request.get_json()
        if not openai_data:
            return {"error": {"message": "请求体必须是 JSON。", "type": "invalid_request_error", "code": None}}, 400
        
        # 提取基本参数
        messages = openai_data.get('messages', [])
        if not messages:
            return {"error": {"message": "缺少 'messages' 字段。", "type": "invalid_request_error", "code": "missing_messages"}}, 400
            
        stream_requested = openai_data.get('stream', False)
        model_mapping = config_instance._model_mapping
        default_endpoint_id = config_instance.get('default_endpoint_id')
        requested_model_name = openai_data.get('model', list(model_mapping.keys())[0] if model_mapping else default_endpoint_id)
        
        # 检查是否有用户消息
        user_messages = [msg for msg in messages if msg.get('role') == 'user' and msg.get('content')]
        if not user_messages:
            return {"error": {"message": "'messages' 中未找到有效的 'user' 角色的消息内容。", "type": "invalid_request_error", "code": "no_user_message"}}, 400
        
        # 创建新的客户端会话
        ondemand_client = None
        email_for_stats = None
        
        with config_instance.client_sessions_lock:
            MAX_ACCOUNT_ATTEMPTS = config_instance.get('max_account_attempts', 3)
            for attempt in range(MAX_ACCOUNT_ATTEMPTS):
                email, password = config.get_next_ondemand_account_details()
                if not email or not password:
                    continue
                
                email_for_stats = email
                client_id = f"{token[:8]}-{email.split('@')[0]}-{request_id[:4]}"
                temp_client = OnDemandAPIClient(email, password, client_id=client_id)
                
                if temp_client.sign_in() and temp_client.create_session():
                    ondemand_client = temp_client
                    ondemand_client._associated_user_identifier = token
                    ondemand_client._associated_request_ip = request.remote_addr
                    break
            
            if not ondemand_client:
                error_msg = "无法创建有效的客户端会话"
                prompt_tokens, _, _ = count_message_tokens(messages, requested_model_name)
                _update_usage_statistics(
                    config_inst=config_instance, request_id=request_id,
                    requested_model_name=requested_model_name,
                    account_email=email_for_stats,
                    is_success=False, duration_ms=int((time.time() - request_start_time) * 1000),
                    is_stream=stream_requested, prompt_tokens_val=prompt_tokens or 0,
                    completion_tokens_val=0, total_tokens_val=prompt_tokens or 0,
                    error_message=error_msg
                )
                return {"error": {"message": error_msg, "type": "api_error", "code": "client_unavailable"}}, 503
        
        # 构建查询 - 使用符合OnDemand API期望的格式
        # 根据API期望将消息构建为JSON对象
        messages_to_send = []
        
        # 处理系统消息 - 将系统消息放在最前面
        system_messages = [msg for msg in messages if msg.get('role') == 'system']
        # 处理用户和助手消息 - 保持原始顺序
        other_messages = [msg for msg in messages if msg.get('role') != 'system']
        
        # 按顺序添加所有消息
        processed_messages = system_messages + other_messages
        
        # 构建最终查询 - 使用JSON格式
        final_query_to_ondemand = json.dumps({
            "messages": processed_messages
        })
        
        # 确保消息不为空
        if not processed_messages:
            final_query_to_ondemand = json.dumps({
                "messages": [{"role": "user", "content": " "}]
            })
            
        # 构建模型配置
        model_configs = {}
        for param_name, api_name in [
            ('temperature', 'temperature'),
            ('max_tokens', 'maxTokens'),
            ('top_p', 'topP'),
            ('frequency_penalty', 'frequency_penalty'),
            ('presence_penalty', 'presence_penalty')
        ]:
            if openai_data.get(param_name) is not None:
                model_configs[api_name] = openai_data.get(param_name)
        
        # 发送查询
        endpoint_id = model_mapping.get(requested_model_name, default_endpoint_id)
        ondemand_result = ondemand_client.send_query(
            final_query_to_ondemand,
            endpoint_id=endpoint_id or default_endpoint_id,
            stream=stream_requested,
            model_configs_input=model_configs
        )
            
        # 处理响应
        if stream_requested:
            # 流式响应
            def generate_openai_stream():
                stream_response_obj = ondemand_result.get("response_obj")
                if not stream_response_obj:
                    prompt_tokens, _, _ = count_message_tokens(messages, requested_model_name)
                    error_json = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": requested_model_name,
                        "choices": [{"delta": {"content": "[流错误：未获取到响应对象]"}, "index": 0, "finish_reason": "error"}],
                        "usage": {"prompt_tokens": prompt_tokens or 0, "completion_tokens": 0, "total_tokens": prompt_tokens or 0}
                    }
                    yield f"data: {json.dumps(error_json, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # 用于聚合完整回复和跟踪token统计
                full_reply = []
                actual_tokens = {"input": 0, "output": 0, "total": 0}
                
                try:
                    for line in stream_response_obj.iter_lines():
                        if not line:
                            continue
                            
                        decoded_line = line.decode('utf-8')
                        if not decoded_line.startswith("data:"):
                            continue
                            
                        json_str = decoded_line[len("data:"):].strip()
                        if json_str == "[DONE]":
                            break
                            
                        try:
                            event_data = json.loads(json_str)
                            event_type = event_data.get("eventType", "")
                            
                            # 处理内容块
                            if event_type == "fulfillment":
                                content = event_data.get("answer", "")
                                if content is not None:
                                    full_reply.append(content)
                                    chunk_data = {
                                        'id': request_id,
                                        'object': 'chat.completion.chunk',
                                        'created': int(time.time()),
                                        'model': requested_model_name,
                                        'choices': [{'delta': {'content': content}, 'index': 0, 'finish_reason': None}]
                                    }
                                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                            
                            # 从metrics事件中提取准确的token计数
                            elif event_type == "metricsLog":
                                metrics = event_data.get("publicMetrics", {})
                                if metrics:
                                    actual_tokens["input"] = metrics.get("inputTokens", 0) or 0
                                    actual_tokens["output"] = metrics.get("outputTokens", 0) or 0
                                    actual_tokens["total"] = metrics.get("totalTokens", 0) or 0
                        except json.JSONDecodeError:
                            continue
                    
                    # 使用实际token或回退到估算
                    if not any(actual_tokens.values()):
                        prompt_tokens, _, _ = count_message_tokens(messages, requested_model_name)
                        completion_tokens = max(1, len("".join(full_reply)) // 4)  # 粗略估算
                        total_tokens = (prompt_tokens or 0) + completion_tokens
                    else:
                        prompt_tokens = actual_tokens["input"]
                        completion_tokens = actual_tokens["output"]
                        total_tokens = actual_tokens["total"]
                    
                    # 发送终止块
                    yield f"data: {json.dumps({
                        'id': request_id,
                        'object': 'chat.completion.chunk',
                        'created': int(time.time()),
                        'model': requested_model_name,
                        'choices': [{'delta': {}, 'index': 0, 'finish_reason': 'stop'}],
                        'usage': {
                            'prompt_tokens': prompt_tokens,
                            'completion_tokens': completion_tokens,
                            'total_tokens': total_tokens
                        }
                    }, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    
                    # 更新使用统计
                    _update_usage_statistics(
                        config_inst=config_instance,
                        request_id=request_id,
                        requested_model_name=requested_model_name,
                        account_email=ondemand_client.email,
                        is_success=True,
                        duration_ms=int((time.time() - request_start_time) * 1000),
                        is_stream=True,
                        prompt_tokens_val=prompt_tokens,
                        completion_tokens_val=completion_tokens,
                        total_tokens_val=total_tokens,
                        prompt_length=len(final_query_to_ondemand)
                    )
                except Exception as e:
                    # 处理异常
                    prompt_tokens, _, _ = count_message_tokens(messages, requested_model_name)
                    yield f"data: {json.dumps({
                        'id': request_id,
                        'object': 'chat.completion.chunk',
                        'created': int(time.time()),
                        'model': requested_model_name,
                        'choices': [{'delta': {'content': f'[流处理异常: {str(e)}]'}, 'index': 0, 'finish_reason': 'error'}],
                        'usage': {'prompt_tokens': prompt_tokens or 0, 'completion_tokens': 0, 'total_tokens': prompt_tokens or 0}
                    }, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    
                    # 更新失败统计
                    _update_usage_statistics(
                        config_inst=config_instance,
                        request_id=request_id,
                        requested_model_name=requested_model_name,
                        account_email=ondemand_client.email,
                        is_success=False,
                        duration_ms=int((time.time() - request_start_time) * 1000),
                        is_stream=True,
                        prompt_tokens_val=prompt_tokens or 0,
                        completion_tokens_val=0,
                        total_tokens_val=prompt_tokens or 0,
                        error_message=str(e)
                    )
                finally:
                    if stream_response_obj:
                        stream_response_obj.close()
            
            return Response(stream_with_context(generate_openai_stream()),
                           content_type='text/event-stream; charset=utf-8')
        else:
            # 非流式响应
            final_content = ondemand_result.get("content", "")
            
            # 计算token数量
            prompt_tokens, _, _ = count_message_tokens(messages, requested_model_name)
            completion_tokens = count_tokens(final_content, requested_model_name)
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
            
            # 构建OpenAI格式响应
            response = {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": requested_model_name,
                "choices": [{
                    "message": {"role": "assistant", "content": final_content},
                    "finish_reason": "stop",
                    "index": 0
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }
            
            # 更新使用统计
            _update_usage_statistics(
                config_inst=config_instance,
                request_id=request_id,
                requested_model_name=requested_model_name,
                account_email=ondemand_client.email,
                is_success=True,
                duration_ms=int((time.time() - request_start_time) * 1000),
                is_stream=False,
                prompt_tokens_val=prompt_tokens,
                completion_tokens_val=completion_tokens,
                total_tokens_val=total_tokens,
                prompt_length=len(final_query_to_ondemand),
                completion_length=len(final_content) if final_content else 0
            )
            
            return response
    
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
            
            # 由于移除了request_history，无法计算具体的平均响应时间和最快响应时间
            # 设置为默认值
            avg_duration = 0
            min_duration = 0
            
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
            
            # 由于移除了request_history，我们无法直接计算成本
            # 这里使用总token数和默认价格来估算总成本
            all_model_prices = config_instance.get('model_prices', {})
            default_model_price = config_instance.get('default_model_price', {'input': 0.50 / 1000000, 'output': 2.00 / 1000000})
            
            # 假设输入输出token比例为1:3来估算成本
            avg_input_price = sum([price['input'] for price in all_model_prices.values()]) / len(all_model_prices) if all_model_prices else default_model_price['input']
            avg_output_price = sum([price['output'] for price in all_model_prices.values()]) / len(all_model_prices) if all_model_prices else default_model_price['output']
            
            estimated_input_tokens = total_prompt_tokens
            estimated_output_tokens = total_completion_tokens
            
            total_cost = (estimated_input_tokens * avg_input_price / 1000000) + (estimated_output_tokens * avg_output_price / 1000000)
            
            # 从model_tokens统计中估算各模型成本
            for model_name, tokens in config_instance.usage_stats["model_tokens"].items():
                model_price = all_model_prices.get(model_name, default_model_price)
                # 假设输入输出token比例为1:3
                input_ratio = 0.25
                output_ratio = 0.75
                estimated_model_cost = (tokens * input_ratio * model_price['input'] / 1000000) + (tokens * output_ratio * model_price['output'] / 1000000)
                model_costs[model_name] = estimated_model_cost
            
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
                # 移除request_history
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