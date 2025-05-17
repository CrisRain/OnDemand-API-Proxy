import requests
import json
import base64
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Optional, Any

from utils import logger, mask_email
import config
from retry import with_retry

class OnDemandAPIClient:
    """OnDemand API 客户端，处理认证、会话管理和查询"""
    
    def __init__(self, email: str, password: str, client_id: str = "default_client"):
        """初始化客户端
        
        Args:
            email: OnDemand账户邮箱
            password: OnDemand账户密码
            client_id: 客户端标识符，用于日志记录
        """
        self.email = email
        self.password = password
        self.client_id = client_id
        self.token = ""
        self.refresh_token = ""
        self.user_id = ""
        self.company_id = ""
        self.session_id = ""
        self.base_url = "https://gateway.on-demand.io/v1"
        self.chat_base_url = "https://api.on-demand.io/chat/v1/client"  # 恢复为原始路径
        self.last_error: Optional[str] = None
        self.last_activity = datetime.now()
        self.lock = threading.RLock()  # 可重入锁，用于线程安全操作

        # 新增属性
        self._associated_user_identifier: Optional[str] = None
        self._associated_request_ip: Optional[str] = None
        self._current_request_context_hash: Optional[str] = None # 新增：用于暂存当前请求的上下文哈希
        
        # 隐藏密码的日志
        masked_email = mask_email(email)
        logger.info(f"已为 {masked_email} 初始化 OnDemandAPIClient (ID: {client_id})")
    
    def _log(self, message: str, level: str = "INFO"):
        """内部日志方法，使用结构化日志记录
        
        Args:
            message: 日志消息
            level: 日志级别
        """
        masked_email = mask_email(self.email)
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(f"[{self.client_id} / {masked_email}] {message}")
        self.last_activity = datetime.now()  # 更新最后活动时间

    def get_authorization(self) -> str:
        """生成登录用 Basic Authorization 头"""
        text = f"{self.email}:{self.password}"
        encoded = base64.b64encode(text.encode("utf-8")).decode("utf-8")
        return encoded
    
    def _do_request(self, method: str, url: str, headers: Dict[str, str],
                   data: Optional[Dict] = None, stream: bool = False,
                   timeout: int = None) -> requests.Response:
        """执行HTTP请求的实际逻辑，不包含重试
        
        Args:
            method: HTTP方法 (GET, POST等)
            url: 请求URL
            headers: HTTP头
            data: 请求数据
            stream: 是否使用流式传输
            timeout: 请求超时时间
            
        Returns:
            requests.Response对象
            
        Raises:
            requests.exceptions.RequestException: 请求失败
        """
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, stream=stream, timeout=timeout)
        elif method.upper() == 'POST':
            json_data = json.dumps(data) if data else None
            response = requests.post(url, data=json_data, headers=headers, stream=stream, timeout=timeout)
        else:
            raise ValueError(f"不支持的HTTP方法: {method}")
        
        response.raise_for_status()
        return response
    
    @with_retry()
    def sign_in(self, context: Optional[str] = None) -> bool:
        """登录以获取 token, refreshToken, userId, 和 companyId"""
        with self.lock:  # 线程安全
            self.last_error = None
            url = f"{self.base_url}/auth/user/signin"
            payload = {"accountType": "default"}
            headers = {
                'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0",
                'Accept': "application/json, text/plain, */*",
                'Content-Type': "application/json",
                'Authorization': f"Basic {self.get_authorization()}",  # 登录时使用 Basic 认证
                'Referer': "https://app.on-demand.io/"
            }
            if context:
                self._current_request_context_hash = context
            
            try:
                masked_email = mask_email(self.email)
                self._log(f"尝试登录 {masked_email}...")
                
                # 使用不带重试的_do_request，因为重试逻辑由装饰器处理
                response = self._do_request('POST', url, headers, payload, timeout=config.get_config_value('request_timeout'))
                data = response.json()
                
                if config.get_config_value('debug_mode'):
                    # 在调试模式下记录响应，但隐藏敏感信息
                    debug_data = data.copy()
                    if 'data' in debug_data and 'tokenData' in debug_data['data']:
                        debug_data['data']['tokenData']['token'] = '***REDACTED***'
                        debug_data['data']['tokenData']['refreshToken'] = '***REDACTED***'
                    self._log(f"登录原始响应: {json.dumps(debug_data, indent=2, ensure_ascii=False)}", "DEBUG")
                
                self.token = data.get('data', {}).get('tokenData', {}).get('token', '')
                self.refresh_token = data.get('data', {}).get('tokenData', {}).get('refreshToken', '')
                self.user_id = data.get('data', {}).get('user', {}).get('userId', '')
                self.company_id = data.get('data', {}).get('user', {}).get('default_company_id', '')
                
                if self.token and self.user_id and self.company_id:
                    self._log(f"登录成功。已获取必要的凭证。")
                    return True
                else:
                    self.last_error = "登录成功，但未能从响应中提取必要的字段。"
                    self._log(f"登录失败: {self.last_error}", level="ERROR")
                    return False
                    
            except requests.exceptions.RequestException as e:
                self.last_error = f"登录请求失败: {e}"
                self._log(f"登录失败: {e}", level="ERROR")
                raise  # 重新抛出异常，让装饰器处理重试
                
            except json.JSONDecodeError as e:
                self.last_error = f"登录 JSON 解码失败: {e}. 响应文本: {response.text if 'response' in locals() else 'N/A'}"
                self._log(self.last_error, level="ERROR")
                return False
                
            except Exception as e:
                self.last_error = f"登录过程中发生意外错误: {e}"
                self._log(self.last_error, level="ERROR")
                return False

    @with_retry()
    def refresh_token_if_needed(self) -> bool:
        """如果令牌过期或无效，则刷新令牌
        
        Returns:
            bool: 刷新成功返回True，否则返回False
        """
        with self.lock:  # 线程安全
            self.last_error = None
            if not self.refresh_token:
                self.last_error = "没有可用的 refresh token 来刷新令牌。"
                self._log(self.last_error, level="WARNING")
                return False

            url = f"{self.base_url}/auth/user/refresh_token"
            payload = {"data": {"token": self.token, "refreshToken": self.refresh_token}}
            headers = {'Content-Type': "application/json"}
            
            try:
                self._log("尝试刷新令牌...")
                
                # 使用不带重试的_do_request，因为重试逻辑由装饰器处理
                response = self._do_request('POST', url, headers, payload, timeout=config.get_config_value('request_timeout'))
                data = response.json()
                
                if config.get_config_value('debug_mode'):
                    # 在调试模式下记录响应，但隐藏敏感信息
                    debug_data = data.copy()
                    if 'data' in debug_data:
                        if 'token' in debug_data['data']:
                            debug_data['data']['token'] = '***REDACTED***'
                        if 'refreshToken' in debug_data['data']:
                            debug_data['data']['refreshToken'] = '***REDACTED***'
                    self._log(f"刷新令牌原始响应: {json.dumps(debug_data, indent=2, ensure_ascii=False)}", "DEBUG")
                
                new_token = data.get('data', {}).get('token', '')
                new_refresh_token = data.get('data', {}).get('refreshToken', '')  # OnDemand 可能不总返回新的 refresh token
                
                if new_token:
                    self.token = new_token
                    if new_refresh_token:  # 仅当返回了新的 refresh token 时才更新
                        self.refresh_token = new_refresh_token
                    self._log("令牌刷新成功。")
                    return True
                else:
                    self.last_error = "令牌刷新成功，但响应中没有新的 token。"
                    self._log(f"令牌刷新失败: {self.last_error}", level="ERROR")
                    return False
                    
            except requests.exceptions.RequestException as e:
                self.last_error = f"令牌刷新请求失败: {e}"
                self._log(f"令牌刷新失败: {e}", level="ERROR")
                
                # 如果是认证错误，可能需要完全重新登录
                if hasattr(e, 'response') and e.response is not None and e.response.status_code == 401:
                    self._log("令牌刷新返回401错误，可能需要完全重新登录", level="WARNING")
                
                raise  # 重新抛出异常，让装饰器处理重试
                
            except json.JSONDecodeError as e:
                self.last_error = f"令牌刷新 JSON 解码失败: {e}. 响应文本: {response.text if 'response' in locals() else 'N/A'}"
                self._log(self.last_error, level="ERROR")
                return False
                
            except Exception as e:
                self.last_error = f"令牌刷新过程中发生意外错误: {e}"
                self._log(self.last_error, level="ERROR")
                return False

    @with_retry()
    def create_session(self, external_user_id: str = "openai-adapter-user", external_context: Optional[str] = None) -> bool:
        """为聊天创建一个新会话
        
        Args:
            external_user_id: 外部用户ID前缀，会附加UUID确保唯一性
            external_context: 外部上下文哈希 (可选)
            
        Returns:
            bool: 创建成功返回True，否则返回False
        """
        with self.lock:  # 线程安全
            self.last_error = None
            if external_context:
                self._current_request_context_hash = external_context
            if not self.token or not self.user_id or not self.company_id:
                self.last_error = "创建会话缺少 token, user_id, 或 company_id。正在尝试登录。"
                self._log(self.last_error, level="WARNING")
                if not self.sign_in():  # 如果未登录，尝试登录
                    self.last_error = f"无法创建会话：登录失败。最近的客户端错误: {self.last_error}"
                    return False  # 如果登录失败，则无法继续

            url = f"{self.chat_base_url}/sessions"
            # 确保 externalUserId 对于每个会话是唯一的，以避免冲突
            unique_id = f"{external_user_id}-{uuid.uuid4().hex}"
            payload = {"externalUserId": unique_id, "pluginIds": []}
            headers = {
                'Content-Type': "application/json",
                'Authorization': f"Bearer {self.token}",  # 恢复为原始认证方式
                'x-company-id': self.company_id,
                'x-user-id': self.user_id
            }
            
            self._log(f"尝试创建会话，company_id: {self.company_id}, user_id: {self.user_id}, external_id: {unique_id}")
            
            try:
                try:
                    # 首先尝试创建会话，使用不带重试的_do_request
                    response = self._do_request('POST', url, headers, payload, timeout=config.get_config_value('request_timeout'))
                except requests.exceptions.HTTPError as e:
                    # 如果是401错误，尝试刷新令牌
                    if e.response.status_code == 401:
                        self._log("创建会话时令牌过期，尝试刷新...", level="INFO")
                        if self.refresh_token_if_needed():
                            headers['Authorization'] = f"Bearer {self.token}"  # 使用新令牌更新头
                            response = self._do_request('POST', url, headers, payload, timeout=config.get_config_value('request_timeout'))
                        else:  # 刷新失败，尝试完全重新登录
                            self._log("令牌刷新失败。尝试完全重新登录以创建会话。", level="WARNING")
                            if self.sign_in():
                                headers['Authorization'] = f"Bearer {self.token}"
                                response = self._do_request('POST', url, headers, payload, timeout=config.get_config_value('request_timeout'))
                            else:
                                self.last_error = f"会话创建失败：令牌刷新和重新登录均失败。最近的客户端错误: {self.last_error}"
                                self._log(self.last_error, level="ERROR")
                                return False
                    else:
                        # 其他HTTP错误，直接抛出
                        raise
                
                data = response.json()
                
                if config.get_config_value('debug_mode'):
                    self._log(f"创建会话原始响应: {json.dumps(data, indent=2, ensure_ascii=False)}", "DEBUG")
                
                session_id_val = data.get('data', {}).get('id', '')
                if session_id_val:
                    self.session_id = session_id_val
                    self._log(f"会话创建成功。会话 ID: {self.session_id}")
                    return True
                else:
                    self.last_error = f"会话创建成功，但响应中没有会话 ID。"
                    self._log(f"会话创建失败: {self.last_error}", level="ERROR")
                    return False
                    
            except requests.exceptions.RequestException as e:
                self.last_error = f"会话创建请求失败: {e}"
                self._log(f"会话创建失败: {e}", level="ERROR")
                raise  # 重新抛出异常，让装饰器处理重试
                
            except json.JSONDecodeError as e:
                self.last_error = f"会话创建 JSON 解码失败: {e}. 响应文本: {response.text if 'response' in locals() else 'N/A'}"
                self._log(self.last_error, level="ERROR")
                return False
                
            except Exception as e:
                self.last_error = f"会话创建过程中发生意外错误: {e}"
                self._log(self.last_error, level="ERROR")
                return False

    @with_retry()
    def send_query(self, query: str, endpoint_id: str = "predefined-claude-3.7-sonnet",
                  stream: bool = False, model_configs_input: Optional[Dict] = None,
                  full_query_override: Optional[str] = None) -> Dict:
        """向聊天会话发送查询，并处理流式或非流式响应
        
        Args:
            query: 查询文本 (如果提供了 full_query_override，则此参数被忽略)
            endpoint_id: OnDemand端点ID
            stream: 是否使用流式响应
            model_configs_input: 模型配置参数，如temperature、maxTokens等
            
        Returns:
            Dict: 包含响应内容或流对象的字典
        """
        with self.lock:  # 线程安全
            self.last_error = None
            
            # 会话检查和创建
            if not self.session_id:
                self.last_error = "没有可用的会话 ID。正在尝试创建新会话。"
                self._log(self.last_error, level="WARNING")
                if not self.create_session():
                    self.last_error = f"查询失败：会话创建失败。最近的客户端错误: {self.last_error}"
                    self._log(self.last_error, level="ERROR")
                    return {"error": self.last_error}
            
            if not self.token:
                self.last_error = "发送查询没有可用的 token。"
                self._log(self.last_error, level="ERROR")
                return {"error": self.last_error}

            url = f"{self.chat_base_url}/sessions/{self.session_id}/query"
            
            # 处理 query 输入
            current_query = ""
            if query is None:
                self._log("警告：查询内容为None，已替换为空字符串", level="WARNING")
            elif not isinstance(query, str):
                current_query = str(query)
                self._log(f"警告：查询内容不是字符串类型，已转换为字符串: {type(query)} -> {type(current_query)}", level="WARNING")
            else:
                current_query = query

            # 优先使用 full_query_override
            query_to_send = full_query_override if full_query_override is not None else current_query
            if full_query_override is not None:
                self._log(f"使用 full_query_override (长度: {len(full_query_override)}) 代替原始 query。", "DEBUG")

            payload = {
                "endpointId": endpoint_id,
                "query": query_to_send, # 使用处理后的 query 或 override
                "pluginIds": [],
                "responseMode": "stream" if stream else "sync",
                "debugMode": "on" if config.get_config_value('debug_mode') else "off",
                "fulfillmentOnly": False
            }
            
            # 处理 model_configs_input
            if model_configs_input:
                # 直接使用传入的 model_configs_input，只包含非 None 值
                # API 应该能处理额外的、非预期的配置项，或者忽略它们
                # 如果API严格要求特定字段，那么这里的逻辑需要更精确地过滤
                processed_model_configs = {k: v for k, v in model_configs_input.items() if v is not None}
                if processed_model_configs: # 只有当有有效配置时才添加modelConfigs
                    payload["modelConfigs"] = processed_model_configs
            
            self._log(f"最终的payload: {json.dumps(payload, ensure_ascii=False)}", level="DEBUG")
            
            headers = {
                'Content-Type': "application/json",
                'Authorization': f"Bearer {self.token}",
                'x-company-id': self.company_id
            }
            
            truncated_query_log = current_query[:100] + "..." if len(current_query) > 100 else current_query
            self._log(f"向端点 {endpoint_id} 发送查询 (stream={stream})。查询内容: {truncated_query_log}")

            try:
                response = self._do_request('POST', url, headers, payload, stream=True, timeout=config.get_config_value('stream_timeout'))

                if stream:
                    self._log("返回流式响应对象供外部处理")
                    return {"stream": True, "response_obj": response}
                else: # stream (方法参数) 为 False
                    full_answer = ""
                    try:
                        # 既然 _do_request 总是 stream=True，我们仍然需要消耗这个流。
                        # OnDemand API 在 responseMode="sync" 时，理论上应该直接返回完整内容。
                        
                        response_body = response.text # 读取整个响应体
                        response.close() # 确保连接关闭

                        self._log(f"非流式响应原始文本 (前500字符): {response_body[:500]}", "DEBUG")
                        
                        try:
                            # 优先尝试将整个响应体按单个JSON对象解析
                            data = json.loads(response_body)
                            if isinstance(data, dict):
                                if "answer" in data and isinstance(data["answer"], str):
                                    full_answer = data["answer"]
                                elif "content" in data and isinstance(data["content"], str): # 备选字段
                                    full_answer = data["content"]
                                elif data.get("eventType") == "fulfillment" and "answer" in data:
                                     full_answer = data.get("answer", "")
                                else:
                                    if not full_answer: # 避免覆盖已找到的答案
                                        self._log(f"非流式响应解析为JSON后，未在顶层或常见字段找到答案: {response_body[:200]}", "WARNING")
                            else:
                                self._log(f"非流式响应解析为JSON后，不是字典类型: {type(data)}", "WARNING")
                        
                        except json.JSONDecodeError:
                            # 如果直接解析JSON失败，再尝试按行解析SSE（作为后备）
                            self._log(f"非流式响应直接解析JSON失败，尝试按SSE行解析: {response_body[:200]}", "WARNING")
                            for line in response_body.splitlines():
                                if line:
                                    decoded_line = line #已经是str
                                    if decoded_line.startswith("data:"):
                                        json_str = decoded_line[len("data:"):].strip()
                                        if json_str == "[DONE]":
                                            break
                                        try:
                                            event_data = json.loads(json_str)
                                            if event_data.get("eventType", "") == "fulfillment":
                                                full_answer += event_data.get("answer", "")
                                        except json.JSONDecodeError:
                                            self._log(f"非流式后备SSE解析时 JSONDecodeError: {json_str}", level="WARNING")
                                            continue
                        
                        self._log(f"非流式响应接收完毕。聚合内容长度: {len(full_answer)}")
                        return {"stream": False, "content": full_answer}

                    except requests.exceptions.RequestException as e: # 这应该在 _do_request 中捕获并重试
                        self.last_error = f"非流式请求时发生错误: {e}"
                        self._log(self.last_error, level="ERROR")
                        # 如果 _do_request 抛异常到这里，说明重试也失败了
                        # raise e # 或者返回错误结构体，让上层处理
                        return {"error": self.last_error, "stream": False, "content": ""}
                    except Exception as e:
                        self.last_error = f"非流式处理中发生意外错误: {e}"
                        self._log(self.last_error, level="ERROR")
                        return {"error": self.last_error, "stream": False, "content": ""}
            
            except requests.exceptions.RequestException as e:
                self.last_error = f"请求失败: {e}"
                self._log(f"查询失败: {e}", level="ERROR")
                raise
                
            except Exception as e:
                error_message = f"send_query 过程中发生意外错误: {e}"
                error_type = type(e).__name__
                self.last_error = error_message
                self._log(f"{error_message} (错误类型: {error_type})", level="CRITICAL")
                return {"error": str(e)}