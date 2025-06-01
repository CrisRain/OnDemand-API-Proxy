import requests
import json
import base64
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Optional, Any, Tuple # Added Tuple

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
                   timeout: Optional[int] = None) -> requests.Response: # Changed int to Optional[int]
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
            
            response: Optional[requests.Response] = None # Initialize response
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
                self.last_error = f"登录 JSON 解码失败: {e}. 响应文本: {response.text if response else 'N/A'}"
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
            
            response: Optional[requests.Response] = None # Initialize response
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
                self.last_error = f"令牌刷新 JSON 解码失败: {e}. 响应文本: {response.text if response else 'N/A'}"
                self._log(self.last_error, level="ERROR")
                return False
                
            except Exception as e:
                self.last_error = f"令牌刷新过程中发生意外错误: {e}"
                self._log(self.last_error, level="ERROR")
                return False

    def _ensure_logged_in(self) -> bool:
        """
        Ensures the client is logged in by checking necessary tokens/IDs.
        Attempts to sign in if not logged in.
        Returns True if logged in or sign-in is successful, False otherwise.
        Sets self.last_error on failure.
        """
        if self.token and self.user_id and self.company_id:
            return True

        original_last_error = self.last_error # Preserve previous error if any
        self.last_error = "操作前检查：缺少 token, user_id, 或 company_id。正在尝试登录。"
        self._log(self.last_error, level="WARNING")
        
        if not self.sign_in(context=self._current_request_context_hash):
            # self.sign_in() will set its own self.last_error.
            # We create a more specific error message for the context of the calling function.
            current_action_error = f"前置登录失败，无法继续操作。最近的客户端错误: {self.last_error}"
            if original_last_error: # If there was an error before this check
                 self.last_error = f"{original_last_error}. Kemudian: {current_action_error}"
            else:
                 self.last_error = current_action_error
            # No need to log here as sign_in already logs its failure.
            return False
        return True

    def _handle_request_auth_error(self,
                                   http_method: str,
                                   url: str,
                                   headers: Dict[str, str],
                                   payload: Optional[Dict],
                                   timeout: Optional[int]) -> requests.Response:
        """
        Handles an authentication error (401) by attempting to refresh the token,
        then attempting a full sign-in, and retrying the original request.
        Re-raises an exception if all attempts fail.
        """
        self._log(f"{url} 请求遇到401错误，尝试刷新令牌...", level="INFO") # Added URL to log
        if self.refresh_token_if_needed():
            headers['Authorization'] = f"Bearer {self.token}"
            self._log(f"令牌刷新成功，使用新令牌重试 {url}。", level="INFO") # Added URL
            return self._do_request(http_method, url, headers, payload, timeout=timeout)
        else:
            self._log(f"令牌刷新失败 {url}。尝试完全重新登录。", level="WARNING") # Added URL
            if self.sign_in(context=self._current_request_context_hash):
                headers['Authorization'] = f"Bearer {self.token}"
                self._log(f"重新登录成功，使用新令牌重试 {url}。", level="INFO") # Added URL
                return self._do_request(http_method, url, headers, payload, timeout=timeout)
            else:
                self.last_error = f"{url} 的请求认证失败：令牌刷新和重新登录均失败。最近的客户端错误: {self.last_error}"
                self._log(self.last_error, level="ERROR")
                # Raising an exception here will be caught by the @with_retry decorator,
                # or by the calling function's RequestException handler if retries are exhausted.
                # This is consistent with how other _do_request failures are handled.
                raise requests.exceptions.HTTPError(self.last_error, response=None) # Or a more specific custom exception

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
            
            # Ensure client is logged in before proceeding
            if not self._ensure_logged_in():
                # _ensure_logged_in handles setting self.last_error and logging
                # We might want to make the error message more specific to create_session context here
                self.last_error = f"无法创建会话: {self.last_error}" # Append to existing error
                return False

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
            
            response: Optional[requests.Response] = None # Initialize response
            try:
                try:
                    # 首先尝试创建会话，使用不带重试的_do_request
                    response = self._do_request('POST', url, headers, payload, timeout=config.get_config_value('request_timeout'))
                except requests.exceptions.HTTPError as e:
                    if hasattr(e, 'response') and e.response is not None and e.response.status_code == 401:
                        # 调用新的辅助函数处理401错误并重试
                        # The helper will re-raise if it can't recover, which will be caught by the outer RequestException handler
                        response = self._handle_request_auth_error('POST', url, headers, payload, config.get_config_value('request_timeout'))
                    else:
                        # 其他HTTP错误，直接抛出，让 @with_retry 处理
                        raise
                
                # 确保 response 对象在后续代码中是有效的 Response 对象
                # 如果 _handle_request_auth_error 成功，它会返回一个新的 response
                # 如果初始的 _do_request 成功，response 也已经设置好了
                # 如果发生其他类型的 HTTPError (非401) 且被上面 else raise，这里不会执行
                # 如果 _handle_request_auth_error 内部认证失败并抛出异常，也会被外层捕获
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
                self.last_error = f"会话创建 JSON 解码失败: {e}. 响应文本: {response.text if response else 'N/A'}"
                self._log(self.last_error, level="ERROR")
                return False
                
            except Exception as e:
                self.last_error = f"会话创建过程中发生意外错误: {e}"
                self._log(self.last_error, level="ERROR")
                return False

    def _handle_sync_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handles a non-streamed (synchronous) response that was fetched in stream mode.
        Aggregates content and handles parsing.
        """
        full_answer = ""
        try:
            # Since _do_request might be called with stream=True even for sync mode,
            # we need to consume the stream here.
            response_body = response.text # Read the entire response body
            response.close() # Ensure the connection is closed

            self._log(f"非流式响应原始文本 (前500字符): {response_body[:500]}", "DEBUG")
            
            try:
                # Attempt to parse the entire response body as a single JSON object first
                data = json.loads(response_body)
                if isinstance(data, dict):
                    if "answer" in data and isinstance(data["answer"], str):
                        full_answer = data["answer"]
                    elif "content" in data and isinstance(data["content"], str): # Fallback field
                        full_answer = data["content"]
                    elif data.get("eventType") == "fulfillment" and "answer" in data:
                         full_answer = data.get("answer", "")
                    else:
                        if not full_answer: # Avoid overwriting already found answer
                            self._log(f"非流式响应解析为JSON后，未在顶层或常见字段找到答案: {response_body[:200]}", "WARNING")
                else:
                    self._log(f"非流式响应解析为JSON后，不是字典类型: {type(data)}", "WARNING")
            
            except json.JSONDecodeError:
                # If direct JSON parsing fails, try parsing line-by-line (SSE fallback)
                self._log(f"非流式响应直接解析JSON失败，尝试按SSE行解析: {response_body[:200]}", "WARNING")
                for line in response_body.splitlines():
                    if line:
                        # decoded_line = line # Already a str from splitlines()
                        if line.startswith("data:"): # Use line directly
                            json_str = line[len("data:"):].strip()
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

        except requests.exceptions.RequestException as e:
            self.last_error = f"非流式请求时发生错误: {e}"
            self._log(self.last_error, level="ERROR")
            return {"error": self.last_error, "stream": False, "content": ""}
        except Exception as e:
            self.last_error = f"非流式处理中发生意外错误: {e}"
            self._log(self.last_error, level="ERROR")
            return {"error": self.last_error, "stream": False, "content": ""}

    def _prepare_query_payload(self,
                               original_query: str, # Renamed to avoid confusion with internal var
                               endpoint_id: str,
                               stream_param: bool,
                               model_configs_input: Optional[Dict],
                               full_query_override: Optional[str]
                               ) -> Tuple[Dict[str, Any], str]:
        """
        Prepares the payload for the send_query method.
        Handles query processing, override, and model configurations.
        Returns the payload dictionary and the current_query_text string for logging.
        """
        current_query_text = ""
        if original_query is None:
            self._log("警告：查询内容为None，已替换为空字符串", level="WARNING")
            # current_query_text remains ""
        elif not isinstance(original_query, str):
            current_query_text = str(original_query)
            self._log(f"警告：查询内容不是字符串类型，已转换为字符串: {type(original_query)} -> {type(current_query_text)}", level="WARNING")
        else:
            current_query_text = original_query

        query_to_send = full_query_override if full_query_override is not None else current_query_text
        if full_query_override is not None:
            # Log the length of the override, not the original query
            self._log(f"使用 full_query_override (长度: {len(full_query_override)}) 代替原始 query (长度: {len(current_query_text)}).", "DEBUG")


        payload = {
            "endpointId": endpoint_id,
            "query": query_to_send,
            "pluginIds": [],
            "responseMode": "stream" if stream_param else "sync",
            "debugMode": "on" if config.get_config_value('debug_mode') else "off",
            "fulfillmentOnly": False
        }

        if model_configs_input:
            processed_model_configs = {k: v for k, v in model_configs_input.items() if v is not None}
            if processed_model_configs:
                payload["modelConfigs"] = processed_model_configs
        
        return payload, current_query_text


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
            
            payload, current_query_for_log = self._prepare_query_payload(
                query, endpoint_id, stream, model_configs_input, full_query_override
            )
            
            self._log(f"最终的payload: {json.dumps(payload, ensure_ascii=False)}", level="DEBUG")
            
            headers = {
                'Content-Type': "application/json",
                'Authorization': f"Bearer {self.token}",
                'x-company-id': self.company_id
            }
            
            # Use current_query_for_log which is the processed original query before override
            truncated_query_log = current_query_for_log[:100] + "..." if len(current_query_for_log) > 100 else current_query_for_log
            self._log(f"向端点 {endpoint_id} 发送查询 (stream={stream})。查询内容 (截断): {truncated_query_log}")

            try:
                response = self._do_request('POST', url, headers, payload, stream=True, timeout=config.get_config_value('stream_timeout'))

                if stream:
                    self._log("返回流式响应对象供外部处理")
                    return {"stream": True, "response_obj": response}
                else: # stream (方法参数) 为 False
                    return self._handle_sync_response(response)
            
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