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

# 类型提示声明，消除类型检查器报错（实际值通过 config.get_config_value 获取）
DEBUG_MODE: bool
REQUEST_TIMEOUT: int
MAX_RETRIES: int
RETRY_DELAY: int
STREAM_TIMEOUT: int

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
        self.retry_count = 0
        self.lock = threading.RLock()  # 可重入锁，用于线程安全操作
        
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
        """生成用于登录的 Basic Authorization 头部"""
        text = f"{self.email}:{self.password}"
        encoded = base64.b64encode(text.encode("utf-8")).decode("utf-8")
        return encoded
    
    def _make_request(self, method: str, url: str, headers: Dict[str, str],
                     data: Optional[Dict] = None, stream: bool = False,
                     timeout: int = None, retry: bool = True) -> requests.Response:
        """发送HTTP请求，处理重试逻辑
        
        Args:
            method: HTTP方法 (GET, POST等)
            url: 请求URL
            headers: HTTP头部
            data: 请求数据
            stream: 是否使用流式传输
            timeout: 请求超时时间
            retry: 是否在失败时重试
            
        Returns:
            requests.Response对象
            
        Raises:
            requests.exceptions.RequestException: 请求失败
        """
        retry_count = 0
        max_retries = config.get_config_value('MAX_RETRIES') if retry else 0
        
        while True:
            try:
                if method.upper() == 'GET':
                    response = requests.get(url, headers=headers, stream=stream, timeout=timeout)
                elif method.upper() == 'POST':
                    json_data = json.dumps(data) if data else None
                    response = requests.post(url, data=json_data, headers=headers, stream=stream, timeout=timeout)
                else:
                    raise ValueError(f"不支持的HTTP方法: {method}")
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count > max_retries:
                    raise
                
                # 根据错误类型决定是否重试
                if isinstance(e, requests.exceptions.ConnectionError):
                    retry_delay = config.get_config_value('RETRY_DELAY') * (2 ** (retry_count - 1))  # 指数退避
                    self._log(f"连接错误，{retry_delay}秒后重试 ({retry_count}/{max_retries}): {e}", "WARNING")
                    time.sleep(retry_delay)
                elif isinstance(e, requests.exceptions.Timeout):
                    retry_delay = config.get_config_value('RETRY_DELAY') * retry_count
                    self._log(f"请求超时，{retry_delay}秒后重试 ({retry_count}/{max_retries}): {e}", "WARNING")
                    time.sleep(retry_delay)
                elif hasattr(e, 'response') and e.response is not None and e.response.status_code >= 500:  # 服务器错误
                    retry_delay = config.get_config_value('RETRY_DELAY') * retry_count
                    self._log(f"服务器错误 {e.response.status_code}，{retry_delay}秒后重试 ({retry_count}/{max_retries})", "WARNING")
                    time.sleep(retry_delay)
                else:
                    # 其他错误不重试
                    raise
    
    def sign_in(self) -> bool:
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
            
            try:
                masked_email = mask_email(self.email)
                self._log(f"尝试登录 {masked_email}...")
                
                response = self._make_request('POST', url, headers, payload, timeout=config.get_config_value('REQUEST_TIMEOUT'))
                data = response.json()
                
                if config.get_config_value('DEBUG_MODE'):
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
                    self.retry_count = 0  # 重置重试计数
                    return True
                else:
                    self.last_error = "登录成功，但未能从响应中提取必要的字段。"
                    self._log(f"登录失败: {self.last_error}", level="ERROR")
                    return False
                    
            except requests.exceptions.RequestException as e:
                self.last_error = f"登录请求失败: {e}"
                self._log(f"登录失败: {e}", level="ERROR")
                return False
                
            except json.JSONDecodeError as e:
                self.last_error = f"登录 JSON 解码失败: {e}. 响应文本: {response.text if 'response' in locals() else 'N/A'}"
                self._log(self.last_error, level="ERROR")
                return False
                
            except Exception as e:
                self.last_error = f"登录过程中发生意外错误: {e}"
                self._log(self.last_error, level="ERROR")
                return False

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
                
                # 使用通用请求方法，支持重试
                response = self._make_request('POST', url, headers, payload, timeout=config.get_config_value('REQUEST_TIMEOUT'))
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
                    self.retry_count = 0  # 重置重试计数
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
                
                return False
                
            except json.JSONDecodeError as e:
                self.last_error = f"令牌刷新 JSON 解码失败: {e}. 响应文本: {response.text if 'response' in locals() else 'N/A'}"
                self._log(self.last_error, level="ERROR")
                return False
                
            except Exception as e:
                self.last_error = f"令牌刷新过程中发生意外错误: {e}"
                self._log(self.last_error, level="ERROR")
                return False

    def create_session(self, external_user_id: str = "openai-adapter-user") -> bool:
        """为聊天创建一个新会话
        
        Args:
            external_user_id: 外部用户ID前缀，会附加UUID确保唯一性
            
        Returns:
            bool: 创建成功返回True，否则返回False
        """
        with self.lock:  # 线程安全
            self.last_error = None
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
                    # 首先尝试创建会话
                    response = self._make_request('POST', url, headers, payload, timeout=config.get_config_value('REQUEST_TIMEOUT'))
                except requests.exceptions.HTTPError as e:
                    # 如果是401错误，尝试刷新令牌
                    if e.response.status_code == 401:
                        self._log("创建会话时令牌过期，尝试刷新...", level="INFO")
                        if self.refresh_token_if_needed():
                            headers['Authorization'] = f"Bearer {self.token}"  # 使用新令牌更新头部
                            response = self._make_request('POST', url, headers, payload, timeout=config.get_config_value('REQUEST_TIMEOUT'))
                        else:  # 刷新失败，尝试完全重新登录
                            self._log("令牌刷新失败。尝试完全重新登录以创建会话。", level="WARNING")
                            if self.sign_in():
                                headers['Authorization'] = f"Bearer {self.token}"
                                response = self._make_request('POST', url, headers, payload, timeout=config.get_config_value('REQUEST_TIMEOUT'))
                            else:
                                self.last_error = f"会话创建失败：令牌刷新和重新登录均失败。最近的客户端错误: {self.last_error}"
                                self._log(self.last_error, level="ERROR")
                                return False
                    else:
                        # 其他HTTP错误，直接抛出
                        raise
                
                data = response.json()
                
                if config.get_config_value('DEBUG_MODE'):
                    self._log(f"创建会话原始响应: {json.dumps(data, indent=2, ensure_ascii=False)}", "DEBUG")
                
                session_id_val = data.get('data', {}).get('id', '')
                if session_id_val:
                    self.session_id = session_id_val
                    self._log(f"会话创建成功。会话 ID: {self.session_id}")
                    self.retry_count = 0  # 重置重试计数
                    return True
                else:
                    self.last_error = f"会话创建成功，但响应中没有会话 ID。"
                    self._log(f"会话创建失败: {self.last_error}", level="ERROR")
                    return False
                    
            except requests.exceptions.RequestException as e:
                self.last_error = f"会话创建请求失败: {e}"
                self._log(f"会话创建失败: {e}", level="ERROR")
                return False
                
            except json.JSONDecodeError as e:
                self.last_error = f"会话创建 JSON 解码失败: {e}. 响应文本: {response.text if 'response' in locals() else 'N/A'}"
                self._log(self.last_error, level="ERROR")
                return False
                
            except Exception as e:
                self.last_error = f"会话创建过程中发生意外错误: {e}"
                self._log(self.last_error, level="ERROR")
                return False

    def send_query(self, query: str, endpoint_id: str = "predefined-claude-3.7-sonnet",
                  stream: bool = False, model_configs: Optional[Dict] = None) -> Dict:
        """向聊天会话发送查询，并处理流式或非流式响应
        
        Args:
            query: 查询文本
            endpoint_id: OnDemand端点ID
            stream: 是否使用流式响应
            model_configs: 模型配置参数，如temperature、maxTokens等
            
        Returns:
            Dict: 包含响应内容或流对象的字典
        """
        with self.lock:  # 线程安全
            self.last_error = None
            
            # 会话检查和创建
            if not self.session_id:
                self.last_error = "没有可用的会话 ID。正在尝试创建新会话。"
                self._log(self.last_error, level="WARNING")
                if not self.create_session():  # 如果没有会话，尝试创建一个
                    self.last_error = f"查询失败：会话创建失败。最近的客户端错误: {self.last_error}"
                    self._log(self.last_error, level="ERROR")
                    return {"error": self.last_error}
            
            # 令牌检查
            if not self.token:  # 理论上 create_session 会处理，但作为安全检查
                self.last_error = "发送查询没有可用的 token。"
                self._log(self.last_error, level="ERROR")
                return {"error": self.last_error}

            # 准备请求
            url = f"{self.chat_base_url}/sessions/{self.session_id}/query"  # 确认路径与文档一致
            
            # 默认模型配置
            default_model_configs = {
                "maxTokens": 4000,  # 根据模型调整
                "temperature": 0.7,  # 根据需求调整
            }
            
            # 合并用户提供的模型配置（如果有）
            if model_configs:
                default_model_configs.update(model_configs)
            
            payload = {
                "endpointId": endpoint_id,
                "query": query,
                "pluginIds": [],
                "reasoningMode": "high",
                "responseMode": "stream" if stream else "sync",
                "debugMode": "off" if not config.get_config_value('DEBUG_MODE') else "on",  # 根据全局调试模式设置
                "modelConfigs": default_model_configs,
                "fulfillmentOnly": False
            }
            
            headers = {
                'Content-Type': "application/json",
                'Authorization': f"Bearer {self.token}",  # 恢复为原始认证方式
                'x-company-id': self.company_id
            }
            
            # 记录查询信息
            truncated_query = query[:100] + "..." if len(query) > 100 else query
            self._log(f"向端点 {endpoint_id} 发送查询 (stream={stream})。查询内容: {truncated_query}")

            try:
                try:
                    # 对于查询，使用较长的超时时间，并始终启用 stream=True 以便 requests 库正确处理流
                    response = self._make_request('POST', url, headers, payload, stream=True,
                                                timeout=config.get_config_value('STREAM_TIMEOUT'), retry=True)
                except requests.exceptions.HTTPError as e:
                    # 如果是401错误，尝试刷新令牌
                    if e.response.status_code == 401:
                        self._log("查询时令牌过期，尝试刷新...", level="INFO")
                        if self.refresh_token_if_needed():
                            headers['Authorization'] = f"Bearer {self.token}"  # 恢复为原始认证方式
                            response = self._make_request('POST', url, headers, payload, stream=True,
                                                        timeout=config.get_config_value('STREAM_TIMEOUT'), retry=True)
                        else:  # 刷新失败，尝试完全重新登录并重新创建会话
                            self._log("令牌刷新失败。尝试完全重新登录并重新创建会话以进行查询。", level="WARNING")
                            if self.sign_in() and self.create_session():  # 重新登录并重新创建会话
                                headers['Authorization'] = f"Bearer {self.token}"  # 恢复为原始认证方式
                                # 重要：session_id 可能已更改，更新 URL
                                url = f"{self.chat_base_url}/sessions/{self.session_id}/query"
                                response = self._make_request('POST', url, headers, payload, stream=True,
                                                            timeout=config.get_config_value('STREAM_TIMEOUT'), retry=True)
                            else:
                                self.last_error = f"查询失败：令牌刷新和重新登录/重新创建会话均失败。最近的客户端错误: {self.last_error}"
                                self._log(self.last_error, level="ERROR")
                                return {"error": self.last_error}
                    else:
                        # 其他HTTP错误，直接抛出
                        raise

                # 处理响应
                if stream:  # 如果外部请求的是流式
                    self._log("返回流式响应对象供外部处理")
                    return {"stream": True, "response_obj": response}  # 返回原始响应对象供外部流式处理
                else:  # 如果外部请求的是非流式 (同步)
                    # on-demand.io 的 "sync" 模式仍然是 SSE 流，只是期望客户端收集所有事件
                    full_answer = ""
                    for line in response.iter_lines():  # iter_lines 自动处理分块和解码
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith("data:"):
                                json_str = decoded_line[len("data:"):].strip()
                                if json_str == "[DONE]":
                                    break  # 到达流末尾
                                try:
                                    event_data = json.loads(json_str)
                                    if event_data.get("eventType", "") == "fulfillment":
                                        full_answer += event_data.get("answer", "")
                                except json.JSONDecodeError:
                                    self._log(f"非流式聚合时 JSONDecodeError: {json_str}", level="WARNING")
                                    continue  # 跳过无法解析的行
                    
                    self._log(f"非流式响应接收完毕。长度: {len(full_answer)}")
                    return {"stream": False, "content": full_answer}

            except requests.exceptions.RequestException as e:
                self.last_error = f"查询请求失败: {e}"
                self._log(f"查询失败: {e}", level="ERROR")
                return {"error": str(e)}
                
            except Exception as e:  # 捕获处理过程中的其他意外错误
                self.last_error = f"send_query 过程中发生意外错误: {e}"
                self._log(self.last_error, level="CRITICAL")
                return {"error": str(e)}