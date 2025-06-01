# OnDemand API 代理服务

## 项目简介

本项目是一个基于 Python Flask 实现的 API 代理服务，旨在作为 [on-demand.io](https://on-demand.io/) 服务的中间层。它提供了与 OpenAI API 兼容的接口，使得现有应用可以无缝切换或同时使用 OnDemand 提供的多种大语言模型服务。

主要目标包括：
*   提供 OpenAI 兼容的 `/v1/chat/completions` 和 `/v1/models` API 端点。
*   支持多个 OnDemand 账户的配置，并实现账户轮询和自动切换机制，以应对单个账户的速率限制或额度问题。
*   通过 Bearer Token 实现对代理服务自身的访问认证。
*   内置请求速率限制功能，保护后端服务。
*   提供详细的用量统计功能，包括请求数、Token 消耗、成本估算等，并提供一个 Web 页面进行展示。
*   支持流式 (streaming) 和非流式响应。
*   通过 JSON 配置文件和环境变量提供灵活的配置选项。
*   集成了健壮的错误处理和自动重试机制，以提高服务的稳定性和可靠性。
*   提供可配置的日志系统，方便监控和问题排查。

## 主要特性

*   **OpenAI API 兼容**: 支持 `/v1/chat/completions` 和 `/v1/models` 端点，方便集成。
*   **多账户轮询与冷却**: 可配置多个 OnDemand 账户，当一个账户遇到速率限制 (429错误) 时，会自动切换到下一个可用账户，并将受限账户置于冷却期。
*   **API 访问认证**: 通过 Bearer Token 保护代理服务接口。
*   **请求速率限制**: 可配置每分钟的请求上限，防止滥用。
*   **用量统计与展示**:
    *   实时跟踪总请求数、成功/失败请求数。
    *   按模型、按账户、按天、按小时统计请求次数和 Token 使用量。
    *   基于配置的模型价格估算累计成本。
    *   提供 `/` 路径的 HTML 页面展示统计数据。
*   **流式响应**: 完全支持 OpenAI API 的流式响应模式。
*   **灵活配置**: 支持通过 `config.json` 文件和环境变量进行详细配置。
*   **自动重试**: 对后端 API 请求（如登录、创建会话、发送查询）实现了自动重试逻辑，能处理连接错误、超时、服务器错误 (5xx) 和速率限制错误 (429)。
*   **Token 计算**: 使用 `tiktoken` 库进行较准确的 Token 数量计算。
*   **可配置日志**: 日志可同时输出到控制台和指定文件，级别和格式可配置。

## 安装与设置

### 环境要求
*   Python 3.7+
*   pip (Python 包安装器)

### 安装步骤

1.  **克隆项目**:
    ```bash
    git clone <your_repository_url>
    cd ondemand-api-proxy
    ```

2.  **安装依赖**:
    项目依赖于 `Flask`, `requests`, `tiktoken` 等库。请通过 `requirements.txt` 文件安装：
    ```bash
    pip install -r requirements.txt
    ```
    *(注意: 请确保 `requirements.txt` 文件包含所有必要的依赖。根据代码分析，至少应包含 `Flask`, `requests`, `tiktoken`)*

3.  **创建配置文件**:
    在项目根目录下创建一个名为 `config.json` 的文件。这是项目的主要配置文件。

## 配置说明

项目配置可以通过 `config.json` 文件和环境变量进行。环境变量的优先级高于 `config.json` 中的同名配置。

### 1. `config.json` 文件

以下是一个 `config.json` 的结构示例及其主要配置项说明：

```json
{
  "accounts": [
    {
      "email": "your_ondemand_email_1@example.com",
      "password": "your_ondemand_password_1"
    },
    {
      "email": "your_ondemand_email_2@example.com",
      "password": "your_ondemand_password_2"
    }
  ],
  "api_access_token": "YOUR_SECRET_PROXY_ACCESS_TOKEN",
  "model_prices": {
    "gpt-3.5-turbo": {"input": 0.25, "output": 0.75},
    "gpt-4o": {"input": 1.25, "output": 5.00},
    "claude-3.5-sonnet": {"input": 1.50, "output": 7.50}
    // 根据需要添加更多模型及其价格 (美元/百万Tokens)
  },
  "default_model_price": {"input": 1.00, "output": 3.00},
  "rate_limit_per_minute": 60,
  "account_cooldown_seconds": 300,
  "ondemand_session_timeout_minutes": 30,
  "session_timeout_minutes": 3600,
  "max_retries": 5,
  "retry_delay": 3,
  "request_timeout": 45,
  "stream_timeout": 180,
  "debug_mode": false,
  "FLASK_DEBUG": false,
  "stats_file_path": "stats_data.json",
  "stats_backup_path": "stats_data_backup.json",
  "stats_save_interval": 300,
  "default_endpoint_id": "predefined-claude-4-sonnet"
}
```

**主要配置项解释**:

*   `accounts` (必需): 一个包含 OnDemand 账户凭据（`email` 和 `password`）的列表。至少需要一个账户。
*   `api_access_token` (必需): 访问此代理服务所需的 Bearer Token。客户端在请求时需要在 `Authorization` 头部提供此 Token。
*   `model_prices`: 一个字典，定义了不同模型的输入和输出价格（单位：美元/百万 Tokens），用于在统计页面估算成本。
*   `default_model_price`: 当请求的模型未在 `model_prices` 中定义时，使用的默认价格。
*   `rate_limit_per_minute`: 代理服务对每个 `api_access_token` 每分钟允许的最大请求数。
*   `account_cooldown_seconds`: 当一个 OnDemand 账户因为速率限制（如429错误）失败时，该账户将被置于冷却状态的秒数，在此期间不会被用于处理请求。
*   `ondemand_session_timeout_minutes`: OnDemand 平台会话的活跃超时时间（分钟）。(此配置项似乎在 `client.py` 中未直接使用，但存在于 `config.py` 默认配置中)。
*   `session_timeout_minutes`: 代理服务自身维护的会话（如 `client_sessions`）的超时时间（分钟）。(此配置项似乎在当前代码中主要影响 `_cleanup_user_sessions`，但该功能已被注释掉)。
*   `max_retries`: 对 OnDemand API 请求失败时的最大重试次数。
*   `retry_delay`: 重试之间的基础延迟时间（秒）。实际延迟时间会根据重试策略（如指数退避、线性退避）调整。
*   `request_timeout`: 普通 HTTP 请求的默认超时时间（秒）。
*   `stream_timeout`: 流式 HTTP 请求的默认超时时间（秒）。
*   `debug_mode`: 项目的通用调试模式开关 (布尔值)。会影响日志详细程度和某些内部行为。
*   `FLASK_DEBUG`: Flask 应用的调试模式开关 (布尔值)。会启用 Flask 的调试器和自动重载。
*   `stats_file_path`: 用量统计数据保存的文件路径 (默认为 `stats_data.json`)。
*   `stats_backup_path`: 统计数据备份文件的路径 (默认为 `stats_data_backup.json`)。
*   `stats_save_interval`: 统计数据自动保存到文件的时间间隔（秒，默认为300秒，即5分钟）。
*   `default_endpoint_id`: 当请求中沒有指定模型，或指定模型无法映射时，使用的默认 OnDemand `endpointId`。

### 2. 环境变量

以下环境变量可用于覆盖 `config.json` 中的设置或提供特定配置：

*   **账户信息**:
    *   `ONDEMAND_ACCOUNTS`: 一个 JSON 格式的字符串，用于提供账户列表。如果设置，它将覆盖 `config.json` 中的 `accounts`。
        *   示例: `ONDEMAND_ACCOUNTS='{"accounts": [{"email": "user1@example.com", "password": "pw1"}, {"email": "user2@example.com", "password": "pw2"}]}'`
*   **代理访问令牌**:
    *   `API_ACCESS_TOKEN`: 覆盖 `config.json` 中的 `api_access_token`。
*   **运行参数**:
    *   `PORT`: Flask 应用监听的端口号 (默认为 `7860`)。
    *   `FLASK_DEBUG`: 设置为 `'true'` (不区分大小写) 以启用 Flask 调试模式。覆盖 `config.json` 中的 `FLASK_DEBUG`。
*   **配置文件路径**:
    *   `APP_CONFIG_PATH`: 指定 `config.json` 文件的自定义路径。如果设置，则会加载此路径的配置文件。
*   **日志配置**:
    *   `LOG_PATH`: 日志文件的完整路径 (默认为 `/tmp/2api.log`)。
    *   `LOG_LEVEL`: 日志级别 (如 `DEBUG`, `INFO`, `WARNING`, `ERROR`, 默认为 `INFO`)。
    *   `LOG_FORMAT`: 日志格式字符串 (默认为 `%(asctime)s - %(name)s - %(levelname)s - %(message)s`)。
*   **其他配置项的环境变量形式**:
    *   `ONDEMAND_SESSION_TIMEOUT_MINUTES` (对应 `ondemand_session_timeout_minutes`)
    *   `SESSION_TIMEOUT_MINUTES` (对应 `session_timeout_minutes`)
    *   `MAX_RETRIES` (对应 `max_retries`)
    *   `RETRY_DELAY` (对应 `retry_delay`)
    *   `REQUEST_TIMEOUT` (对应 `request_timeout`)
    *   `STREAM_TIMEOUT` (对应 `stream_timeout`)
    *   `RATE_LIMIT` (对应 `rate_limit_per_minute`，注意名称差异，但代码中 `RateLimiter` 初始化时读取的是 `rate_limit`)
    *   `DEBUG_MODE` (对应 `debug_mode`, 设置为 `'true'` 或 `'false'`)

### 配置优先级
1.  环境变量
2.  `config.json` 文件中定义的值
3.  代码中定义的默认值

## 运行项目

### 开发模式
直接运行 `app.py` 文件：
```bash
python app.py
```
服务将默认在 `0.0.0.0:7860` 启动。您可以通过 `PORT` 环境变量更改端口，通过 `FLASK_DEBUG` 环境变量或 `config.json` 中的 `FLASK_DEBUG` 键启用 Flask 的调试模式。

### 生产模式
在生产环境中，建议使用更健壮的 WSGI 服务器，例如 Gunicorn：
```bash
gunicorn --workers 4 --bind 0.0.0.0:<PORT> app:app
```
请将 `<PORT>` 替换为您希望服务运行的实际端口号。`app:app` 指向 `app.py` 文件中的 Flask 应用实例。

## API 端点说明

### 1. 聊天补全
*   **端点**: `POST /v1/chat/completions`
*   **描述**: 接收与 OpenAI ChatCompletion API 兼容的请求，并将其代理到 OnDemand 服务。
*   **认证**:
    *   请求头必须包含 `Authorization: Bearer <your_api_access_token>`，其中 `<your_api_access_token>` 是您在 `config.json` 或 `API_ACCESS_TOKEN` 环境变量中设置的代理访问令牌。
*   **请求体**:
    *   与 OpenAI `POST /v1/chat/completions` API 的请求体格式相同。
    *   必需字段: `messages` (一个消息对象列表), `model` (要使用的模型名称，如 "gpt-3.5-turbo", "claude-3.5-sonnet" 等，这些名称会通过内部映射到 OnDemand 的 `endpointId`)。
    *   可选参数: `stream` (布尔值，控制是否流式响应), `temperature`, `max_tokens`, `top_p` 等标准 OpenAI 参数。
*   **响应体**:
    *   如果 `stream: false` (或未提供)，返回与 OpenAI ChatCompletion API 相同的 JSON 结构。
    *   如果 `stream: true`，返回 Server-Sent Events (SSE) 流，每条事件与 OpenAI ChatCompletionChunk API 的格式兼容。
*   **示例请求 (curl)**:
    ```bash
    curl -X POST http://localhost:7860/v1/chat/completions \
    -H "Authorization: Bearer YOUR_SECRET_PROXY_ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "gpt-3.5-turbo",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
      ],
      "stream": false
    }'
    ```

### 2. 列出模型
*   **端点**: `GET /v1/models`
*   **描述**: 返回代理服务当前配置支持的、与 OpenAI 兼容的模型列表。列表内容基于 `config.py` 中 `_model_mapping` 的键。
*   **认证**: 此端点当前**不需要** `Authorization` 头部即可访问。
*   **响应体**:
    *   与 OpenAI `GET /v1/models` API 的响应体格式相同，包含一个模型对象列表。
    *   示例:
      ```json
      {
        "object": "list",
        "data": [
          {
            "id": "gpt-3.5-turbo",
            "object": "model",
            "created": 1677610602,
            "owned_by": "on-demand.io"
          },
          // ... 其他模型
        ]
      }
      ```

### 3. 用量统计页面
*   **端点**: `GET /`
*   **描述**: 在浏览器中访问此端点会显示一个 HTML 页面，其中包含详细的 API 用量统计信息，如总请求数、成功/失败请求、Token 使用量、成本估算、各模型使用情况等。
*   **认证**: 无。

### 4. 健康检查
*   **端点**: `GET /health`
*   **描述**: 返回服务的健康状态。如果服务正常运行，返回 `{"status": "ok", "message": "2API服务运行正常"}` 和 HTTP 200 状态码。
*   **认证**: 无。

### 5. 手动保存统计
*   **端点**: `POST /save_stats`
*   **描述**: 手动触发将当前内存中的用量统计数据保存到配置的 `stats_file_path` 文件中。
*   **认证**: 无。成功后会重定向到统计页面 (`/`)。

## 高级特性说明

### 账户轮询与冷却机制
当通过 `/v1/chat/completions` 接口向 OnDemand API 发送请求时，如果某个 OnDemand 账户因为请求过于频繁而收到 429 (Too Many Requests) 错误，代理服务会自动执行以下操作：
1.  将当前发生 429 错误的账户标记为“冷却中”，在配置的 `account_cooldown_seconds` 时间内不再使用该账户。
2.  从 `config.json` 或 `ONDEMAND_ACCOUNTS` 环境变量配置的账户列表中选择下一个可用的账户。
3.  使用新账户重新登录 OnDemand 并创建新的会话。
4.  使用新账户和新会话重试原始请求。
此机制有助于在单个账户达到速率限制时，服务仍能继续处理请求（如果配置了多个可用账户）。

### 自动重试
代理服务在与 OnDemand API 通信时，内置了自动重试机制，可以处理以下类型的临时性错误：
*   **连接错误** (`requests.exceptions.ConnectionError`): 采用指数退避策略重试。
*   **请求超时** (`requests.exceptions.Timeout`): 采用线性退避策略重试。
*   **服务器错误** (HTTP 5xx): 采用线性退避策略重试。
*   **速率限制错误** (HTTP 429): 如上所述，会触发账户切换和立即重试。
最大重试次数和基础重试延迟可以通过 `config.json` 中的 `max_retries` 和 `retry_delay` 或相应的环境变量进行配置。

### Token 计算
为了进行用量统计和成本估算，代理服务使用 `tiktoken` 库来计算请求和响应中的 token 数量。计算会根据请求中指定的模型（或默认模型）选择合适的编码器，以尽可能准确地模拟 OpenAI 的计费方式。

## 日志
项目使用 Python 的 `logging` 模块进行日志记录。
*   日志会同时输出到控制台和文件。
*   可以通过以下环境变量配置日志行为：
    *   `LOG_PATH`: 日志文件的完整路径 (默认: `/tmp/2api.log`)。
    *   `LOG_LEVEL`: 日志记录的最低级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL; 默认: `INFO`)。
    *   `LOG_FORMAT`: 日志消息的格式 (默认: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`)。

## 依赖项
主要依赖项包括：
*   Flask
*   requests
*   tiktoken

请查看 `requirements.txt` 获取完整的依赖列表。
