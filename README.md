# OnDemand-API-Proxy 代理服务

一款基于 Flask 的 API 代理服务，提供兼容 OpenAI API 的接口，支持多种大型语言模型，实现多账户轮询和会话管理。

## 功能特点

- **兼容 OpenAI API**：提供标准的 `/v1/models` 和 `/v1/chat/completions` 接口
- **多模型支持**：支持 GPT-4o、Claude 3.7 Sonnet、Gemini 2.0 Flash 等多种模型
- **多轮对话**：通过会话管理保持对话上下文
- **账户轮换**：自动轮询使用多个 on-demand.io 账户，平衡负载
- **会话管理**：自动处理会话超时和重新连接
- **统计面板**：提供实时使用统计和图表展示
- **可配置的认证**：支持通过环境变量或配置文件设置 API 访问令牌
- **Docker 支持**：易于部署到 Hugging Face Spaces 或其他容器环境

## 支持的模型

服务支持以下模型（部分列表）：

| API 模型名称 | 实际使用模型 |
|------------|------------|
| `gpt-4o` | predefined-openai-gpt4o |
| `gpt-4o-mini` | predefined-openai-gpt4o-mini |
| `gpt-3.5-turbo` / `gpto3-mini` | predefined-openai-gpto3-mini |
| `gpt-4-turbo` / `gpt-4.1` | predefined-openai-gpt4.1 |
| `gpt-4.1-mini` | predefined-openai-gpt4.1-mini |
| `gpt-4.1-nano` | predefined-openai-gpt4.1-nano |
| `claude-3.5-sonnet` / `claude-3.7-sonnet` | predefined-claude-3.7-sonnet |
| `claude-3-opus` | predefined-claude-3-opus |
| `claude-3-haiku` | predefined-claude-3-haiku |
| `gemini-1.5-pro` / `gemini-2.0-flash` | predefined-gemini-2.0-flash |
| `deepseek-v3` | predefined-deepseek-v3 |
| `deepseek-r1` | predefined-deepseek-r1 |

## 配置说明

### 配置文件 (config.json)

配置文件支持以下参数：

```json
{
  "api_access_token": "你的自定义访问令牌",
  "accounts": [
    {"email": "账户1@example.com", "password": "密码1"},
    {"email": "账户2@example.com", "password": "密码2"}
  ],
  "session_timeout_minutes": 30,
  "max_retries": 3,
  "retry_delay": 1,
  "request_timeout": 30,
  "stream_timeout": 120,
  "rate_limit": 60,
  "debug_mode": false
}
```

### 环境变量

所有配置也可以通过环境变量设置：

- `API_ACCESS_TOKEN`: API 访问令牌
- `ONDEMAND_ACCOUNTS`: JSON 格式的账户信息
- `SESSION_TIMEOUT_MINUTES`: 会话超时时间（分钟）
- `MAX_RETRIES`: 最大重试次数
- `RETRY_DELAY`: 重试延迟（秒）
- `REQUEST_TIMEOUT`: 请求超时（秒）
- `STREAM_TIMEOUT`: 流式请求超时（秒）
- `RATE_LIMIT`: 速率限制（每分钟请求数）
- `DEBUG_MODE`: 调试模式（true/false）

## API 接口说明

### 获取模型列表

```
GET /v1/models
```

返回支持的模型列表，格式与 OpenAI API 兼容。

### 聊天补全

```
POST /v1/chat/completions
```

**请求头：**
```
Authorization: Bearer 你的API访问令牌
Content-Type: application/json
```

**请求体：**
```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "你好，请介绍一下自己。"}
  ],
  "temperature": 0.7,
  "max_tokens": 2000,
  "stream": false
}
```

**参数说明：**
- `model`: 使用的模型名称
- `messages`: 对话消息数组
- `temperature`: 温度参数（0-1）
- `max_tokens`: 最大生成令牌数
- `stream`: 是否使用流式响应
- `top_p`: 核采样参数（0-1）
- `frequency_penalty`: 频率惩罚（0-2）
- `presence_penalty`: 存在惩罚（0-2）

## 统计面板

访问根路径 `/` 可以查看使用统计面板，包括：

- 总请求数和成功率
- Token 使用统计
- 每日和每小时使用量图表
- 模型使用情况
- 最近请求历史

## 部署指南

### Hugging Face Spaces 部署（推荐）

1. **创建 Hugging Face 账户**：
   - 访问 [https://huggingface.co/](https://huggingface.co/) 注册账户

2. **创建 Space**：
   - 点击 [创建新的 Space](https://huggingface.co/new-space)
   - 填写 Space 名称
   - **重要**：选择 `Docker` 作为 Space 类型
   - 设置权限（公开或私有）

3. **上传代码**：
   - 将以下文件上传到你的 Space 代码仓库：
     - `app.py`（主程序）
     - `routes.py`（路由定义）
     - `config.py`（配置管理）
     - `auth.py`（认证模块）
     - `client.py`（客户端实现）
     - `utils.py`（工具函数）
     - `requirements.txt`（依赖列表）
     - `Dockerfile`（Docker 配置）
     - `templates/`（模板目录）
     - `static/`（静态资源目录）

4. **配置账户信息和 API 访问令牌**：
   - 进入 Space 的 "Settings" → "Repository secrets"
   - 添加 `ONDEMAND_ACCOUNTS` Secret：
     ```json
     {
       "accounts": [
         {"email": "你的邮箱1@example.com", "password": "你的密码1"},
         {"email": "你的邮箱2@example.com", "password": "你的密码2"}
       ]
     }
     ```
   - 添加 `API_ACCESS_TOKEN` Secret 设置自定义访问令牌
     - 如果不设置，将使用默认值 "sk-2api-ondemand-access-token-2025"

5. **可选配置**：
   - 添加其他环境变量如 `SESSION_TIMEOUT_MINUTES`、`RATE_LIMIT` 等

6. **完成部署**：
   - Hugging Face 会自动构建 Docker 镜像并部署你的 API
   - 访问你的 Space URL（如 `https://你的用户名-你的space名称.hf.space`）

### 本地部署

1. **克隆代码**：
   ```bash
   git clone https://github.com/你的用户名/ondemand-api-proxy.git
   cd ondemand-api-proxy
   ```

2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

3. **配置**：
   - 创建 `config.json` 文件：
     ```json
     {
       "api_access_token": "你的自定义访问令牌",
       "accounts": [
         {"email": "账户1@example.com", "password": "密码1"},
         {"email": "账户2@example.com", "password": "密码2"}
       ]
     }
     ```
   - 或设置环境变量

4. **启动服务**：
   ```bash
   python app.py
   ```
   
5. **访问服务**：
   - API 接口：`http://localhost:5000/v1/chat/completions`
   - 统计面板：`http://localhost:5000/`

### Docker 部署

```bash
# 构建镜像
docker build -t ondemand-api-proxy .

# 运行容器
docker run -p 7860:7860 \
  -e API_ACCESS_TOKEN="你的访问令牌" \
  -e ONDEMAND_ACCOUNTS='{"accounts":[{"email":"账户1@example.com","password":"密码1"}]}' \
  ondemand-api-proxy
```

## 客户端连接

### Cherry Studio 连接

1. 打开 Cherry Studio
2. 进入设置 → API 设置
3. 选择 "OpenAI API"
4. API 密钥填入你配置的 API 访问令牌
5. API 地址填入你的服务地址（如 `https://你的用户名-你的space名称.hf.space/v1`）

### 其他 OpenAI 兼容客户端

任何支持 OpenAI API 的客户端都可以连接到此服务，只需将 API 地址修改为你的服务地址即可。

## 故障排除

### 常见问题

1. **认证失败**：
   - 检查 API 访问令牌是否正确配置
   - 确认请求头中包含 `Authorization: Bearer 你的令牌`

2. **账户连接问题**：
   - 确认 on-demand.io 账户信息正确
   - 检查账户是否被限制或封禁

3. **模型不可用**：
   - 确认请求的模型名称在支持列表中
   - 检查 on-demand.io 是否支持该模型

4. **统计图表显示错误**：
   - 清除浏览器缓存后重试
   - 检查浏览器控制台是否有错误信息

## 安全建议

1. **永远不要**在代码中硬编码账户信息和访问令牌
2. 使用环境变量或安全的配置管理系统存储敏感信息
3. 定期更换 API 访问令牌
4. 限制 API 的访问范围，只允许受信任的客户端连接
5. 启用速率限制防止滥用

## 贡献与反馈

欢迎提交 Issue 和 Pull Request 来改进此项目。如有任何问题或建议，请随时联系。

## 许可证

本项目采用 MIT 许可证。
