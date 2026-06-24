# Anthropic / OpenAI Adapter

本项目是一个 FastAPI 本地适配器：

- OpenAI client / OpenCode: `POST /v1/chat/completions`
- Anthropic client / Claude Code: `POST /v1/messages`
- 上游：通过动态加载的 provider 接入 OpenAI-compatible chat endpoint
- 支持 HTTP/2 SSE streaming、request_id、tool calling 转换、上游非标准错误归一化、body limit、日志降噪。

## 启动

```bash
pip install -r requirements.txt
cp .env.example .env
python -m app.main
```

默认端口：`http://127.0.0.1:18002`

## Provider 加载机制

通用配置只负责声明要加载哪个 provider 类：

```env
PROVIDER_CLASS=app.provider.openai_compatible.OpenAICompatibleProvider
PROVIDER_TIMEOUT=300000
```

启动时，`app.provider.loader.load_provider()` 会动态导入该类，并把 provider 实例注入到 OpenAI / Anthropic adapter 中。

这样通用入口不需要知道具体上游的 base URL、模型接口、鉴权 header 或 token 获取方式。不同上游可以通过不同 provider 类横向对比和 Debug。

## 自定义 Provider

每个 provider 建议放在 `app/provider/` 下的独立文件里，并提供独立类。`BaseProvider` 提供默认接口形状和 header 构建逻辑，`OpenAICompatibleProvider` 提供常规 OpenAI-compatible HTTP 实现。相似 provider 只覆盖少量类属性；差异较大的 provider 可以重写 `headers()` 做请求级 header 注入。

```python
from app.provider.openai_compatible import OpenAICompatibleProvider


class MyProvider(OpenAICompatibleProvider):
    BASE_URL = "https://your-upstream.example"
    AUTH_TOKEN = "your-token"
    HEADERS = {"X-Custom-Header": "value"}

    def headers(self, *, request_id: str | None = None, stream: bool = False) -> dict[str, str]:
        headers = super().headers(request_id=request_id, stream=stream)
        headers["X-Request-ID"] = request_id or "-"
        return headers
```

然后在 `.env` 中切换：

```env
PROVIDER_CLASS=app.provider.my_provider.MyProvider
PROVIDER_TIMEOUT=300000
```

## DeepSeek Provider

项目内置了一个测试用 DeepSeek provider：

```env
PROVIDER_CLASS=app.provider.deepseek.DeepSeekProvider
PROVIDER_TIMEOUT=300000
```

然后在 `app/provider/deepseek.py` 里填写：

```python
AUTH_TOKEN = "你的 DeepSeek API Key"
```

## GLM Provider

项目内置了一个测试用 GLM provider：

```env
PROVIDER_CLASS=app.provider.glm.GLMProvider
PROVIDER_TIMEOUT=300000
```

然后在 `app/provider/glm.py` 里填写：

```python
AUTH_TOKEN = "你的 GLM API Key"
```

## 验证

```bash
python -m compileall app
pytest -q
```

## 重要说明

- `app/tools/` 是项目通用工具，不是模型 tool calling。
- 模型 tool calling 协议转换在 `app/protocol/tool_calling.py`。
- `main.py` 是唯一装配入口。
- `gateway/` 只通过 `request.app.state` 使用 provider/adapter，不直接实例化 provider。


## 启动

- python -m uvicorn app.main:app --host 0.0.0.0 --port 18002 --reload
- python -m app.main