# llm-adapter simple

一个本地 FastAPI 适配器：

- 对外兼容 OpenAI `/v1/chat/completions`
- 对外兼容 Anthropic `/v1/messages`
- 上游调用 OpenAI-compatible 接口
- 支持流式输出、tool calling、thinking / reasoning、usage、错误转换
- 支持客户端 header 黑名单透传
- 支持 provider 自定义 header，例如 `X-Auth-Token`

这版重点是**简化配置，不砍功能**：核心就是自定义 chat 接口、自定义 models 接口、自定义 header，然后保留 OpenAI / Anthropic 客户端兼容。

## 核心边界

```text
config.py
    只管本地 adapter 怎么启动，以及 provider HTTP 默认值。

provider/*.py
    完整描述一个上游怎么请求：BASE_URL、endpoint、headers、timeout、verify、trust_env、http2。

adapter/*.py
    只管 OpenAI / Anthropic 两种客户端入口。

protocol/*.py
    只管 OpenAI / Anthropic 格式转换、流式事件、tool calling、reasoning、usage、错误。
```

## .env 最小配置

```env
HOST=127.0.0.1
PORT=18002
SERVER_API_KEY=sk-llm-adapter

PROVIDER_CLASS=app.provider.xxx.xxxProvider

REQUEST_TIMEOUT=120000
BODY_LIMIT=20mb
LOG_LEVEL=info
DEBUG_STREAM=false
DROP_THINKING_BLOCK=false

HTTP_TIMEOUT=300
HTTP_CONNECT_TIMEOUT=30
HTTP_VERIFY_SSL=true
HTTP_TRUST_ENV=false
HTTP2=true
HTTP_MAX_CONNECTIONS=200
HTTP_MAX_KEEPALIVE_CONNECTIONS=40
HTTP_KEEPALIVE_EXPIRY=30

xxx_AUTH_TOKEN=your-token
```

`PROVIDER_CLASS` 是启动时选择上游 provider 的唯一入口。不要再把上游地址、endpoint、鉴权 header 写进全局 config。

## 自定义 provider 示例

```python
from __future__ import annotations

from typing import Mapping

from app.config import env
from app.provider.openai_compatible import OpenAICompatibleProvider


class xxxProvider(OpenAICompatibleProvider):
    BASE_URL = "https://xxxcli.com/xxxPro"
    CHAT_ENDPOINT = "/chat/completions"
    MODELS_ENDPOINT = "/chat/models"

    VERIFY_SSL = False
    TRUST_ENV = True
    HTTP2 = True
    TIMEOUT = 300.0

    CLIENT_HEADER_BLACKLIST = OpenAICompatibleProvider.CLIENT_HEADER_BLACKLIST + (
        # "cookie",
        # "authorization",
    )

    def provider_headers(self, client_headers: Mapping[str, str]) -> dict[str, str]:
        token = env("xxx_AUTH_TOKEN")
        return {"X-Auth-Token": token} if token else {}
```

规则很简单：provider 类里写了什么，上游请求就用什么。`BaseProvider.__init__()` 不会再用 config 覆盖子类的 `BASE_URL / CHAT_ENDPOINT / MODELS_ENDPOINT / VERIFY_SSL / TRUST_ENV / HTTP2`。

## Header 合并规则

```text
基础 JSON/SSE headers
→ 客户端 headers 黑名单过滤后透传
→ provider.HEADERS
→ provider.provider_headers(client_headers)
→ X-Request-ID
```

默认黑名单过滤传输层 header，并且 `Accept` / `Content-Type` 由 adapter 根据 `stream` 统一设置：

```python
connection
accept
content-type
content-length
host
keep-alive
proxy-authenticate
proxy-authorization
te
trailer
transfer-encoding
upgrade
```

`Authorization`、`X-Auth-Token`、`Cookie` 是否透传由具体 provider 决定；某个上游不能接收时，在该 provider 里追加到 `CLIENT_HEADER_BLACKLIST`。`xxxProvider` 默认只使用 `X-Auth-Token` 认证，因此会过滤客户端侧 `authorization` 和 Anthropic / Stainless 元数据 header。

## 运行

```bash
pip install -r requirements.txt
cp .env.example .env
python -m app.main
```

OpenAI 客户端：

```bash
curl http://127.0.0.1:18002/v1/chat/completions \
  -H "Authorization: Bearer sk-llm-adapter" \
  -H "Content-Type: application/json" \
  -d '{"model":"demo","messages":[{"role":"user","content":"hi"}]}'
```

Anthropic 客户端：

```bash
curl http://127.0.0.1:18002/v1/messages \
  -H "x-api-key: sk-llm-adapter" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model":"demo","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}'
```

## 日志

启动时默认只打印 provider 摘要，避免把完整配置刷到 INFO 日志里：

```text
Loaded provider class=app.provider.xxx.xxxProvider
[startup] provider=xxxProvider base_url=... chat=/chat/completions models=/chat/modles verify_ssl=False trust_env=True http2=True headers=['x-auth-token']
```

请求上游前会打印最终 URL、model、stream、messages/tools 数量和 header names。敏感 header 值会打码，不打印完整 prompt，不打印 token 明文。

所有项目 logger 都支持结构化打印：

```python
log.debug_json("provider.describe", provider.describe())
log.info_json("startup.models", {"count": len(models), "sample": sample})
```

`LOG_LEVEL=debug` 时，`debug_json()` 会把 dict/list/tuple 格式化成缩进 JSON；INFO 级别下不会输出 `debug_json()` 内容。`DEBUG_STREAM=true` 时，上游请求 payload 也会输出，适合排查 Anthropic/OpenAI 转换问题；普通运行不建议开启。


### 断开与错误定位日志

这版补齐了三类关键日志：

```text
[req_xxx] upstream stream start status=200 elapsed=...
[req_xxx] upstream stream first_chunk elapsed=...
[req_xxx] upstream stream cancelled, closing response elapsed=...
[req_xxx] upstream stream closed cancelled=True status=200 chunks=... first_chunk=True elapsed=...
```

看到 `upstream stream closed`，就能确认 provider 层的 async generator 已经退出，`httpx.stream()` 的上下文也已经释放。

非流式上游 HTTP 错误会先尝试解析上游 JSON error，能保留 `rate_limit_error`、`authentication_error` 这类结构；解析不了时才 fallback 成 `upstream_http_error`。

模型列表也会打印 normalization 前后的结构摘要：

```text
[models] _normalize_models input {...}
[models] _normalize_models output {...}
```

这样可以确认上游 `/models` 返回的是 `data`、`models`、`result` 还是裸 list，以及最后实际暴露给客户端的模型数量和 sample。

## 保留能力

- OpenAI 非流式 / 流式
- Anthropic 非流式 / 流式
- OpenAI stream 转 Anthropic stream 生命周期
- tool call / tool use / tool result 转换
- thinking / reasoning 转换
- usage 转换
- 上游错误归一化
- 客户端 header 透传
- provider 自定义 header


## 取消语义

当前 HTTP stream 断开时，当前 upstream stream 会通过 `aclosing()` 和 `httpx.stream()` 上下文释放。

同时，adapter 会在流式请求活跃期间记录 OpenCode / Claude Code 的 `x-claude-code-session-id`、`x-claude-code-agent-id`、`x-claude-code-parent-agent-id` 和 parent session header。父 stream 被客户端取消时，会主动取消已注册的子 stream / 孙 stream，让主 agent 双 Esc 这类中断可以传递到仍在运行的 subagent 请求。

这个机制只影响已经在 adapter 内活跃的 stream 请求；不会因为某个 parent 已经退出，就拒绝客户端之后新发来的 HTTP 请求。


## OpenCode subagent cancellation diagnostics

The adapter guarantees current-request stream cleanup:

```text
current client HTTP stream cancelled
→ current provider stream is closed
→ current upstream response is released
```

For active OpenCode / Claude Code subagent streams, parent cancellation is also propagated through the registered session/agent relationship:

```text
parent client HTTP stream cancelled
→ active child stream task is cancelled
→ child provider stream is closed
→ child upstream response is released
```

Use the close-reason logs to distinguish real cancellation from normal completion:

```text
openai stream closed reason=completed done_sent=True
openai stream closed reason=client_cancelled done_sent=False cancelled_children=1
upstream stream closed reason=completed status=200 ...
upstream stream closed reason=client_cancelled status=200 ...
upstream stream closed reason=consumer_closed status=200 ...
```

If OpenCode keeps sending new requests with `x-parent-session-id` after the parent was interrupted, that is a new HTTP request from the client. The adapter logs and forwards it; cancellation is applied to active streams, not to future requests.
