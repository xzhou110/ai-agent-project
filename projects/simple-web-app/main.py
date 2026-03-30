import asyncio
import json
import logging
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False

app = FastAPI()

_STATIC_DIR = Path(__file__).resolve().parent / "static"

_base_url = "https://space.ai-builders.com/backend/v1"
_search_url = f"{_base_url.rstrip('/')}/search/"
_api_key = os.getenv("SUPER_MIND_API_KEY")
_client = (
    AsyncOpenAI(api_key=_api_key, base_url=_base_url) if _api_key else None
)

# OpenAI-compatible "tools" entry the LLM receives (JSON-serializable).
WEB_SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the internal web index for up-to-date or factual information. "
            "Use for recent events, sports, news, or anything that may have changed "
            "after training data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords or natural-language question.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

READ_PAGE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_page",
        "description": (
            "Fetch a public web page by URL and return its main visible text. "
            "HTML tags, scripts, and styles are stripped. Use after web_search "
            "when you need details from a specific link (e.g. release notes, changelogs)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute http(s) URL to fetch.",
                }
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },
}

# Tools passed to the chat completions API for the agent loop.
AGENT_TOOLS: list[dict[str, Any]] = [
    WEB_SEARCH_TOOL_SCHEMA,
    READ_PAGE_TOOL_SCHEMA,
]

_MAX_READ_PAGE_TEXT_CHARS = 80_000


def read_page(url: str) -> dict[str, Any]:
    """
    GET the URL, parse HTML, and return stripped main text (no scripts/styles).
    """
    u = (url or "").strip()
    if not u.lower().startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; SimpleWebApp/1.0; +https://example.local) "
            "AppleWebKit/537.36 (KHTML, like Gecko)"
        ),
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    }
    with httpx.Client(timeout=60.0, follow_redirects=True, headers=headers) as client:
        response = client.get(u)
        response.raise_for_status()
        content_type = (response.headers.get("content-type") or "").lower()
        if "html" not in content_type and "xml" not in content_type:
            # Still try parsing; many servers omit charset or mislabel.
            pass
        raw = response.text

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = "\n".join(lines)

    truncated = len(text) > _MAX_READ_PAGE_TEXT_CHARS
    if truncated:
        text = text[:_MAX_READ_PAGE_TEXT_CHARS]

    return {
        "url": u,
        "text": text,
        "truncated": truncated,
        "length": len(text),
    }


def web_search(query: str) -> Any:
    """
    POST to the internal search API with the same auth as the chat client.
    """
    api_key = os.getenv("SUPER_MIND_API_KEY")
    if not api_key:
        raise ValueError("SUPER_MIND_API_KEY is not set")
    payload = {"keywords": [query], "max_results": 3}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=60.0) as client:
        response = client.post(_search_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


def _assistant_message_to_dict(message: Any) -> dict[str, Any]:
    """Build a Chat Completions message dict for the next request (incl. tool_calls)."""
    d: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return d


def _safe_stdout_print(text: str, *, end: str = "\n") -> None:
    """Avoid UnicodeEncodeError on Windows consoles when dumping large Unicode tool payloads."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    payload = text + end
    data = payload.encode(enc, errors="replace")
    try:
        sys.stdout.buffer.write(data)
    except AttributeError:
        sys.stdout.write(payload.encode("ascii", errors="replace").decode("ascii"))
    sys.stdout.flush()


def _print_full_agent_history(
    messages: list[dict[str, Any]],
    final_assistant_content: str | None,
) -> None:
    """
    Dump the full trail sent to/returned from the model before returning to the client:
    user message, each assistant tool_calls block, each tool result, then the final reply.
    """
    sep = "=" * 72
    _safe_stdout_print(f"\n{sep}")
    _safe_stdout_print("[CHAT] Full message history leading to final response")
    _safe_stdout_print(sep)

    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "user":
            _safe_stdout_print(f"\n--- [{i}] INITIAL USER MESSAGE ---")
            _safe_stdout_print(m.get("content") or "")
        elif role == "assistant":
            _safe_stdout_print(f"\n--- [{i}] ASSISTANT (tool_calls decision) ---")
            c = m.get("content")
            if c:
                _safe_stdout_print("assistant_content (same turn, if any):")
                _safe_stdout_print(c)
            tcs = m.get("tool_calls")
            if tcs:
                _safe_stdout_print("tool_calls (what the model decided to run):")
                _safe_stdout_print(json.dumps(tcs, indent=2, ensure_ascii=False))
        elif role == "tool":
            tid = m.get("tool_call_id", "")
            _safe_stdout_print(
                f"\n--- [{i}] TOOL RESULT (raw payload to model) "
                f"tool_call_id={tid!r} ---"
            )
            _safe_stdout_print(m.get("content") or "")

    _safe_stdout_print("\n--- FINAL ASSISTANT MESSAGE (returned to user) ---")
    _safe_stdout_print(final_assistant_content or "")
    _safe_stdout_print(f"{sep}\n")


def _tool_log_label(tool_name: str) -> str:
    """Human-friendly tool name for logs (matches user-facing 'search' wording)."""
    if tool_name == "web_search":
        return "search"
    return tool_name


def _format_tool_output_for_log(content: str, max_len: int = 2000) -> str:
    if len(content) <= max_len:
        return content
    return content[:max_len] + "... [truncated]"


async def _execute_tool_call(name: str, arguments_json: str) -> str:
    """Run tool by name; return string content for a tool role message."""
    if name == "web_search":
        try:
            args = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as e:
            return json.dumps({"error": "invalid_tool_arguments", "detail": str(e)})
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return json.dumps({"error": "missing_or_empty_query"})
        try:
            result = await asyncio.to_thread(web_search, query.strip())
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": "web_search_failed", "detail": str(e)})

    if name == "read_page":
        try:
            args = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as e:
            return json.dumps({"error": "invalid_tool_arguments", "detail": str(e)})
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            return json.dumps({"error": "missing_or_empty_url"})
        try:
            result = await asyncio.to_thread(read_page, url.strip())
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": "read_page_failed", "detail": str(e)})

    return json.dumps({"error": "unknown_tool", "name": name})


class ChatRequest(BaseModel):
    user_message: str


@app.get("/chat-ui")
async def chat_ui():
    """Customer-facing chat page (register before any /{param} catch-alls)."""
    path = _STATIC_DIR / "chat.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="chat.html missing under static/")
    return FileResponse(path, media_type="text/html")


EmitFn = Callable[[dict[str, Any]], Awaitable[None]]


async def run_agent_chat(
    user_message: str,
    *,
    emit: EmitFn | None = None,
) -> str:
    """
    Shared agent loop for /chat and /chat/stream.
    If emit is set, JSON-serializable event dicts are sent for the customer UI.
    """
    if _client is None:
        raise HTTPException(
            status_code=500,
            detail="SUPER_MIND_API_KEY is missing; set it in .env",
        )

    async def _emit(ev: dict[str, Any]) -> None:
        if emit:
            await emit(ev)

    max_turns = 3
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_message},
    ]

    await _emit({"type": "status", "message": "Received your message. Starting the agent…"})

    for turn in range(1, max_turns + 1):
        await _emit(
            {
                "type": "status",
                "message": f"Calling the model (step {turn} of {max_turns}, tools allowed)…",
            }
        )
        try:
            completion = await _client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                tools=AGENT_TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

        message = completion.choices[0].message

        if not message.tool_calls:
            final = (message.content or "").strip()
            logger.info("[Agent] Final Answer: %r", final)
            if not final:
                raise HTTPException(
                    status_code=502,
                    detail="Assistant returned no text content",
                )
            await _emit({"type": "status", "message": "Got final answer from the model."})
            _print_full_agent_history(messages, message.content)
            return message.content

        messages.append(_assistant_message_to_dict(message))
        await _emit(
            {
                "type": "status",
                "message": f"The model requested {len(message.tool_calls)} tool call(s). Running…",
            }
        )

        for tc in message.tool_calls:
            label = _tool_log_label(tc.function.name)
            logger.info("[Agent] Decided to call tool: %r", label)
            args_raw = tc.function.arguments or ""
            ap = args_raw if len(args_raw) <= 600 else args_raw[:600] + "…"
            await _emit(
                {
                    "type": "tool_call",
                    "name": tc.function.name,
                    "label": label,
                    "arguments_preview": ap,
                }
            )

            output = await _execute_tool_call(
                tc.function.name,
                tc.function.arguments or "",
            )
            logger.info(
                "[System] Tool Output: %r",
                _format_tool_output_for_log(output),
            )
            await _emit(
                {
                    "type": "tool_result",
                    "label": label,
                    "preview": _format_tool_output_for_log(output),
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                }
            )

        if turn == max_turns:
            logger.warning(
                "[Agent] Max tool turns (%s) reached; 4th call: final text-only synthesis",
                max_turns,
            )
            await _emit(
                {
                    "type": "synthesis",
                    "message": "Tool budget used. Running final synthesis (no more tools)…",
                }
            )
            try:
                final_completion = await _client.chat.completions.create(
                    model="gpt-5",
                    messages=messages,
                )
            except Exception as e:
                raise HTTPException(status_code=502, detail=str(e)) from e

            final_msg = final_completion.choices[0].message
            final_text = (final_msg.content or "").strip()
            logger.info("[Agent] Final Answer: %r", final_text)
            if not final_text:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Agent used max tool turns but the synthesis call returned "
                        "no text. Try narrowing the question."
                    ),
                )
            _print_full_agent_history(messages, final_msg.content)
            return final_msg.content

    raise HTTPException(status_code=502, detail="Unexpected agent loop exit")


@app.get("/tools/web-search/schema")
async def get_web_search_tool_schema():
    """Expose the tool definition JSON for clients or documentation."""
    return WEB_SEARCH_TOOL_SCHEMA


@app.get("/tools/read-page/schema")
async def get_read_page_tool_schema():
    """Expose the read_page tool definition JSON."""
    return READ_PAGE_TOOL_SCHEMA


@app.post("/verify/web-search-tool-call")
async def verify_web_search_tool_call():
    """
    Ask the model a question that should trigger web_search; return any tool_calls.
    Does not execute tools (no agent loop).
    """
    if _client is None:
        raise HTTPException(
            status_code=500,
            detail="SUPER_MIND_API_KEY is missing; set it in .env",
        )
    try:
        completion = await _client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "user",
                    "content": "Who won the Super Bowl?",
                }
            ],
            tools=[WEB_SEARCH_TOOL_SCHEMA],
            tool_choice="auto",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    message = completion.choices[0].message
    tool_calls_out: list[dict[str, Any]] = []
    if message.tool_calls:
        for tc in message.tool_calls:
            tool_calls_out.append(
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
            )

    parsed_args: list[Any] = []
    for tc in tool_calls_out:
        args_raw = tc["function"]["arguments"]
        try:
            parsed_args.append(json.loads(args_raw) if args_raw else {})
        except json.JSONDecodeError:
            parsed_args.append(None)

    return {
        "question": "Who won the Super Bowl?",
        "has_tool_calls": bool(tool_calls_out),
        "tool_calls": tool_calls_out,
        "parsed_arguments": parsed_args,
        "assistant_content": message.content,
        "valid_web_search_call": any(
            tc["function"]["name"] == "web_search"
            and parsed_args[i] is not None
            and "query" in (parsed_args[i] or {})
            for i, tc in enumerate(tool_calls_out)
        ),
    }


@app.post("/chat")
async def chat(body: ChatRequest):
    content = await run_agent_chat(body.user_message)
    return {"content": content}


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    """NDJSON stream of status / tool / done / error events for the customer UI."""

    async def ndjson_generator() -> Any:
        queue: asyncio.Queue[Any] = asyncio.Queue()

        async def emit(ev: dict[str, Any]) -> None:
            await queue.put(ev)

        async def worker() -> None:
            try:
                content = await run_agent_chat(body.user_message, emit=emit)
                await queue.put({"type": "done", "content": content})
            except HTTPException as he:
                d = he.detail
                await queue.put(
                    {
                        "type": "error",
                        "detail": d if isinstance(d, str) else str(d),
                    }
                )
            except Exception as e:
                await queue.put({"type": "error", "detail": str(e)})
            finally:
                await queue.put(None)

        task = asyncio.create_task(worker())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield json.dumps(item, ensure_ascii=False) + "\n"
        finally:
            await task

    return StreamingResponse(
        ndjson_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/hello/{user_input}")
async def hello(user_input: str):
    """Demo JSON route; not a catch-all (avoids shadowing /chat-ui and similar paths)."""
    return {"message": f"Hello, World {user_input}"}
