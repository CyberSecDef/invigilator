"""Model adapters.

Anthropic goes through the official `anthropic` SDK. The OpenAI-compatible
providers (OpenAI, DeepSeek, xAI) share one adapter, and Gemini and Ollama get
thin ones of their own -- all over plain HTTP, since this file is deliberately
provider-neutral and pulling six vendor SDKs in would not make it clearer.
TypeSafe is the odd one out: it answers typed questions rather than prompts.
`claude-cli` drives the Claude Code CLI instead of an API, for comparing what a
Claude Code subscription answers with what the API does.

Every adapter returns the same `Reply`, and none of them retries on a refusal:
a refusal is a measurement, not an error.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

TIMEOUT = 120


@dataclass
class Reply:
    text: str
    stop_reason: str = ""
    hard_refusal: bool = False   # provider-level refusal, not just an evasive answer
    usage: dict = field(default_factory=dict)


class ProviderError(RuntimeError):
    """A transport-level failure. Explicitly NOT a refusal.

    Conflating the two is how a rate-limited run turns into a fabricated
    finding that "the model declined 55 questions".
    """

    def __init__(self, message, status=None, retry_after=None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after

    @property
    def retryable(self):
        return self.status in (408, 409, 425, 429, 500, 502, 503, 504) or self.status is None


def load_dotenv(path: str | Path) -> int:
    """Read KEY=VALUE lines into os.environ without clobbering what is set."""
    path = Path(path).expanduser()
    if not path.exists():
        return 0
    loaded = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and not os.environ.get(key):
            os.environ[key] = value
            loaded += 1
    return loaded


def _retry_after(headers, body: str):
    """Seconds the server asked us to wait, from a header or a Google-style body."""
    value = headers.get("Retry-After") if headers else None
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    match = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', body)
    return float(match.group(1)) if match else None


def _http_json(url: str, payload: dict, headers: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise ProviderError(
            f"HTTP {exc.code}: {body[:300]}",
            status=exc.code,
            retry_after=_retry_after(exc.headers, body),
        ) from None
    except urllib.error.URLError as exc:
        raise ProviderError(f"connection failed: {exc.reason}") from None


class Provider:
    key_env: str = ""
    max_concurrency: int = 8
    # Minimum seconds between requests. Retry-after backoff reacts to a rate
    # limit after tripping it; on an RPM-capped free tier that degenerates into
    # every request failing once. Pacing up front avoids the limit entirely.
    min_interval: float = 0.0
    # True for judgement models that take typed questions instead of a prompt.
    structured: bool = False

    def __init__(self):
        self._gate = threading.Lock()
        self._next_at = 0.0

    def available(self) -> bool:
        return not self.key_env or bool(os.environ.get(self.key_env))

    def _pace(self) -> None:
        if not self.min_interval:
            return
        with self._gate:
            now = time.monotonic()
            wait = self._next_at - now
            if wait > 0:
                time.sleep(wait)
            self._next_at = max(now, self._next_at) + self.min_interval

    def complete(self, model: str, system: str, user: str, max_tokens: int,
                 deliberate: bool = False) -> Reply:
        """`deliberate` asks for the model's full reasoning instead of a quick
        disposition. Questionnaires leave it off; exams turn it on. Providers
        without a reasoning control ignore it."""
        self._pace()
        return self._complete(model, system, user, max_tokens, deliberate=deliberate)

    def _complete(self, model: str, system: str, user: str, max_tokens: int,
                  deliberate: bool = False) -> Reply:
        raise NotImplementedError

    def list_models(self) -> list[str]:
        return []


class Anthropic(Provider):
    name = "anthropic"
    key_env = "ANTHROPIC_API_KEY"
    default_model = "claude-opus-5"

    def __init__(self):
        super().__init__()
        self._client = None
        # Older / smaller models reject output_config.effort outright; remember
        # which ones so we only pay for the 400 once.
        self._no_effort = set()

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ProviderError("pip install anthropic") from None
            self._client = anthropic.Anthropic()
        return self._client

    def _complete(self, model, system, user, max_tokens, deliberate=False):
        client = self._get_client()
        # Thinking is left at the Opus 5 default (adaptive) and effort is held
        # low: these are one-line judgements, and we want the model's ordinary
        # disposition rather than a reasoned-out position. Server-side refusal
        # fallbacks are deliberately NOT enabled -- silently rerouting a refused
        # item to a different model would corrupt the very thing being measured.
        try:
            response = self._call(client, model, system, user, max_tokens, deliberate)
        except Exception as exc:
            if "effort parameter" in str(exc):
                # Retry unconditionally, not just for the first caller to notice:
                # under concurrency every in-flight request trips this together,
                # and the retry cannot recur because _complete now omits effort.
                self._no_effort.add(model)
                return self._complete(model, system, user, max_tokens, deliberate)
            status = getattr(exc, "status_code", None)
            if status is None and not isinstance(exc, ProviderError):
                raise ProviderError(f"{type(exc).__name__}: {exc}") from None
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(f"HTTP {status}: {exc}", status=status) from None
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        return Reply(
            text=text.strip(),
            stop_reason=response.stop_reason or "",
            hard_refusal=response.stop_reason == "refusal",
            usage={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
        )

    def _call(self, client, model, system, user, max_tokens, deliberate=False):
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if model not in self._no_effort and not deliberate:
            # One-line judgements: keep thinking at its default and effort low.
            # Deliberate requests (exams) leave effort at the model's default.
            kwargs["output_config"] = {"effort": "low"}
        return client.messages.create(**kwargs)

    def list_models(self):
        return [m.id for m in self._get_client().models.list()]


class OpenAICompatible(Provider):
    """OpenAI, DeepSeek and xAI all speak /v1/chat/completions."""

    def __init__(self, name: str, base_url: str, key_env: str, default_model: str):
        super().__init__()
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.key_env = key_env
        self.default_model = default_model

    def _headers(self):
        return {"Authorization": f"Bearer {os.environ.get(self.key_env, '')}"}

    def _complete(self, model, system, user, max_tokens, deliberate=False):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_tokens,
        }
        try:
            data = _http_json(f"{self.base_url}/chat/completions", payload, self._headers())
        except ProviderError as exc:
            # Older / non-OpenAI endpoints still want the legacy field name.
            if "max_completion_tokens" not in str(exc):
                raise
            payload.pop("max_completion_tokens")
            payload["max_tokens"] = max_tokens
            data = _http_json(f"{self.base_url}/chat/completions", payload, self._headers())

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return Reply(
            text=(message.get("content") or "").strip(),
            stop_reason=choice.get("finish_reason") or "",
            hard_refusal=bool(message.get("refusal")),
            usage=data.get("usage") or {},
        )

    def list_models(self):
        request = urllib.request.Request(f"{self.base_url}/models", headers=self._headers())
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read())
        return sorted(m["id"] for m in data.get("data", []))


class Gemini(Provider):
    name = "gemini"
    key_env = "GEMINI_API_KEY"
    max_concurrency = 1
    min_interval = 6.5
    default_model = "gemini-3.7-flash"
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _headers(self):
        return {"x-goog-api-key": os.environ.get(self.key_env, "")}

    def _complete(self, model, system, user, max_tokens, deliberate=False):
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        data = _http_json(
            f"{self.base_url}/models/{model}:generateContent", payload, self._headers()
        )
        candidates = data.get("candidates") or []
        if not candidates:
            # A prompt blocked by safety filters returns no candidate at all.
            block = (data.get("promptFeedback") or {}).get("blockReason", "")
            return Reply(text="", stop_reason=f"blocked:{block}", hard_refusal=True)
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        finish = candidate.get("finishReason", "")
        return Reply(
            text="".join(p.get("text", "") for p in parts).strip(),
            stop_reason=finish,
            hard_refusal=finish in {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT"},
            usage=data.get("usageMetadata") or {},
        )

    def list_models(self):
        request = urllib.request.Request(f"{self.base_url}/models", headers=self._headers())
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read())
        return sorted(
            m["name"].removeprefix("models/")
            for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        )


class Ollama(Provider):
    name = "ollama"
    key_env = ""
    max_concurrency = 1
    default_model = "gemma4:26b"
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    def _complete(self, model, system, user, max_tokens, deliberate=False):
        payload = {
            "model": model,
            "stream": False,
            # Local reasoning models (gemma4 among them) will otherwise spend the
            # entire token budget thinking and return empty content, which reads
            # as a refusal when it is nothing of the sort. Deliberate requests
            # (exams) get the room to think that this switch otherwise denies.
            "think": deliberate,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"num_predict": max_tokens},
        }
        try:
            data = _http_json(f"{self.base_url}/api/chat", payload, {})
        except ProviderError as exc:
            if "think" not in str(exc):
                raise
            payload.pop("think")
            data = _http_json(f"{self.base_url}/api/chat", payload, {})
        message = data.get("message") or {}
        text = (message.get("content") or "").strip()
        return Reply(
            text=text,
            stop_reason=data.get("done_reason") or "",
            usage={"input": data.get("prompt_eval_count"), "output": data.get("eval_count")},
        )

    def list_models(self):
        with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=30) as response:
            data = json.loads(response.read())
        return sorted(m["name"] for m in data.get("models", []))


class TypeSafe(Provider):
    """TypeSafe's System One models (Jev).

    These return typed judgements with probabilities, not text, so there is
    nothing for `complete()` to do: survey.py sees `structured` and asks each
    item as a Score question through `judge()` instead. Jev cannot refuse --
    every question gets a distribution -- so a TypeSafe run's refusal count is
    zero by construction, not by disposition.
    """

    name = "typesafe"
    key_env = "TYPESAFE_API_KEY"
    default_model = "jev-latest"
    structured = True
    base_url = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1").rstrip("/")

    def _headers(self):
        return {"Authorization": f"Bearer {os.environ.get(self.key_env, '')}"}

    def _complete(self, model, system, user, max_tokens, deliberate=False):
        raise ProviderError("typesafe models answer typed questions, not prompts", status=400)

    def judge(self, model: str, state: dict, questions: dict) -> dict:
        """One /systemone request. Returns the raw response body."""
        self._pace()
        payload = {"model": model, "state": state, "questions": questions}
        return _http_json(f"{self.base_url}/systemone", payload, self._headers())


class ClaudeCLI(Provider):
    """Claude through the Claude Code CLI (`claude -p`), on the user's login.

    Each item is one non-interactive session, isolated so the model sees the
    instrument's prompt and little else: our system prompt replaces Claude
    Code's, every tool is off (no calculator, no web search, no code), no MCP
    servers, no setting sources (so no hooks or permissions from the user's
    settings), no slash commands or skills, no saved session, and an empty
    working directory so no CLAUDE.md is found. Claude Code still adds a
    one-line SDK identity, the account's email and an environment block
    (directory, OS, model, date) -- none of it bears on an answer.

    `--bare` would isolate further but authenticates only by API key, which
    defeats the purpose. For the same reason ANTHROPIC_API_KEY, which `.env`
    may have loaded, is removed from the subprocess's environment: the CLI
    must use the login, not quietly bill the API.

    The CLI takes no token limit, so `max_tokens` is not passed; a deliberate
    request keeps the model's default effort, a quick one asks for low effort,
    mirroring the API adapter.
    """

    name = "claude-cli"
    key_env = ""
    max_concurrency = 6
    default_model = "default"      # whatever the CLI is configured to use
    timeout = 900

    def __init__(self):
        super().__init__()
        self._cwd = None

    def available(self) -> bool:
        return shutil.which("claude") is not None

    def _complete(self, model, system, user, max_tokens, deliberate=False):
        if self._cwd is None:
            self._cwd = tempfile.mkdtemp(prefix="invigilator-cli-")
        command = [
            shutil.which("claude") or "claude", "-p", "--output-format", "json",
            "--system-prompt", system,
            "--tools", "", "--strict-mcp-config", "--setting-sources", "",
            "--disable-slash-commands", "--no-session-persistence",
        ]
        if model and model != "default":
            command += ["--model", model]
        if not deliberate:
            command += ["--effort", "low"]
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        try:
            done = subprocess.run(command, input=user, capture_output=True, text=True,
                                  cwd=self._cwd, env=env, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise ProviderError(f"claude -p timed out after {self.timeout}s", status=408) from None
        try:
            data = json.loads(done.stdout)
        except json.JSONDecodeError:
            raise ProviderError(
                f"claude -p exit {done.returncode}: {(done.stderr or done.stdout)[:300]}") from None
        if data.get("is_error"):
            message = str(data.get("result") or data.get("subtype") or "error")
            limited = re.search(r"usage limit|rate limit|overloaded|429|529", message, re.I)
            raise ProviderError(f"claude -p: {message[:300]}", status=429 if limited else None)
        usage = data.get("usage") or {}
        return Reply(
            text=(data.get("result") or "").strip(),
            stop_reason=data.get("stop_reason") or "",
            hard_refusal=data.get("stop_reason") == "refusal",
            usage={"input": usage.get("input_tokens"), "output": usage.get("output_tokens"),
                   "models": list((data.get("modelUsage") or {})),
                   "cost_usd": data.get("total_cost_usd")},
        )

    def list_models(self):
        return ["default", "opus", "sonnet", "haiku"]


def registry() -> dict[str, Provider]:
    return {
        "anthropic": Anthropic(),
        "openai": OpenAICompatible(
            "openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-5.5"
        ),
        "deepseek": OpenAICompatible(
            "deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-v4-pro"
        ),
        "xai": OpenAICompatible(
            "xai", "https://api.x.ai/v1", "GROK_API_KEY", "grok-4.6"
        ),
        "gemini": Gemini(),
        "ollama": Ollama(),
        "typesafe": TypeSafe(),
        "claude-cli": ClaudeCLI(),
    }


def resolve(spec: str) -> tuple[Provider, str]:
    """Turn "provider:model" (or bare "provider") into a provider and model id."""
    providers = registry()
    name, _, model = spec.partition(":")
    if name not in providers:
        raise ProviderError(f"unknown provider {name!r}; have {', '.join(providers)}")
    provider = providers[name]
    return provider, model or provider.default_model
