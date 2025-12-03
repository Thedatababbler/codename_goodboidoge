"""LLM client abstractions used by ACE components."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Union

from dotenv import load_dotenv


@dataclass
class LLMResponse:
    """Container for LLM outputs."""

    text: str
    raw: Optional[Dict[str, Any]] = None


class LLMClient(ABC):
    """Abstract interface so ACE can plug into any chat/completions API."""

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Return the model text for a given prompt."""


class DummyLLMClient(LLMClient):
    """Deterministic LLM stub for testing and dry runs."""

    def __init__(self, responses: Optional[Deque[str]] = None) -> None:
        super().__init__(model="dummy")
        self._responses: Deque[str] = responses or deque()

    def queue(self, text: str) -> None:
        """Enqueue a response to be used on the next completion call."""
        self._responses.append(text)

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        if not self._responses:
            raise RuntimeError("DummyLLMClient ran out of queued responses.")
        return LLMResponse(text=self._responses.popleft())


class TransformersLLMClient(LLMClient):
    """LLM client powered by `transformers` pipelines for chat-style models."""

    def __init__(
            self,
            model_path: str,
            *,
            max_new_tokens: int = 512,
            temperature: float = 0.0,
            top_p: float = 0.9,
            device_map: Union[str, Dict[str, int]] = "auto",
            torch_dtype: Union[str, "torch.dtype"] = "auto",
            trust_remote_code: bool = True,
            system_prompt: Optional[str] = None,
            generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model=model_path)

        # Import transformers lazily to avoid mandatory dependency for all users.
        from transformers import AutoTokenizer, pipeline  # type: ignore[import-untyped]

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self._pipeline = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=self._tokenizer,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self._system_prompt = system_prompt or (
            "You are a JSON-only assistant that MUST reply with a single valid JSON object without extra text.\n"
            "Reasoning: low\n"
            "Do not expose analysis or chain-of-thought. Respond using the final JSON only."
        )
        self._defaults: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0.0,
            "return_full_text": False,
        }
        if generation_kwargs:
            self._defaults.update(generation_kwargs)

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        call_kwargs = dict(self._defaults)
        kwargs = dict(kwargs)
        kwargs.pop("refinement_round", None)
        call_kwargs.update(kwargs)

        # Build chat-formatted messages to leverage harmony template.
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        outputs = self._pipeline(messages, **call_kwargs)
        text = self._postprocess_text(self._extract_text(outputs))
        return LLMResponse(text=text, raw={"outputs": outputs})

    def _extract_text(self, outputs: Any) -> str:
        """Normalize pipeline outputs into a single string response."""
        if not outputs:
            return ""
        candidate = outputs[0]

        # Newer transformers versions return {"generated_text": [{"role": ..., "content": ...}, ...]}
        if isinstance(candidate, dict) and "generated_text" in candidate:
            generated = candidate["generated_text"]
            if isinstance(generated, list):
                # Grab the assistant role content if present.
                for message in generated:
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        content = message.get("content")
                        if isinstance(content, str):
                            return content.strip()
                # Fallback to last item's content/text.
                last = generated[-1]
                if isinstance(last, dict):
                    return str(last.get("content") or last.get("text") or "")
                return str(last)
            if isinstance(generated, dict):
                return str(generated.get("content") or generated.get("text") or "")
            return str(generated)

        # Older versions might return {"generated_text": "..."}
        if isinstance(candidate, dict) and isinstance(candidate.get("generated_text"), str):
            return candidate["generated_text"].strip()

        # Ultimate fallback: string representation.
        return str(candidate).strip()

    def _postprocess_text(self, text: str) -> str:
        """Trim analyzer prefixes and isolate JSON payloads when present."""
        trimmed = text.strip()
        if not trimmed:
            return trimmed

        marker = "assistantfinal"
        if marker in trimmed:
            trimmed = trimmed.split(marker, 1)[1].strip()

        if trimmed.startswith(marker):
            trimmed = trimmed[len(marker):].strip()

        # Attempt to extract the first JSON object substring.
        if trimmed and trimmed[0] != "{":
            start = trimmed.find("{")
            end = trimmed.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = trimmed[start: end + 1].strip()
                candidate_clean = candidate.replace("\r", " ").replace("\n", " ")
                try:
                    json.loads(candidate_clean)
                    return candidate_clean
                except json.JSONDecodeError:
                    pass

        return trimmed.replace("\r", " ").replace("\n", " ")

# deepseek兼容openai接口client
class DeepseekLLMClient(LLMClient):
    def __init__(self,
                 model: str = "deepseek-chat",
                 api_key: Optional[str] = None,
                 base_url="https://api.deepseek.com",
                 system_prompt: str = None,
                 ) -> None:
        super().__init__(model=model)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "请先安装OpenAI库:\n"
                "  pip install openai\n"
                "或:\n"
                "  pip install openai>=1.0.0"
            ) from e
        load_dotenv()
        self._system_prompt = system_prompt or (
            "You are a JSON-only assistant that MUST reply with a single valid JSON object without extra text.\n"
            "Reasoning: low\n"
            "Do not expose analysis or chain-of-thought. Respond using the final JSON only."
        )
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )


# ace/llm_openai.py
# from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv

try:
    # OpenAI Python SDK >= 1.0
    from openai import OpenAI
except Exception as _e:  # pragma: no cover
    OpenAI = None  # type: ignore

# from . import LLMClient, LLMResponse


class OpenAIClient(LLMClient):
    """
    OpenAI-backed LLM client that accepts a *string prompt* and returns model text.
    - Prefers the Responses API (client.responses.create)
    - Falls back to Chat Completions API (client.chat.completions.create) if needed
    """

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
        extra_request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            model: e.g., "gpt-4.1" / "gpt-4o" / "o3-mini" 等
            api_key: if None, read from env OPENAI_API_KEY
            base_url: if None, use default; can override with OPENAI_BASE_URL
            timeout: request timeout (seconds)
            max_retries: number of retries with exponential backoff
            temperature: decoding temperature
            max_output_tokens: hard cap on tokens (None = model default)
            extra_request_kwargs: passed through to OpenAI SDK
        """
        load_dotenv(override=False)
        super().__init__(model=model or os.getenv("OPENAI_MODEL", "gpt-4.1"))
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")

        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.extra_request_kwargs = extra_request_kwargs or {}

        if OpenAI is None:
            raise RuntimeError(
                "openai Python SDK is not available. Install with: pip install openai>=1.0.0"
            )

        # Build OpenAI client
        # Note: base_url is optional; only pass if provided
        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = OpenAI(**client_kwargs)

    # -------------- Public API --------------
    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """
        Execute a single-turn text completion using OpenAI.
        Accepts a raw string prompt (already assembled by ACE's Generator).
        """
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # Prefer Responses API (new)
                text, raw = self._complete_via_responses_api(prompt, **kwargs)
                return LLMResponse(text=text, raw=raw)
            except Exception as e_primary:
                last_err = e_primary
                # Try fallback with Chat Completions API
                try:
                    text, raw = self._complete_via_chat_api(prompt, **kwargs)
                    return LLMResponse(text=text, raw=raw)
                except Exception as e_fallback:
                    last_err = e_fallback

            # exponential backoff before retry
            if attempt < self.max_retries:
                sleep_s = min(2 ** (attempt - 1), 8)
                time.sleep(sleep_s)

        # Exhausted retries
        assert last_err is not None
        raise last_err

    # -------------- Private helpers --------------
    def _complete_via_responses_api(self, prompt: str, **kwargs: Any) -> tuple[str, Dict[str, Any]]:
        """
        Use the Responses API:
            client.responses.create(
                model=..., input=prompt, temperature=...
            )
        """
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if self.max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = self.max_output_tokens

        # user overrides
        request_kwargs.update(self.extra_request_kwargs)
        request_kwargs.update(kwargs)

        # Timeout control (SDK exposes 'timeout' in ._client.request)
        # For compatibility, we rely on default; advanced: use httpx client injection.

        resp = self._client.responses.create(**request_kwargs)
        # Robust text extraction
        text = getattr(resp, "output_text", None)
        if not text:
            # Fall back to parsing content parts
            text_parts = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) == "output_text" and getattr(c, "text", None):
                        text_parts.append(c.text)
            text = "\n".join(text_parts).strip()

        if not text:
            raise RuntimeError("OpenAI Responses API returned empty text.")

        return text, self._to_dict(resp)

    def _complete_via_chat_api(self, prompt: str, **kwargs: Any) -> tuple[str, Dict[str, Any]]:
        """
        Fallback to Chat Completions:
            client.chat.completions.create(
                model=..., messages=[{"role":"user","content": prompt}], ...
            )
        """
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if self.max_output_tokens is not None:
            # different name in chat API
            request_kwargs["max_tokens"] = self.max_output_tokens

        request_kwargs.update(self.extra_request_kwargs)
        request_kwargs.update(kwargs)

        resp = self._client.chat.completions.create(**request_kwargs)
        choice0 = (getattr(resp, "choices", None) or [None])[0]
        if not choice0 or not getattr(choice0, "message", None):
            raise RuntimeError("OpenAI Chat Completions API returned no choices.")
        text = (choice0.message.content or "").strip()
        if not text:
            raise RuntimeError("OpenAI Chat Completions API returned empty message content.")
        return text, self._to_dict(resp)

    @staticmethod
    def _to_dict(obj: Any) -> Dict[str, Any]:
        """Best-effort convert OpenAI SDK object to plain dict for logging/debug."""
        try:
            # openai sdk objects often support .model_dump() / .to_dict()
            if hasattr(obj, "model_dump"):
                return obj.model_dump()  # pydantic v2
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
        except Exception:
            pass
        # Fallback: repr
        return {"repr": repr(obj)}
