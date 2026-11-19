
"""Local-only Ollama client for Harvey."""

import httpx


OLLAMA_BASE_URL = "http://127.0.0.1:11434"


class OllamaError(Exception):
    """Raised when the local Ollama service cannot complete a request."""


def _non_negative_int(value: object, field: str) -> int:
    """Validate a non-negative integer returned by Ollama."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OllamaError(f"Ollama returned an invalid {field}.")
    return value


class OllamaClient:
    """Small async client restricted to the local Ollama service."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        if base_url.rstrip("/") != OLLAMA_BASE_URL:
            raise ValueError("Only the local Ollama endpoint is permitted.")
        self.base_url = OLLAMA_BASE_URL

    async def list_models(self, timeout: float = 5.0) -> set[str]:
        """Return model names currently available in local Ollama."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                payload = response.json()

            if not isinstance(payload, dict):
                raise OllamaError("Ollama returned an invalid model-list response.")

            models = payload.get("models")
            if not isinstance(models, list):
                raise OllamaError("Ollama returned an invalid model list.")

            names = set()
            for item in models:
                if not isinstance(item, dict):
                    raise OllamaError("Ollama returned an invalid model entry.")
                name = item.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise OllamaError("Ollama returned a model without a valid name.")
                names.add(name)

            return names

        except OllamaError:
            raise
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            raise OllamaError(
                f"Cannot reach local Ollama or read its model list: {exc}"
            ) from exc

    async def chat(
        self,
        model: str,
        prompt: str,
        timeout: float = 300.0,
    ) -> dict:
        """Run one non-streaming local chat request."""
        if not isinstance(model, str) or not model.strip():
            raise OllamaError("No Ollama model is configured.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise OllamaError("The prompt is empty.")

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    },
                )
                response.raise_for_status()
                payload = response.json()

            if not isinstance(payload, dict):
                raise OllamaError("Ollama returned an invalid chat response.")

            message = payload.get("message")
            if not isinstance(message, dict):
                raise OllamaError("Ollama response has no valid message.")

            content = message.get("content")
            if not isinstance(content, str):
                raise OllamaError("Ollama response has no text content.")

            response_model = payload.get("model")
            if not isinstance(response_model, str) or not response_model.strip():
                raise OllamaError("Ollama response has no valid model name.")

            input_tokens = _non_negative_int(
                payload.get("prompt_eval_count"), "input token count"
            )
            output_tokens = _non_negative_int(
                payload.get("eval_count"), "output token count"
            )
            duration_ns = _non_negative_int(
                payload.get("total_duration"), "total duration"
            )

            return {
                "text": content,
                "model": response_model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "duration_ms": duration_ns // 1_000_000,
            }

        except OllamaError:
            raise
        except httpx.TimeoutException as exc:
            raise OllamaError(
                f"Local Ollama request timed out "
                f"({type(exc).__name__}): {exc!r}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Local Ollama request failed "
                f"({type(exc).__name__}): {exc!r}"
            ) from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise OllamaError(
                f"Invalid local Ollama response "
                f"({type(exc).__name__}): {exc!r}"
            ) from exc
