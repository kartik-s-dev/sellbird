import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from harvey.ollama_client import OllamaClient, OllamaError


@pytest.mark.asyncio
async def test_list_models_returns_local_model_names():
    response = MagicMock()
    response.json.return_value = {
        "models": [
            {"name": "phi3:latest"},
            {"name": "nomic-embed-text:latest"},
        ]
    }
    response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=response)

    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=mock_client)
    context.__aexit__ = AsyncMock(return_value=None)

    with patch("harvey.ollama_client.httpx.AsyncClient", return_value=context):
        models = await OllamaClient().list_models()

    assert models == {"phi3:latest", "nomic-embed-text:latest"}
    mock_client.get.assert_awaited_once_with(
        "http://127.0.0.1:11434/api/tags"
    )


@pytest.mark.asyncio
async def test_chat_sends_local_non_streaming_request():
    response = MagicMock()
    response.json.return_value = {
        "model": "phi3:latest",
        "message": {"content": "Hello from local Ollama"},
        "prompt_eval_count": 12,
        "eval_count": 8,
        "total_duration": 2_500_000_000,
    }
    response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=response)

    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=mock_client)
    context.__aexit__ = AsyncMock(return_value=None)

    with patch("harvey.ollama_client.httpx.AsyncClient", return_value=context):
        result = await OllamaClient().chat("phi3:latest", "Say hello")

    assert result == {
        "text": "Hello from local Ollama",
        "model": "phi3:latest",
        "input_tokens": 12,
        "output_tokens": 8,
        "duration_ms": 2500,
    }

    mock_client.post.assert_awaited_once_with(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": "phi3:latest",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": False,
        },
    )


@pytest.mark.asyncio
async def test_list_models_reports_server_connection_failure():
    mock_client = MagicMock()
    mock_client.get = AsyncMock(
        side_effect=httpx.ConnectError("connection refused")
    )

    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=mock_client)
    context.__aexit__ = AsyncMock(return_value=None)

    with patch("harvey.ollama_client.httpx.AsyncClient", return_value=context):
        with pytest.raises(OllamaError, match="Cannot reach local Ollama"):
            await OllamaClient().list_models()


@pytest.mark.asyncio
async def test_chat_rejects_missing_model_and_empty_prompt():
    client = OllamaClient()

    with pytest.raises(OllamaError, match="No Ollama model"):
        await client.chat("", "Hello")

    with pytest.raises(OllamaError, match="prompt is empty"):
        await client.chat("phi3:latest", "   ")


def test_client_rejects_remote_endpoint():
    with pytest.raises(ValueError, match="Only the local Ollama endpoint"):
        OllamaClient("https://example.com")


def _mock_ollama_context(method_name, response):
    mock_client = MagicMock()
    setattr(mock_client, method_name, AsyncMock(return_value=response))

    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=mock_client)
    context.__aexit__ = AsyncMock(return_value=None)

    return context


@pytest.mark.asyncio
async def test_list_models_rejects_non_dictionary_response():
    response = MagicMock()
    response.json.return_value = ["not", "a", "dictionary"]
    response.raise_for_status = MagicMock()

    context = _mock_ollama_context("get", response)

    with patch("harvey.ollama_client.httpx.AsyncClient", return_value=context):
        with pytest.raises(OllamaError, match="invalid model-list response"):
            await OllamaClient().list_models()


@pytest.mark.asyncio
async def test_list_models_rejects_malformed_model_entry():
    response = MagicMock()
    response.json.return_value = {"models": ["invalid-entry"]}
    response.raise_for_status = MagicMock()

    context = _mock_ollama_context("get", response)

    with patch("harvey.ollama_client.httpx.AsyncClient", return_value=context):
        with pytest.raises(OllamaError, match="invalid model entry"):
            await OllamaClient().list_models()


@pytest.mark.asyncio
async def test_chat_rejects_non_dictionary_response():
    response = MagicMock()
    response.json.return_value = ["invalid", "response"]
    response.raise_for_status = MagicMock()

    context = _mock_ollama_context("post", response)

    with patch("harvey.ollama_client.httpx.AsyncClient", return_value=context):
        with pytest.raises(OllamaError, match="invalid chat response"):
            await OllamaClient().chat("phi3:latest", "Hello")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("prompt_eval_count", -1, "input token count"),
        ("eval_count", "eight", "output token count"),
        ("eval_count", True, "output token count"),
        ("total_duration", -100, "total duration"),
    ],
)
async def test_chat_rejects_invalid_numeric_metadata(field, value, error):
    payload = {
        "model": "phi3:latest",
        "message": {"content": "Hello"},
        "prompt_eval_count": 12,
        "eval_count": 8,
        "total_duration": 2_500_000_000,
    }
    payload[field] = value

    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status = MagicMock()

    context = _mock_ollama_context("post", response)

    with patch("harvey.ollama_client.httpx.AsyncClient", return_value=context):
        with pytest.raises(OllamaError, match=error):
            await OllamaClient().chat("phi3:latest", "Hello")


@pytest.mark.asyncio
async def test_chat_reports_http_server_error():
    request = httpx.Request("POST", "http://127.0.0.1:11434/api/chat")
    http_response = httpx.Response(500, request=request)
    error = httpx.HTTPStatusError(
        "Internal Server Error",
        request=request,
        response=http_response,
    )

    response = MagicMock()
    response.raise_for_status.side_effect = error

    context = _mock_ollama_context("post", response)

    with patch("harvey.ollama_client.httpx.AsyncClient", return_value=context):
        with pytest.raises(OllamaError, match="Local Ollama request failed"):
            await OllamaClient().chat("phi3:latest", "Hello")
