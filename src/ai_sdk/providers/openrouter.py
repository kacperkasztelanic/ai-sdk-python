from __future__ import annotations

import os
from typing import Any, Optional

import openai as _openai

from .openai import OpenAIEmbeddingModel, OpenAIModel


class OpenRouterModel(OpenAIModel):
    """Implementation of LanguageModel for OpenRouter (via OpenAI client)."""

    def __init__(
        self, model: str, *, api_key: Optional[str] = None, **default_kwargs: Any
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._client = _openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
        self._model = model
        self._default_kwargs = default_kwargs


class OpenRouterEmbeddingModel(OpenAIEmbeddingModel):
    """Implementation of EmbeddingModel for OpenRouter."""

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        max_batch_size: Optional[int] = None,
        **default_kwargs: Any,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._client = _openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
        self._model = model
        self._default_kwargs = default_kwargs
        if max_batch_size is not None:
            self.max_batch_size = max_batch_size  # type: ignore[assignment]


def openrouter(
    model: str, *, api_key: Optional[str] = None, **default_kwargs: Any
) -> OpenRouterModel:
    """Return a configured OpenRouterModel instance."""
    return OpenRouterModel(model, api_key=api_key, **default_kwargs)


def embedding(
    model: str,
    *,
    api_key: Optional[str] = None,
    **default_kwargs: Any,
) -> OpenRouterEmbeddingModel:
    """Factory helper that returns an OpenRouterEmbeddingModel instance."""
    return OpenRouterEmbeddingModel(model, api_key=api_key, **default_kwargs)


setattr(openrouter, "embedding", embedding)
