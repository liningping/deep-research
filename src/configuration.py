"""
Configuration Module

Provides configurable fields for the research assistant.
Simplified to support OpenAI provider only.
"""

import os
from typing import Optional
from langchain_core.runnables import RunnableConfig
from enum import Enum


class SearchAPI(Enum):
    TAVILY = "tavily"


class LLMProvider(Enum):
    OPENAI = "openai"


class ActivityVerbosity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Configuration:
    """The configurable fields for the research assistant."""

    def __init__(self, **kwargs):
        self._max_web_research_loops = kwargs.get("max_web_research_loops")
        self._search_api = kwargs.get("search_api")
        self._fetch_full_page = kwargs.get("fetch_full_page")
        self._include_raw_content = kwargs.get("include_raw_content")
        self._llm_provider = kwargs.get("llm_provider")
        self._llm_model = kwargs.get("llm_model")
        self._enable_activity_generation = kwargs.get("enable_activity_generation", True)
        self._activity_verbosity = kwargs.get("activity_verbosity", "medium")
        self._activity_llm_provider = kwargs.get("activity_llm_provider", "openai")
        self._activity_llm_model = kwargs.get("activity_llm_model", "o3-mini")

    @property
    def max_web_research_loops(self) -> int:
        if self._max_web_research_loops is not None:
            return self._max_web_research_loops
        env_value = os.environ.get("MAX_WEB_RESEARCH_LOOPS")
        return int(env_value or "10")

    @property
    def search_api(self):
        if self._search_api is not None:
            return self._search_api
        return SearchAPI(os.environ.get("SEARCH_API") or "tavily")

    @property
    def fetch_full_page(self) -> bool:
        if self._fetch_full_page is not None:
            return self._fetch_full_page
        return (os.environ.get("FETCH_FULL_PAGE") or "False").lower() in ("true", "1", "t")

    @property
    def include_raw_content(self) -> bool:
        if self._include_raw_content is not None:
            return self._include_raw_content
        return (os.environ.get("INCLUDE_RAW_CONTENT") or "True").lower() in ("true", "1", "t")

    @property
    def llm_provider(self):
        if self._llm_provider is not None:
            return self._llm_provider
        return LLMProvider(os.environ.get("LLM_PROVIDER") or "openai")

    @property
    def llm_model(self) -> str:
        if self._llm_model is not None:
            return self._llm_model
        return os.environ.get("LLM_MODEL") or "o3-mini"

    @property
    def enable_activity_generation(self) -> bool:
        if self._enable_activity_generation is not None:
            return self._enable_activity_generation
        return (os.environ.get("ENABLE_ACTIVITY_GENERATION") or "True").lower() in ("true", "1", "t")

    @property
    def activity_verbosity(self) -> ActivityVerbosity:
        if self._activity_verbosity is not None:
            return self._activity_verbosity
        return ActivityVerbosity((os.environ.get("ACTIVITY_VERBOSITY") or "medium").lower())

    @property
    def activity_llm_provider(self) -> LLMProvider:
        if self._activity_llm_provider is not None:
            return self._activity_llm_provider
        return LLMProvider((os.environ.get("ACTIVITY_LLM_PROVIDER") or "openai").lower())

    @property
    def activity_llm_model(self) -> str:
        if self._activity_llm_model is not None:
            return self._activity_llm_model
        return os.environ.get("ACTIVITY_LLM_MODEL") or "o3-mini"

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config["configurable"] if config and "configurable" in config else {}

        properties = [
            "max_web_research_loops", "search_api", "fetch_full_page",
            "include_raw_content", "llm_provider", "llm_model",
            "enable_activity_generation", "activity_verbosity",
            "activity_llm_provider", "activity_llm_model",
        ]

        values = {}
        for prop in properties:
            env_value = os.environ.get(prop.upper())
            config_value = configurable.get(prop)
            if config_value is not None:
                values[prop] = config_value
            elif env_value is not None:
                values[prop] = env_value

        return cls(**values)
