import os
from typing import Optional, Dict, Any
import requests
import logging

from openai import OpenAI


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


logging.getLogger('google').setLevel(logging.WARNING)
logging.getLogger('google.genai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# Read API keys from environment variables
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
READ_API_KEY = os.environ.get("JINA_API_KEY", "")

# OpenAI-compatible gateway defaults (e.g. third-party proxies exposing /v1/chat/completions)
FACT_Model = "gemini-2.5-flash-preview-05-20"
Model = "gemini-2.5-pro-preview-06-05"


def _eval_backend() -> str:
    return os.environ.get("DRB_EVAL_BACKEND", "openai_compat").strip().lower()


def _default_main_model() -> str:
    env = os.environ.get("DRB_EVAL_MODEL")
    if env:
        return env
    return "gemini-2.5-pro" if _eval_backend() == "google_genai" else Model


def _default_fact_model() -> str:
    env = os.environ.get("DRB_FACT_MODEL")
    if env:
        return env
    return "gemini-2.5-flash" if _eval_backend() == "google_genai" else FACT_Model


def _genai_response_text(response: Any) -> str:
    t = getattr(response, "text", None)
    if t:
        return t
    cands = getattr(response, "candidates", None) or []
    if not cands:
        return ""
    parts = getattr(cands[0].content, "parts", None) or []
    out = []
    for p in parts:
        txt = getattr(p, "text", None)
        if txt:
            out.append(txt)
    return "".join(out)


class AIClient:
    """DRB / RACE 打分用 LLM。

    - ``DRB_EVAL_BACKEND=openai_compat``（默认）：OpenAI SDK + ``OPENAI_BASE_URL``，模型名为网关侧字符串。
    - ``DRB_EVAL_BACKEND=google_genai``：官方 `google-genai`，使用 `GOOGLE_API_KEY`（或 `GEMINI_API_KEY`）。
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._backend = _eval_backend()
        self.model = model if model is not None else _default_main_model()
        self._genai_client = None

        if self._backend == "google_genai":
            self.api_key = (
                api_key
                or os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
            )
            if not self.api_key:
                raise ValueError(
                    "DRB_EVAL_BACKEND=google_genai 需要设置 GOOGLE_API_KEY（或 GEMINI_API_KEY）。"
                )
            self.client = None
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key not provided! Please set OPENAI_API_KEY environment variable."
                )
            _kw: Dict[str, Any] = {"api_key": self.api_key}
            if BASE_URL:
                _kw["base_url"] = BASE_URL
            self.client = OpenAI(**_kw)

    def _ensure_genai_client(self):
        if self._genai_client is None:
            try:
                from google import genai
            except ImportError as e:
                raise ImportError(
                    "请安装 google-genai：pip install google-genai"
                ) from e
            self._genai_client = genai.Client(api_key=self.api_key)
        return self._genai_client

    def generate(self, user_prompt: str, system_prompt: str = "", model: Optional[str] = None) -> str:
        model_to_use = model or self.model
        try:
            if self._backend == "google_genai":
                return self._generate_google_genai(
                    user_prompt, system_prompt, model_to_use
                )
            return self._generate_openai_compat(
                user_prompt, system_prompt, model_to_use
            )
        except Exception as e:
            raise Exception(f"Failed to generate content: {str(e)}") from e

    def _generate_openai_compat(
        self, user_prompt: str, system_prompt: str, model_to_use: str
    ) -> str:
        response = self.client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI-compatible API returned empty message content")
        return content

    def _generate_google_genai(
        self, user_prompt: str, system_prompt: str, model_to_use: str
    ) -> str:
        from google.genai import types

        client = self._ensure_genai_client()
        config = (
            types.GenerateContentConfig(system_instruction=system_prompt)
            if system_prompt
            else None
        )
        response = client.models.generate_content(
            model=model_to_use,
            contents=user_prompt,
            config=config,
        )
        text = _genai_response_text(response)
        if not text:
            raise RuntimeError("Google GenAI returned empty text")
        return text

class WebScrapingJinaTool:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("Jina API key not provided! Please set JINA_API_KEY environment variable.")

    def __call__(self, url: str) -> Dict[str, Any]:
        try:
            jina_url = f'https://r.jina.ai/{url}'
            headers = {
                "Accept": "application/json",
                'Authorization': self.api_key,
                'X-Timeout': "60000",
                "X-With-Generated-Alt": "true",
            }
            response = requests.get(jina_url, headers=headers)

            if response.status_code != 200:
                raise Exception(f"Jina AI Reader Failed for {url}: {response.status_code}")

            response_dict = response.json()

            return {
                'url': response_dict['data']['url'],
                'title': response_dict['data']['title'],
                'description': response_dict['data']['description'],
                'content': response_dict['data']['content'],
                'publish_time': response_dict['data'].get('publishedTime', 'unknown')
            }

        except Exception as e:
            logger.error(str(e))
            return {
                'url': url,
                'content': '',
                'error': str(e)
            }
        
jina_tool = WebScrapingJinaTool()

def scrape_url(url: str) -> Dict[str, Any]:
    return jina_tool(url)
    
def call_model(user_prompt: str) -> str:
    client = AIClient(model=_default_fact_model())
    return client.generate(user_prompt)

if __name__ == "__main__":
    url = ""
    result = scrape_url(url)
    print(result)