"""
LLM Client Module - OpenAI Only

Provides unified LLM client interfaces for the deep research engine.
Supports OpenAI API compatible endpoints (including custom base URLs).

Public interfaces preserved for future frontend/backend reconnection:
- get_llm_client(provider, model_name) -> sync client
- get_async_llm_client(provider, model_name) -> async client
- get_model_response(llm, system_prompt, user_prompt, config) -> str
- get_formatted_system_prompt() -> str
- get_available_providers() -> list
- MODEL_CONFIGS, SYSTEM_PROMPT_TEMPLATE, SimpleOpenAIClient, ReasoningEffortOpenAIClient
"""

import os
import logging
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage, ChatMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ==================== API Keys ====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# ==================== Token Limits ====================
OPENAI_MAX_TOKENS = 30000

# ==================== Date Information ====================
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")
CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month
CURRENT_DAY = datetime.now().day
ONE_YEAR_AGO = datetime.now().replace(year=datetime.now().year - 1).strftime("%Y-%m-%d")
YTD_START = f"{CURRENT_YEAR}-01-01"

# ==================== Model Configurations ====================
MODEL_CONFIGS = {
    "openai": {
        "available_models": [
            "gpt-4o",
            "o4-mini",
            "o4-mini-high",
            "o3-mini",
            "o3-mini-reasoning",
        ],
        "default_model": "o4-mini",
        "requires_api_key": OPENAI_API_KEY,
    },
}

# ==================== System Prompt ====================
SYSTEM_PROMPT_TEMPLATE = """
<intro>
You excel at the following tasks:
1. Information gathering, fact-checking, and documentation
2. Data processing, analysis, and visualization
3. Writing multi-chapter articles and in-depth research reports
4. Creating websites, applications, and tools
5. Using programming to solve various problems beyond development
6. Various tasks that can be accomplished using computers and the internet
7. IMPORTANT: The current date is {current_date}. Always use this as your reference instead of datetime.now().
</intro>

<date_information>
Current date: {current_date}
Current year: {current_year}
Current month: {current_month}
Current day: {current_day}
One year ago: {one_year_ago}
Year-to-date start: {ytd_start}
</date_information>

<requirements>
When writing code, your code MUST:
1. Start with installing necessary packages (e.g., '!pip install matplotlib pandas')
2. Include robust error handling for data retrieval and processing
3. Print sample data to validate successful retrieval
4. Properly handle data structures based on the returned format:
   - For multi-level dataframes: Use appropriate indexing like df['Close'] or df.loc[:, ('Price', 'Close')]
   - For single-level dataframes: Access columns directly
5. If asked to create visualizations with professional styling including:
   - Clear title and axis labels
   - Grid lines for readability
   - Appropriate date formatting on x-axis using matplotlib.dates
   - Legend when plotting multiple series
6. Include data validation to check for:
   - Dataset sizes and date ranges
   - Missing values (NaN) and their handling
   - Data types and any necessary conversions
7. Implement appropriate data transformations like:
   - Normalizing prices to a common baseline
   - Calculating moving averages or other indicators
   - Computing ratios or correlations between assets
8. IMPORTANT: When fetching date-sensitive data:
   - DO NOT use datetime.now() in your code
   - Instead, use these fixed dates: current="{current_date}", year_start="{ytd_start}", year_ago="{one_year_ago}"
</requirements>
"""

ERROR_CORRECTION_PROMPT = """
<error_correction>
The previous code failed to execute properly. I'm providing the error logs below.
Please fix the code to address these issues and ensure it runs correctly:

ERROR LOGS:
{error_logs}

Common issues to check:
1. Date handling issues - Use explicit date ranges (e.g., '{ytd_start}' instead of datetime.now())
2. Data structure validation - Verify the expected structure of returned data
3. Library compatibility - Ensure all functions used are available in the imported libraries
4. Error handling - Add more robust try/except blocks
</error_correction>
"""


# ==================== OpenAI Clients ====================

class SimpleOpenAIClient:
    """OpenAI client that bypasses LangChain for models that don't support temperature (o-series)."""

    def __init__(self, model_name: str, api_key: str, max_tokens: int = OPENAI_MAX_TOKENS):
        self.model_name = model_name
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )

    @traceable
    def invoke(self, messages, config=None):
        """Invoke the model with the given messages without using temperature."""
        try:
            openai_messages = []
            for msg in messages:
                role = msg.type
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                openai_messages.append({"role": role, "content": msg.content})

            # o-series models use max_completion_tokens
            token_param = {}
            if self.model_name.startswith(("o3", "o4", "o1")):
                token_param = {"max_completion_tokens": self._max_tokens}
            else:
                token_param = {"max_tokens": self._max_tokens}

            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                **token_param,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[SimpleOpenAIClient ERROR] {str(e)}")
            raise


class ReasoningEffortOpenAIClient(SimpleOpenAIClient):
    """OpenAI client with reasoning_effort parameter for enhanced models like o3-mini-high."""

    @traceable
    def invoke(self, messages, config=None):
        """Invoke the model with high reasoning effort."""
        try:
            openai_messages = []
            for msg in messages:
                role = msg.type
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                openai_messages.append({"role": role, "content": msg.content})

            token_param = {}
            if self.model_name.startswith(("o3", "o4", "o1")):
                token_param = {"max_completion_tokens": self._max_tokens}
            else:
                token_param = {"max_tokens": self._max_tokens}

            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                reasoning_effort="high",
                **token_param,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ReasoningEffortOpenAIClient ERROR] {str(e)}")
            raise


# ==================== Public Interface Functions ====================

def get_available_providers():
    """Returns a list of available providers based on configured API keys."""
    available_providers = []
    for provider, config in MODEL_CONFIGS.items():
        if config.get("requires_api_key"):
            available_providers.append(provider)
    return available_providers


def get_llm_client(provider, model_name=None):
    """
    Get the appropriate LLM client based on provider and model name.

    Args:
        provider: The provider name (currently only 'openai' supported)
        model_name: The model name (optional, uses default if not provided)

    Returns:
        A synchronous LangChain chat model client or SimpleOpenAIClient
    """
    if provider != "openai":
        raise ValueError(
            f"Unsupported provider: {provider}. Only 'openai' is supported. "
            f"Set LLM_PROVIDER=openai in your .env file."
        )

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in environment")

    if not model_name:
        model_name = MODEL_CONFIGS["openai"]["default_model"]

    # o4-mini-high: use ReasoningEffortOpenAIClient
    if model_name == "o4-mini-high":
        print(f"Using ReasoningEffortOpenAIClient for {model_name}")
        return ReasoningEffortOpenAIClient(
            model_name="o4-mini", api_key=OPENAI_API_KEY, max_tokens=OPENAI_MAX_TOKENS
        )
    # o3-mini-reasoning: use ReasoningEffortOpenAIClient
    elif model_name == "o3-mini-reasoning":
        print(f"Using ReasoningEffortOpenAIClient for o3-mini with high reasoning effort")
        return ReasoningEffortOpenAIClient(
            model_name="o3-mini", api_key=OPENAI_API_KEY, max_tokens=OPENAI_MAX_TOKENS
        )
    else:
        # Standard ChatOpenAI for all other models
        print(f"Using ChatOpenAI for {model_name}")
        return ChatOpenAI(
            model_name=model_name.replace("-reasoning", ""),
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            max_tokens=OPENAI_MAX_TOKENS,
            streaming=False,
        )


async def get_async_llm_client(provider, model_name=None):
    """
    Get an asynchronous LLM client for the given provider.

    Args:
        provider: The provider name (currently only 'openai' supported)
        model_name: The model name (optional, uses default if not provided)

    Returns:
        An async LangChain chat model client for OpenAI
    """
    if provider != "openai":
        raise ValueError(
            f"Async client not supported for provider: {provider}. Only 'openai' is supported."
        )

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in environment")

    if not model_name:
        model_name = MODEL_CONFIGS["openai"]["default_model"]

    # Handle special model name variants
    if model_name == "o4-mini-high":
        effective_model_name = "o4-mini"
    else:
        effective_model_name = model_name.replace("-reasoning", "")

    logger.info(f"[get_async_llm_client] Creating async ChatOpenAI with model {effective_model_name}")

    return ChatOpenAI(
        model_name=effective_model_name,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        max_tokens=OPENAI_MAX_TOKENS,
        streaming=False,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_model_response(llm, system_prompt: str, user_prompt: str, config=None):
    """
    Get a response from an LLM.

    Args:
        llm: The chat model client (LangChain model or SimpleOpenAIClient)
        system_prompt: The system prompt to use
        user_prompt: The user prompt to use
        config: Optional config for LangSmith tracing

    Returns:
        The model's response as a string
    """
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        if isinstance(llm, SimpleOpenAIClient):
            model_name = llm.model_name
        else:
            model_name = getattr(llm, "model_name", None)
            if model_name is None:
                model_name = getattr(llm, "model", "unknown model")

        print(f"🔄 Sending messages to {model_name}...")
        response = llm.invoke(messages, config=config)

        if isinstance(response, str):
            return response
        else:
            return response.content
    except Exception as e:
        print(f"[Model API ERROR] {str(e)}")
        raise


def get_formatted_system_prompt():
    """Format the system prompt template with current date information."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        current_date=CURRENT_DATE,
        current_year=CURRENT_YEAR,
        current_month=CURRENT_MONTH,
        current_day=CURRENT_DAY,
        one_year_ago=ONE_YEAR_AGO,
        ytd_start=YTD_START,
    )
