from typing import Dict, Tuple, Any, List
from openai import OpenAI
import requests
import os
import time
import random
from utils.json import extract_json
from utils.logging import setup_logger

logger = setup_logger(__name__)

AVAIBLE_MODELS = {
    "openai": ["gpt-4o", "o4-mini"],
    "ollama": ["deepseek-r1:14b"],
    "openrouter": ["gemini-2.5-flash"],
}


class LLMService:
    """Service for interacting with Language Learning Models (LLM)."""

    DEFAULT_CONFIG = {
        "llm_service": "openai",
        "model": "",
        "temperature": 1,
        "max_completion_tokens": 4096,
        "use_response_api": False,
        "reasoning_enabled": False,
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize LLM service with configuration.

        Args:
            config: Dictionary containing service configuration
        """
        self.config = {**self.DEFAULT_CONFIG, **config}
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self._setup_parameters()
        self.client = self._initialize_client()

    def _setup_parameters(self) -> None:
        """Set up service parameters from config."""
        self.service = self.config["llm_service"]
        self.model = self.config["model"]
        self.temperature = self.config["temperature"]
        self.max_completion_tokens = self.config["max_completion_tokens"]
        self.model_specific_config = self.config.get("model_specific_config", {})
        self.use_response_api = self.config.get("use_response_api", False)
        self.reasoning_enabled = self.config.get("reasoning_enabled", False)

    def _initialize_client(self) -> OpenAI:
        """Initialize the OpenAI client based on service type."""
        if self.service == "openai":
            self.sdk = "openai"
            return OpenAI(api_key=self.openai_api_key)
        elif self.service == "ollama":
            self.sdk = "openai"
            return OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
        elif self.service == "openrouter":
            self.sdk = "openrouter"
            return None

        raise ValueError(f"Unsupported LLM service: {self.service}")

    def _prepare_completion_args(
        self, messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Prepare arguments for completion API call."""
        completion_args = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
        }
        if self.model_specific_config:
            completion_args.update(self.model_specific_config)

        return completion_args

    def _prepare_response_args(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Prepare arguments for response API call."""
        completion_args = {
            "model": self.model,
            "input": messages,
            "temperature": self.temperature,
            "max_output_tokens": self.max_completion_tokens,
        }
        if self.model_specific_config:
            completion_args.update(self.model_specific_config)

        return completion_args

    def _prepare_openrouter_args(
        self,
        messages: List[Dict[str, str]],
        reasoning_enabled: bool = False,
    ) -> Dict[str, Any]:
        """Prepare arguments for openrouter API call."""

        completion_args = {
            "model": self.model,
            "messages": messages,
            # OpenRouter follows the OpenAI Chat Completions API
            # Use "max_tokens" (not max_completion_tokens) and include temperature
            "temperature": self.temperature,
            "max_tokens": self.max_completion_tokens,
        }
        if reasoning_enabled:
            completion_args["reasoning"] = {"enabled": True}

        if self.model_specific_config:
            completion_args.update(self.model_specific_config)

        logger.debug(completion_args)

        return completion_args

    def _process_response_content(
        self,
        content: str,
    ) -> Tuple[Any, Dict[str, str]]:
        """Process the response content and extract any extra information."""
        extra = {}

        # Extract thoughts from think tags if present
        if "<think>" in content and "</think>" in content:
            start = content.find("<think>")
            end = content.find("</think>") + 8
            extra["think"] = content[start + 7 : end - 8]
            content = content[:start] + content[end:]

        parsed_results = extract_json(content)
        if not parsed_results:
            raise Exception(
                f"Could not parse JSON from LLM response: {content[:200]}..."
            )

        result = parsed_results[0]

        return result, extra

    def _extract_usage_info(self, usage_obj) -> Dict[str, Any]:
        """Extract usage information from either a CompletionUsage object or dictionary."""
        if isinstance(usage_obj, dict):
            return {
                "prompt_tokens": usage_obj.get("prompt_tokens", 0),
                "completion_tokens": usage_obj.get("completion_tokens", 0),
                "total_tokens": usage_obj.get("total_tokens", 0),
            }
        else:
            # Handle CompletionUsage object
            return {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                "total_tokens": getattr(usage_obj, "total_tokens", 0),
            }

    def _extract_response_usage_info(self, usage_obj) -> Dict[str, Any]:
        """Extract usage information for response API from either an object or dictionary."""
        if isinstance(usage_obj, dict):
            return {
                "input_tokens": usage_obj.get("input_tokens", 0),
                "output_tokens": usage_obj.get("output_tokens", 0),
                "total_tokens": usage_obj.get("total_tokens", 0),
            }
        else:
            # Handle response usage object
            return {
                "input_tokens": getattr(usage_obj, "input_tokens", 0),
                "output_tokens": getattr(usage_obj, "output_tokens", 0),
                "total_tokens": getattr(usage_obj, "total_tokens", 0),
            }

    def get_completion(
        self,
        messages: List[Dict[str, str]],
        save_messages_n_contents: bool = False,
        return_raw_response: bool = False,
        return_all_api_responses: bool = False,
    ) -> Tuple[Any, Dict[str, Any], Dict[str, str]]:
        """Get completion from the LLM service.

        Args:
            messages: List of message dictionaries
            require_response_formatter: Whether to require response formatting - kept for backward compatibility
            save_messages_n_contents: Whether to save messages and content in extra info
            return_raw_response: Whether to return raw response without processing

        Returns:
            Tuple containing (result, usage_stats, extra_info)
        """

        if self.sdk == "openai":
            return (
                self._get_response(
                    messages, save_messages_n_contents, return_raw_response
                )
                if self.use_response_api
                else self._get_completion(
                    messages, save_messages_n_contents, return_raw_response
                )
            )
        elif self.sdk == "openrouter":
            return self._get_openrouter_response(
                messages,
                save_messages_n_contents,
                return_raw_response,
                return_all_api_responses,
            )

    def _get_openrouter_response(
        self,
        messages: List[Dict[str, str]],
        save_messages_n_contents: bool = False,
        return_raw_response: bool = False,
        return_all_api_responses: bool = False,
    ) -> Tuple[Any, Dict[str, Any], Dict[str, str]]:

        completion_args = self._prepare_openrouter_args(
            messages, self.reasoning_enabled
        )

        # Chat completion (POST /chat/completions)
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        # Optional attribution headers if provided via env (recommended by OpenRouter)
        app_url = os.getenv("OPENROUTER_APP_URL")
        app_title = os.getenv("OPENROUTER_APP_TITLE")
        if app_url:
            headers["HTTP-Referer"] = app_url
        if app_title:
            headers["X-Title"] = app_title

        # Retry/backoff for transient OpenRouter errors (429/5xx/timeouts)
        max_attempts = int(self.config.get("openrouter_max_attempts", 8))
        base_backoff_s = float(self.config.get("openrouter_backoff_base_s", 1.0))
        max_backoff_s = float(self.config.get("openrouter_backoff_max_s", 60.0))
        timeout_s = float(self.config.get("openrouter_timeout_s", 120.0))

        last_err: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=completion_args,
                    timeout=timeout_s,
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                # Retryable transport error
                if attempt >= max_attempts:
                    raise RuntimeError(f"OpenRouter request failed after retries: {e}")
                sleep_s = min(max_backoff_s, base_backoff_s * (2 ** (attempt - 1)))
                sleep_s *= 0.5 + random.random()  # jitter in [0.5, 1.5)
                time.sleep(sleep_s)
                continue

            # Fast-path: retry on 429/5xx without needing JSON parse
            if response.status_code == 429 or 500 <= response.status_code <= 599:
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"OpenRouter API error after retries (status {response.status_code}): {response.text[:500]}"
                    )
                sleep_s = min(max_backoff_s, base_backoff_s * (2 ** (attempt - 1)))
                sleep_s *= 0.5 + random.random()
                time.sleep(sleep_s)
                continue

            # Robust error handling: ensure we have a valid JSON and expected fields
            try:
                response_json = response.json()
            except ValueError as e:
                last_err = e
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"OpenRouter returned non-JSON (status {response.status_code}): {response.text[:500]}"
                    )
                sleep_s = min(max_backoff_s, base_backoff_s * (2 ** (attempt - 1)))
                sleep_s *= 0.5 + random.random()
                time.sleep(sleep_s)
                continue

            if response.status_code >= 400 or "error" in response_json:
                error_message = (
                    response_json.get("error", {}).get("message")
                    if isinstance(response_json.get("error"), dict)
                    else response_json.get("error")
                )

                # Retry only for clearly transient cases
                if (
                    response.status_code in (408, 429)
                    or 500 <= response.status_code <= 599
                ):
                    if attempt >= max_attempts:
                        raise RuntimeError(
                            f"OpenRouter API error after retries (status {response.status_code}): {error_message or response_json}"
                        )
                    sleep_s = min(max_backoff_s, base_backoff_s * (2 ** (attempt - 1)))
                    sleep_s *= 0.5 + random.random()
                    time.sleep(sleep_s)
                    continue

                raise RuntimeError(
                    f"OpenRouter API error (status {response.status_code}): {error_message or response_json}"
                )

            # Success path: break out with parsed JSON
            break
        else:
            # Should be unreachable, but keep a clear failure mode
            raise RuntimeError(f"OpenRouter request failed: {last_err}")

        if "choices" not in response_json or not response_json["choices"]:
            raise RuntimeError(f"OpenRouter response missing choices: {response_json}")

        content = response_json["choices"][0]["message"]["content"]

        extra = {}

        if return_raw_response:
            result = content
        else:
            result, extra = self._process_response_content(content)
        if save_messages_n_contents:
            extra["messages"] = messages
            extra["content"] = content

        if return_all_api_responses:
            all_api_responses = response_json
            extra["all_api_responses"] = all_api_responses

        usage = response_json.get("usage", {})

        return result, usage, extra

    def _get_completion(
        self,
        messages: List[Dict[str, str]],
        save_messages_n_contents: bool = False,
        return_raw_response: bool = False,
    ) -> Tuple[Any, Dict[str, Any], Dict[str, str]]:
        """Get completion from the LLM service.

        Args:
            messages: List of message dictionaries
            output_format: Desired output format ("json" or other)

        Returns:
            Tuple containing (result, usage_stats, extra_info)
        """

        completion_args = self._prepare_completion_args(
            messages,
        )
        response = self.client.chat.completions.create(**completion_args)

        content = response.choices[0].message.content

        if return_raw_response:
            result = content
            extra = {}
        else:
            result, extra = self._process_response_content(content)
        if save_messages_n_contents:
            extra["messages"] = messages
            extra["content"] = content

        usage = self._extract_usage_info(response.usage)

        return result, usage, extra

    def _get_response(
        self,
        messages: List[Dict[str, str]],
        save_messages_n_contents: bool = False,
        return_raw_response: bool = False,
    ) -> Tuple[Any, Dict[str, Any], Dict[str, str]]:
        """Get response from the LLM service.

        Args:
            messages: List of message dictionaries
            output_format: Desired output format ("json" or other)

        Returns:
            Tuple containing (result, usage_stats, extra_info)
        """

        response_args = self._prepare_response_args(
            messages,
        )
        response = self.client.responses.create(**response_args)

        content = response.output_text

        if return_raw_response:
            result = content
            extra = {}
        else:
            result, extra = self._process_response_content(
                content,
            )
        if save_messages_n_contents:
            extra["messages"] = messages
            extra["content"] = content

        usage = self._extract_response_usage_info(response.usage)

        return result, usage, extra
