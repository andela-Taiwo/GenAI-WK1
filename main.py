"""
Simple LLM Application using Groq API
"""

import os
import logging
from typing import Optional, Dict, Any
from groq import Groq
from openai import OpenAI
from config import env_config
from util import get_provider

logger = logging.getLogger(__name__)


class LLMApp:
    # Provider configuration
    PROVIDERS = {
        "OpenAI": {
            "client_class": OpenAI,
            "env_var": "OPENAI_API_KEY",
            "models": [
                "gpt-4",
                "gpt-4o",
                "gpt-3.5-turbo",
                "gpt-4-turbo",
                "gpt-5",
                "gpt-5-mini",
                "gpt-5-nano",
            ],
        },
        "Groq": {
            "client_class": Groq,
            "env_var": "GROQ_API_KEY",
            "models": [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
            ],
        },
    }
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_TIMEOUT = 30

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
        **client_kwargs,
    ):
        """
        Initialize the LLM application

        Args:
            api_key: Groq API key or OPENAI API key(if None, reads from GROQ_API_KEY env var)
            model: Model to use for completions
            conversation_history: List of converstaions in a session
            max_retries: Maximum number of retries for API calls.
            timeout: Request timeout in seconds.
            **client_kwargs: Additional arguments to pass to the client constructor.
        """

        # self.conversation_history = []
        #  self._validate_imports()

        self.model = self._validate_model(model)
        self.provider = get_provider(self.model)
        self.api_key = self._get_api_key(api_key, self.provider)
        self.max_retries = max_retries
        self.timeout = timeout
        self.conversation_history: List[Dict[str, Any]] = []
        self.client_kwargs = client_kwargs

        self.client = self._initialize_client()
        self._verify_client_connection()

        logger.info(
            f"LLMApp initialized with provider: {self.provider}, model: {self.model}"
        )

    def _get_api_key(self, api_key: Optional[str], provider: str) -> str:
        """
        Get and validate API key from parameters or environment.

        Args:
            api_key: API key from parameter.
            provider: Provider name.

        Returns:
            Validated API key.

        Raises:
            ValueError: If API key is not found or invalid.
        """
        env_var = self.PROVIDERS[provider]["env_var"]
        print(api_key, "APIKEYEYEYEY")
        # Try parameter first, then environment variable
        final_api_key = api_key or os.getenv(env_var)

        if not final_api_key:
            raise ValueError(
                f"{provider} API key must be provided as parameter or set in "
                f"`{env_var}` environment variable"
            )

        if not isinstance(final_api_key, str) or not final_api_key.strip():
            raise ValueError(f"{provider} API key must be a non-empty string")

        # Basic validation (could be enhanced with regex patterns)
        if len(final_api_key.strip()) < 10:
            raise ValueError(f"{provider} API key appears to be invalid (too short)")

        return final_api_key.strip()

    def _initialize_client(self) -> Any:
        """
        Initialize the appropriate client for the provider.

        Returns:
            Initialized client instance.

        Raises:
            ConfigurationError: If client initialization fails.
        """
        provider_config = self.PROVIDERS[self.provider]
        client_class = provider_config["client_class"]

        try:
            # Common client configuration
            client_config = {
                "api_key": self.api_key,
                "max_retries": self.max_retries,
                "timeout": self.timeout,
            }

            # Add any additional client-specific kwargs
            client_config.update(self.client_kwargs)

            client = client_class(**client_config)
            logger.debug(f"Successfully initialized {self.provider} client")
            return client

        except Exception as e:
            logger.error("Failed to initialize %s client: %s", self.provider, e)
            raise ConfigurationError(
                f"Failed to initialize {self.provider} client: {str(e)}"
            ) from e

    def _verify_client_connection(self) -> None:
        """
        Verify that the client can connect to the API.

        Raises:
            ConnectionError: If connection test fails.
        """
        try:
            # Simple connection test - list models (lightweight operation)

            # Groq: similar lightweight operation
            self.client.models.list()

            logger.debug("Successfully verified connection to %s API", self.provider)

        except Exception as e:
            logger.error("Connection test failed for %s: %s", self.provider, e)
            raise ConnectionError(
                f"Failed to connect to {self.provider} API. "
                f"Please check your API key and network connection. Error: {str(e)}"
            ) from e

    def chat(self, user_message, system_prompt=None, temperature=1.0, max_tokens=1024):
        """
        Send a message and get a response

        Args:
            user_message: The user's message
            system_prompt: Optional system prompt to set context
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response

        Returns:
            The assistant's response text
        """

        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": f"""
                        {system_prompt}
                        Key attributes of the chatbot:
                        Name: {env_config.CHATBOT_IDENTITY["name"]}
                    """,
                }
            )

        # Add conversation history
        if self.conversation_history:
            messages.extend(self.conversation_history)

        # Add current user's message
        messages.append({"role": "user", "content": f"{user_message}"})
        chat_params = self.formatModelParameters(
            self.model, temperature, max_tokens, messages
        )
        # chat_params.update({"stream": True})
        print(chat_params)
        response = self.client.chat.completions.create(**chat_params)

        assistant_message = response.choices[0].message.content

        # Persist conversation history for this session
        try:
            # store both the user and assistant messages
            self.conversation_history.append(
                {"role": "user", "content": f"{user_message}"}
            )
            self.conversation_history.append(
                {"role": "assistant", "content": assistant_message}
            )
        except Exception:
            # If anything goes wrong while persisting history, ignore to avoid breaking the response
            pass

        return assistant_message

    def _validate_model(self, model: str) -> str:
        """
        Validate the model name and return normalized version.

        Args:
            model: Model name to validate.

        Returns:
            Normalized model name.

        Raises:
            ValueError: If model is not supported.
        """
        if not model or not isinstance(model, str):
            raise ValueError("Model must be a non-empty string")

        normalized_model = model.strip()

        # Check if model is supported by any provider
        supported_models = []
        for provider_config in self.PROVIDERS.values():
            supported_models.extend(provider_config["models"])

        if normalized_model not in supported_models:
            logger.warning(
                "Model '%s' not in predefined list. Supported models: %s. "
                "Attempting to use anyway.",
                normalized_model,
                supported_models,
            )

        return normalized_model

    def formatModelParameters(self, model_name, temperature, max_tokens, messages):
        model_info = get_provider(model_name)

        base_params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }

        if model_info == "OpenAI":
            base_params["max_completion_tokens"] = max_tokens
        else:
            base_params["max_tokens"] = max_tokens

        return base_params


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""

    pass


if __name__ == "__main__":
    # Initialize the app
    app = LLMApp()

    # while True:
    message = input("What do you want to ask: ")
    response = app.chat(message)
    print(f"\nAssistant Response: {response}\n")
