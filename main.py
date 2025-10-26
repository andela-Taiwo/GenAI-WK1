"""
Simple LLM Application using Groq API
"""

from groq import Groq
from openai import OpenAI
from config import env_config
from util import get_provider


class LLMApp:

    def __init__(self, api_key=None, model="llama-3.3-70b-versatile"):
        """
        Initialize the LLM application

        Args:
            api_key: Groq API key or OPENAI API key(if None, reads from GROQ_API_KEY env var)
            model: Model to use for completions
            conversation_history: List of converstaions in a session
        """

        self.api_key = api_key or env_config.groq_api_key
        if not self.api_key:
            raise ValueError(
                "Groq API key must be provided or set in `GROQ_API_KEY` environment variable"
            )
        self.provider = get_provider(model)
        if self.provider == "OpenAI":
            print("ettintg hhehehehehe")
            self.api_key = env_config.openai_api_key
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = Groq(api_key=self.api_key)
        self.model = model
        self.conversation_history = []

    def chat(self, user_message, system_prompt=None, temperature=0.5, max_tokens=1024):
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
                        Name: {env_config.CHATBOT_IDENTITY['name']}
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
        response = self.client.chat.completions.create(**chat_params)

        assistant_message = response.choices[0].message.content

        # Persist conversation history for this session
        try:
            # store both the user and assistant messages
            self.conversation_history.append({"role": "user", "content": f"{user_message}"})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
        except Exception:
            # If anything goes wrong while persisting history, ignore to avoid breaking the response
            pass

        return assistant_message


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


if __name__ == "__main__":

    # Initialize the app
    app = LLMApp()

    # while True:
    message = input("What do you want to ask: ")
    response = app.chat(message)
    print(f"\nAssistant Response: {response}\n")


