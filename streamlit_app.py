"""
Enhanced Streamlit Frontend for Multi-Provider LLM Chat Application
Supports both Groq and OpenAI models
"""

import streamlit as st
from typing import Dict, Any, Optional
from main import LLMApp
from config import env_config
from util import get_provider

# Page configuration
st.set_page_config(
    page_title="Multi-Provider LLM Chat",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


class ChatConfig:
    """Configuration manager for chat settings"""

    def __init__(self):
        self.CHATBOT_IDENTITY = env_config.CHATBOT_IDENTITY
        self.MODELS = {
            "Groq": [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ],
            "OpenAI": [
                "gpt-5",
                "gpt-5-mini",
                "gpt-5-nano",
            ],
        }
        self.ALL_MODELS = self.MODELS["Groq"] + self.MODELS["OpenAI"]


class SessionStateManager:
    """Manages session state initialization and persistence"""

    def __init__(self, chatbot_identity: Dict[str, str], models: list):
        self.chatbot_identity = chatbot_identity
        self.models = models

        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = self._get_initial_greeting()

        if "llm_app" not in st.session_state:
            st.session_state.llm_app = None

        if "selected_model" not in st.session_state:
            st.session_state.selected_model = models[0]

        if "api_key" not in st.session_state:
            st.session_state.api_key = LLMApp().api_key

    def _get_initial_greeting(self) -> list:
        """Get the initial greeting message based on chatbot identity"""
        return [
            {
                "role": "assistant",
                "content": f"Hi! I'm {self.chatbot_identity['name']}, your {self.chatbot_identity['role'].lower()}! ✨\n\n"
                f"I'm here to help you with {self.chatbot_identity['expertise'].lower()}. "
                f"How can I assist you today?",
            }
        ]

    def clear_history(self):
        """Clear chat history while maintaining initial greeting"""
        st.session_state.messages = self._get_initial_greeting()


class ParameterFormatter:
    """Handles formatting of model parameters for different providers"""

    def __init__(self, chatbot_identity: Dict[str, str]):
        self.chatbot_identity = chatbot_identity

    def get_system_prompt(self, custom_prompt: Optional[str] = None) -> str:
        """Get the system prompt, using custom if provided"""
        if custom_prompt and custom_prompt.strip():
            return custom_prompt

        return f"""
        You are {self.chatbot_identity["name"]}, a {self.chatbot_identity["role"]}.
        
        PERSONALITY: {self.chatbot_identity["personality"]}
        EXPERTISE: {self.chatbot_identity["expertise"]}
        COMMUNICATION STYLE: {self.chatbot_identity["style"]}
        BACKSTORY: {self.chatbot_identity["backstory"]}
        
        Key behaviors:
        - Always introduce yourself as {self.chatbot_identity["name"]}
        - Maintain your {self.chatbot_identity["personality"]} tone
        - Use your expertise in {self.chatbot_identity["expertise"]} to provide helpful guidance
        - Occasionally reference your backstory naturally
        - Be engaging and encouraging
        
        Never break character or reveal you're an AI model. Stay in role at all times.
        """

    def format_parameters(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
        user_message: str,
    ) -> Dict[str, Any]:
        """Format parameters for the LLM call"""
        final_system_prompt = self.get_system_prompt(system_prompt)

        params = {
            "user_message": user_message,
            "system_prompt": final_system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        return params


class ChatInterface:
    """Main chat interface handler"""

    def __init__(self):
        self.config = ChatConfig()
        self.state_manager = SessionStateManager(
            self.config.CHATBOT_IDENTITY, self.config.ALL_MODELS
        )
        self.formatter = ParameterFormatter(self.config.CHATBOT_IDENTITY)

    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        with st.sidebar:
            # Chatbot Identity Section
            st.header(f"👋 Meet {self.config.CHATBOT_IDENTITY['name']}!")
            st.write(f"**Role:** {self.config.CHATBOT_IDENTITY['role']}")
            st.write(f"**Expertise:** {self.config.CHATBOT_IDENTITY['expertise']}")
            st.write(f"**Personality:** {self.config.CHATBOT_IDENTITY['personality']}")

            st.divider()

            # Configuration Section
            st.header("⚙️ Configuration")

            # API Key Input
            api_key = st.text_input(
                "API Key",
                type="password",
                help="Enter your Groq or OpenAI API key",
                key="api_key_input",
                # value=st.session_state.get("api_key", ""),
            )
            st.session_state.api_key = api_key

            # Provider Selection
            provider = st.radio(
                "Select Provider:",
                options=["Groq", "OpenAI"],
                index=0,
                key="provider_selector",
                help="Choose between Groq and OpenAI models",
            )

            # Model Selection based on provider
            model = st.selectbox(
                "Model",
                options=self.config.MODELS[provider],
                index=0,
                help="Select the model to use",
                key="model_selector",
            )
            st.session_state.selected_model = model

            # Model parameters
            col1, col2 = st.columns(2)

            with col1:
                st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Controls randomness: Lower = more deterministic, Higher = more creative",
                    key="temp_slider",
                )

            with col2:
                st.slider(
                    "Max Tokens",
                    min_value=128,
                    max_value=4096,
                    value=1024,
                    step=128,
                    help="Maximum length of response",
                    key="tokens_slider",
                )

            # System Prompt
            st.text_area(
                "Custom System Prompt (Optional)",
                placeholder="Override the default system prompt...",
                help="Customize the assistant's behavior and context",
                key="system_prompt_input",
                height=100,
            )

            st.divider()

            # Chat Management

            if st.button("🗑️ Clear Chat", use_container_width=True):
                self.state_manager.clear_history()
                st.rerun()

    def render_chat_messages(self):
        """Render all chat messages"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def initialize_llm_app(self, api_key: str, model: str) -> bool:
        """Initialize the LLM app with error handling"""
        try:
            st.session_state.llm_app = LLMApp(api_key=api_key, model=model)
            return True
        except Exception as e:
            st.error(f"❌ Error initializing LLM App: {str(e)}")
            return False

    def handle_user_input(self, user_input: str):
        """Process user input and generate response"""
        # Validation
        if not st.session_state.api_key:
            st.warning("🔑 Please enter your API key in the sidebar")
            return

        # Initialize LLM app if needed
        if st.session_state.llm_app is None:
            if not self.initialize_llm_app(
                st.session_state.api_key, st.session_state.selected_model
            ):
                return

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("💭 Thinking..."):
                try:
                    # Get current settings
                    temperature = st.session_state.temp_slider
                    max_tokens = st.session_state.tokens_slider
                    system_prompt = st.session_state.system_prompt_input

                    # Format parameters
                    chat_params = self.formatter.format_parameters(
                        st.session_state.selected_model,
                        temperature,
                        max_tokens,
                        system_prompt,
                        user_input,
                    )

                    # Generate response
                    response = st.session_state.llm_app.chat(**chat_params)

                    # Display response
                    st.markdown(response)

                    # Update conversation history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    # Don't add error messages to conversation history

    def render_main_interface(self):
        """Render the main chat interface"""
        st.title("💬 Multi-Provider LLM Chat")

        # Provider badge
        provider = get_provider(st.session_state.selected_model)
        provider_color = "blue" if provider == "OpenAI" else "green"
        st.markdown(
            f"**Current Provider:** :{provider_color}[{provider}] | "
            f"**Model:** `{st.session_state.selected_model}`"
        )

        st.markdown(
            "Chat with powerful LLMs from different providers. "
            "Configure your settings in the sidebar. 👈"
        )

        st.divider()

        # Display chat messages
        self.render_chat_messages()

        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            self.handle_user_input(prompt)

    def run(self):
        """Main application runner"""
        self.render_sidebar()
        self.render_main_interface()


def main():
    """Main application entry point"""
    app = ChatInterface()
    app.run()


if __name__ == "__main__":
    main()
