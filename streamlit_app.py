"""
Streamlit Frontend for Groq LLM Application
"""

import streamlit as st
from main import LLMApp
from config import env_config

# page configuration
st.set_page_config(
    page_title="Simple LLM Chat Application", page_icon="🤖", layout="centered"
)


CHATBOT_IDENTITY = env_config.CHATBOT_IDENTITY
MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120a",
    "openai/gpt-oss-120b",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
]


def get_system_prompt():
    return f"""
    You are {CHATBOT_IDENTITY['name']}, a {CHATBOT_IDENTITY['role']}.
    
    EXPERTISE: {CHATBOT_IDENTITY['expertise']}
    COMMUNICATION STYLE: {CHATBOT_IDENTITY['style']}
    
    Key behaviors:
    - Maintain your {CHATBOT_IDENTITY['personality']} tone
    - Use your expertise in {CHATBOT_IDENTITY['expertise']} to provide helpful guidance
    - Be engaging and encouraging
    """


# initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": f"Hi! I'm {CHATBOT_IDENTITY['name']}, your {CHATBOT_IDENTITY['role'].lower()}! ✨\n\nI'm here to help you with {CHATBOT_IDENTITY['expertise'].lower()}",
        }
    ]

if "llm_app" not in st.session_state:
    st.session_state.llm_app = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = MODELS[0]

# Title and description
st.title("Groq LLM Chat Application")
st.markdown(
    "Chat with a powerful LLM from Groq. Please enter your Groq API key in the sidebar to get started. You can get one at https://console.groq.com/"
)


def clear_history():
    st.session_state.messages = []


def handleChangeModel():
    newModel = st.session_state.model_selector
    st.session_state.selected_model = newModel
    # st.session_state.llm_app = LLMApp(api_key=api_key, model=model)


# Implement sidebar for configuration
with st.sidebar:
    st.header(f"Meet {CHATBOT_IDENTITY['name']}! 🤖")
    st.write(f"**Role:** {CHATBOT_IDENTITY['role']}")
    st.write(f"**Expertise:** {CHATBOT_IDENTITY['expertise']}")
    st.divider()
    st.header("Configuration")
    # API key input
    api_key = st.sidebar.text_input(
        "Groq API Key", type="password", help="Enter your Groq API key"
    )
    if not api_key:
        api_key = LLMApp().api_key

    # Model selection
    model = st.selectbox(
        "Model",
        options=MODELS,
        index=MODELS.index(st.session_state.selected_model),
        help="Select the model to use",
        on_change=handleChangeModel,
        key="model_selector",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Select a value to control response randomness. Higher values make output more random.",
    )

    max_tokens = st.slider(
        "Max Tokens",
        min_value=256,
        max_value=2048,
        value=1024,
        step=256,
        help="Set the response length",
    )

    system_prompt = st.text_area(
        "System Prompt (Optional)",
        placeholder="You are a helpful assistant...",
        help="Set the context and behaviour of the assistant",
    )

    # Clear chat button
    st.button("Clear Chat History", use_container_width=True, on_click=clear_history)


# display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    if st.session_state.llm_app is None:
        try:
            st.session_state.llm_app = LLMApp(
                api_key=api_key, model=st.session_state.selected_model
            )
        except Exception as e:
            st.error(f"Error initializing LLM App: {str(e)}")
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar")

    else:
        st.session_state.messages.append({"role": "user", "content": f"{prompt}"})

        with st.chat_message("user"):
            st.markdown(prompt)

        # get assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.llm_app.chat(
                        user_message=prompt,
                        system_prompt=(
                            system_prompt if system_prompt else get_system_prompt()
                        ),
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"{response}"}
                    )
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
