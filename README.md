
# 🤖 Multi-Provider LLM Chat Application

A sophisticated Streamlit-based chat application that supports both **Groq** and **OpenAI** language models. Features a customizable AI assistant personality, real-time chat interface, and seamless switching between different LLM providers.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-00FF7F?style=for-the-badge)

## ✨ Features

### 🔄 Multi-Provider Support
- **Groq Models**: LLaMA 3.1 8B, LLaMA 3.3 70B, Mixtral 8x7B, Gemma2 9B
- **OpenAI Models**: GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo, GPT-4o Mini
- Seamless switching between providers

### 🎭 Customizable AI Personality
- Define your assistant's name, role, and expertise
- Custom personality traits and communication style
- Backstory integration for consistent character
- Optional custom system prompts

### ⚙️ Advanced Configuration
- Adjustable temperature (creativity control)
- Configurable response length (max tokens)
- Real-time parameter tuning
- Session persistence

### 💬 User Experience
- Real-time chat interface
- Conversation history
- Message streaming (if supported by provider)
- Error handling and validation
- Mobile-responsive design

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API Key ([Get one here](https://console.groq.com/))
- OpenAI API Key ([Get one here](https://platform.openai.com/))
- Streamlit: For building the interactive web frontend.
- python-dotenv: For loading environment variables (API keys).
- [UV](https://github.com/astral-sh/uv) - The ultra-fast Python package installer


 ### Installation with UV

  1. **Install UV** (if you haven't already):
     ```bash
     # On macOS/Linux
     curl -LsSf https://astral.sh/uv/install.sh | sh

     # On Windows
     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

     # Or via pip
     pip install uv

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/andela-Taiwo/llm-chat-app.git
   cd llm-chat-app

   # UV will automatically create a virtual environment
    uv sync

2. **Configure environment**
   - cp .env.example .env

3. ### Run the application
   - streamlit run main.py