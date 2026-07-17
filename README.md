# agent-forge

🚀 **Your Modular AI Agent Framework**

Built and maintained by **[Naveen Muppana](https://github.com/naveenmuppana)**.

---

## 💡 About
`agent-forge` is a lightweight, modular foundation for building intelligent AI agents. It provides a structured path from direct LLM interactions to complex, tool-calling agents and RAG-enabled document search.

## 🛠 Features
- **FastAPI Backend**: Deployable service for chat and agent logic.
- **Tool-Calling**: Out-of-the-box weather agent integration.
- **RAG Capability**: Semantic search and Q&A powered by MongoDB Atlas.
- **Modular Core**: Clean separation of concerns for easier maintenance and customization.

## 📂 Project Structure
- `agent_service.py`: Main API application.
- `chat_client.py`: Interactive CLI client.
- `weather_agent.py`: Agent demonstrating function calling.
- `core.py`: Shared utilities and client configurations.

## 🚀 Getting Started
```bash
git clone https://github.com/naveenmuppana/agent-forge.git
cd agent-forge
pip install -r requirements.txt
# Configure your .env file
uvicorn agent_service:app --reload
```

---

*Built for rapid AI experimentation and production-ready agent deployment.*
