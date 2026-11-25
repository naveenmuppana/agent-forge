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

## 🤖 Personal Multi-Agent Assistant (`assistant/`)
A pipeline of cooperating agents (Intake → Context → Planner → Router → Specialists → Guardian → Verifier → Reporter → Learner) that completes tasks end to end, recovers from errors, and learns from every episode. Full design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **Web screen:** run the server, then open **http://localhost:8000/ui**. From there you can run tasks with live step progress, approve or decline held actions, give feedback, see what it has learned, and upload or ask about documents.
- `POST /assistant/task` submits a goal (`"wait": false` runs it in the background; poll `GET /assistant/task/{id}`). Approve escalated actions with `/assistant/task/{id}/approve`, and send 👍/👎 or corrections with `/assistant/task/{id}/feedback`.
- What it may do on its own is set **only** by [assistant/policy.yaml](assistant/policy.yaml).
- `python -m evals.run_evals` is the weekly self-improvement job: it proposes prompt changes and promotes them only if evals don't regress.
- Calendar, comms and tasks use local store-backed adapters; swap in real API clients to go live.
- Run the tests (offline, no keys needed): `python -m unittest discover -s tests -t .`

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
