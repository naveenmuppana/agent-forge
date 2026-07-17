# My AI Agent Platform

A modular AI agent framework, redesigned and personal to the user. This platform provides foundational capabilities for chat, tool-based agents, and Retrieval-Augmented Generation (RAG) backed by MongoDB Atlas.

## Structure
- `agent_service.py`: FastAPI application providing endpoints for chat, weather agents, and RAG.
- `chat_client.py`: Standalone interactive chat interface.
- `weather_agent.py`: CLI agent demonstrating tool-calling (Weather API).
- `core.py`: Central utility module for configuration, OpenAI/MongoDB clients, and shared logic.

## Functionality
- **Chat**: Direct LLM interaction.
- **Tools**: Automated weather forecasting via Open-Meteo.
- **RAG**: Document ingestion and semantic search using MongoDB Atlas Vector Search.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Configure `.env` file.
3. Start the service: `uvicorn agent_service:app --reload`
# agent-forge
