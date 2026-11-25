# Author: Naveen Muppana
import logging
import os
import csv
from io import BytesIO
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI

from chat_client import get_basic_llm_response
from weather_agent import run_agent as weather_agent_run
from core import (
    OPENAI_MODEL,
    get_embeddings,
    get_mongo_collection,
    get_openai_key,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(title="LLM + RAG API Server")
VECTOR_INDEX_NAME = os.getenv("MONGODB_VECTOR_INDEX", "vector_index")


# -------------------- SHARED CONFIG --------------------

_rag_instruction = (
    "You are a helpful assistant.\n"
    "Use only the provided context to answer the user's question.\n"
    "Write in a clear, natural, human tone.\n"
    "If the answer is not present in the context, say exactly: \"I don't know.\""
)


def _extract_text_from_uploaded_file(file: UploadFile, file_bytes: bytes) -> str:
    filename = (file.filename or "").lower()

    if filename.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            logger.exception("pypdf is required for PDF upload parsing.")
            raise HTTPException(
                status_code=500,
                detail="PDF upload requires 'pypdf'. Install dependencies from requirements.txt.",
            ) from exc

        try:
            reader = PdfReader(BytesIO(file_bytes))
            page_texts = [page.extract_text() or "" for page in reader.pages]
            extracted = "\n\n".join(page_texts).strip()
            logger.info("Extracted text from PDF '%s' (%s pages).", file.filename, len(reader.pages))
            return extracted
        except Exception as exc:
            logger.exception("Failed to parse PDF '%s'.", file.filename)
            raise HTTPException(
                status_code=400,
                detail=f"Unable to parse PDF '{file.filename}'. Upload a readable text PDF.",
            ) from exc

    if filename.endswith(".docx") or filename.endswith(".doc"):
        try:
            from docx import Document
        except ImportError as exc:
            logger.exception("python-docx is required for Word upload parsing.")
            raise HTTPException(
                status_code=500,
                detail="Word upload requires 'python-docx'. Install dependencies from requirements.txt.",
            ) from exc

        try:
            doc = Document(BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
            extracted = "\n".join(paragraphs).strip()
            logger.info("Extracted text from Word file '%s'.", file.filename)
            if not extracted and filename.endswith(".doc"):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Could not extract text from '{file.filename}'. "
                        "Legacy .doc files may not be supported; convert to .docx."
                    ),
                )
            return extracted
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Failed to parse Word file '%s'.", file.filename)
            raise HTTPException(
                status_code=400,
                detail=f"Unable to parse Word file '{file.filename}'. Upload a readable .docx file.",
            ) from exc

    if filename.endswith(".csv"):
        try:
            decoded = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            logger.exception("Failed decoding CSV file '%s'.", file.filename)
            raise HTTPException(
                status_code=400,
                detail=f"Unable to decode CSV '{file.filename}'. Use UTF-8 encoded CSV.",
            ) from exc

        rows = []
        reader = csv.reader(decoded.splitlines())
        for row in reader:
            row_text = ", ".join(cell.strip() for cell in row if cell and cell.strip())
            if row_text:
                rows.append(row_text)
        extracted = "\n".join(rows).strip()
        logger.info("Extracted %s non-empty CSV rows from '%s'.", len(rows), file.filename)
        return extracted

    # Default: plain-text-like file decoding.
    try:
        return file_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        logger.exception("Failed UTF-8 decoding for file '%s'.", file.filename)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format for '{file.filename}'. "
                "Upload .txt/.md/.json/.csv or .pdf files."
            ),
        )


# -------------------- REQUEST / RESPONSE SCHEMAS --------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    instruction: Optional[str] = None


class ChatResponse(BaseModel):
    response: str


class RagInstructionRequest(BaseModel):
    instruction: str = Field(..., min_length=1)


class RagInstructionResponse(BaseModel):
    instruction: str


class RagUploadResponse(BaseModel):
    stored_chunks: int
    sources: List[str]


class RagQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=3, ge=1, le=10)


class ChunkResult(BaseModel):
    score: float
    text: str
    source: str


class RagQueryResponse(BaseModel):
    answer: str
    instruction: str
    top_chunks: List[ChunkResult]


# -------------------- BASIC LLM API --------------------

@app.post("/chat/direct", response_model=ChatResponse)
def chat_direct(req: ChatRequest):
    logger.info("Received /chat/direct request.")
    instruction = req.instruction or "You are a helpful assistant. Answer clearly and briefly."
    try:
        result = get_basic_llm_response(req.message, instruction)
    except ValueError as exc:
        logger.exception("Direct chat failed due to configuration error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info("Returning /chat/direct response.")
    return ChatResponse(response=result)


# -------------------- WEATHER TOOL API --------------------

@app.post("/chat/weather", response_model=ChatResponse)
def chat_weather(req: ChatRequest):
    logger.info("Received /chat/weather request.")
    try:
        result = weather_agent_run(req.message)
    except Exception as exc:
        logger.exception("Weather chat failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Weather agent failed: {exc}") from exc

    logger.info("Returning /chat/weather response.")
    return ChatResponse(response=result)


# -------------------- RAG INSTRUCTION API --------------------

@app.post("/rag/instruction", response_model=RagInstructionResponse)
def set_rag_instruction(req: RagInstructionRequest):
    global _rag_instruction
    _rag_instruction = req.instruction.strip()
    logger.info("RAG instruction updated.")
    return RagInstructionResponse(instruction=_rag_instruction)


# -------------------- RAG UPLOAD DOCS API --------------------

@app.post("/rag/upload", response_model=RagUploadResponse)
async def upload_rag_docs(
    file: UploadFile = File(...),
):
    # Receive one uploaded file and store its chunks in MongoDB Atlas Vector Search.
    logger.info("Received /rag/upload request.")
    try:
        # Build vector store client using configured MongoDB collection + embedding model.
        collection = get_mongo_collection()
        embeddings = get_embeddings()
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name=VECTOR_INDEX_NAME,
            relevance_score_fn="cosine",
        )
    except ValueError as exc:
        logger.exception("RAG upload failed during setup: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info("Processing uploaded file: %s", file.filename)
    file_bytes = await file.read()
    if not file_bytes:
        logger.warning("Uploaded file is empty: %s", file.filename)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Extract clean text based on file type (.txt/.pdf/.docx/.csv, etc.).
    content = _extract_text_from_uploaded_file(file, file_bytes)
    if not content:
        logger.warning("Uploaded file contains no usable text: %s", file.filename)
        raise HTTPException(status_code=400, detail="Uploaded file contains no usable text.")

    # Split large text into overlapping chunks for better retrieval quality.
    source = file.filename or "uploaded_file.txt"
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(content)
    if not chunks:
        logger.warning("No chunks generated for source: %s", source)
        raise HTTPException(status_code=400, detail="No valid text chunks found to store.")

    # Store chunks with source metadata; embeddings are generated automatically.
    metadatas = [{"source": source} for _ in chunks]
    vector_store.add_texts(texts=chunks, metadatas=metadatas)
    stored_chunks = len(chunks)

    logger.info("Inserted %s chunks into MongoDB from source %s.", stored_chunks, source)
    return RagUploadResponse(stored_chunks=stored_chunks, sources=[source])


# -------------------- RAG QUERY API --------------------

@app.post("/rag/query", response_model=RagQueryResponse)
def query_rag(req: RagQueryRequest):
    # Retrieve top-k chunks from MongoDB vector index, then answer using LLM + context.
    logger.info("Received /rag/query request with k=%s.", req.k)
    try:
        # Initialize vector store for similarity search.
        collection = get_mongo_collection()
        embeddings = get_embeddings()
        openai_key = get_openai_key()
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name=VECTOR_INDEX_NAME,
            relevance_score_fn="cosine",
        )
    except ValueError as exc:
        logger.exception("RAG query failed during setup: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        # MongoDB Atlas Vector Search returns (Document, score) pairs.
        top_docs_with_scores = vector_store.similarity_search_with_score(req.query, k=req.k)
    except Exception as exc:
        logger.exception("MongoDB vector search failed. Check index '%s': %s", VECTOR_INDEX_NAME, exc)
        raise HTTPException(
            status_code=500,
            detail=(
                f"MongoDB vector search failed. Ensure Atlas vector index "
                f"'{VECTOR_INDEX_NAME}' exists on path 'embedding'."
            ),
        ) from exc

    if not top_docs_with_scores:
        logger.warning("RAG query returned no vector search results.")
        raise HTTPException(status_code=404, detail="No relevant documents found. Upload docs first.")

    # Build context from retrieved chunks and ask the LLM to answer from this context only.
    context = "\n\n".join(doc.page_content for doc, _ in top_docs_with_scores)

    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=openai_key)
    prompt = f"""
{_rag_instruction}

Context:
{context}

Question:
{req.query}
""".strip()
    answer = llm.invoke(prompt).content
    logger.info("RAG query answered successfully using %s chunks.", len(top_docs_with_scores))

    # Return answer plus supporting chunks and similarity scores for transparency.
    response_chunks = [
        ChunkResult(
            score=round(score, 4),
            text=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
        )
        for doc, score in top_docs_with_scores
    ]

    return RagQueryResponse(
        answer=answer,
        instruction=_rag_instruction,
        top_chunks=response_chunks,
    )


# -------------------- MULTI-AGENT ASSISTANT API --------------------
# Pipeline and design: docs/ARCHITECTURE.md. The Assistant is created on first use
# (it needs OPENAI_API_KEY) and resumes any task interrupted by a restart.

_assistant_instance = None


def _assistant():
    global _assistant_instance
    if _assistant_instance is None:
        from assistant.orchestrator import Assistant

        try:
            _assistant_instance = Assistant()
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        resumed = _assistant_instance.resume_running()
        if resumed:
            logger.info("Resumed %s interrupted assistant task(s).", len(resumed))
    return _assistant_instance


class TaskRequest(BaseModel):
    message: str = Field(..., min_length=1)
    source: str = Field(default="chat", pattern="^(chat|email|cron|webhook)$")
    user_id: str = "default"


class ApproveRequest(BaseModel):
    approve: List[str] = Field(default_factory=list)
    reject: List[str] = Field(default_factory=list)


class ClarifyRequest(BaseModel):
    answer: str = Field(..., min_length=1)


class FeedbackRequest(BaseModel):
    rating: Optional[str] = Field(default=None, pattern="^(up|down)$")
    correction: str = ""
    preferences: dict = Field(default_factory=dict)
    undo: bool = False


def _task_view(state) -> dict:
    return {
        "task_id": state.task_id,
        "status": state.status.value,
        "report": state.report,
        "pending_approvals": [
            {"action_id": a.action_id, "kind": a.kind, "summary": a.summary, "params": a.params}
            for a in state.pending_approvals
        ],
        "steps": [
            {"step_id": s.step_id, "capability": s.capability, "instruction": s.instruction,
             "status": state.results[s.step_id].status.value if s.step_id in state.results else "pending"}
            for s in (state.plan.steps if state.plan else [])
        ],
        "replans": state.replans,
        "llm_calls": state.llm_calls,
        "cost_usd": round(state.cost_usd, 4),
        "errors": state.errors,
    }


def _load_or_404(task_id: str):
    state = _assistant().get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"No task {task_id}")
    return state


@app.post("/assistant/task")
def assistant_submit(req: TaskRequest):
    logger.info("Received /assistant/task from source=%s.", req.source)
    return _task_view(_assistant().submit(req.message, source=req.source, user_id=req.user_id))


@app.get("/assistant/task/{task_id}")
def assistant_get(task_id: str):
    return _task_view(_load_or_404(task_id))


@app.post("/assistant/task/{task_id}/approve")
def assistant_approve(task_id: str, req: ApproveRequest):
    _load_or_404(task_id)
    return _task_view(_assistant().approve(task_id, approve=req.approve, reject=req.reject))


@app.post("/assistant/task/{task_id}/clarify")
def assistant_clarify(task_id: str, req: ClarifyRequest):
    _load_or_404(task_id)
    return _task_view(_assistant().clarify(task_id, req.answer))


@app.post("/assistant/task/{task_id}/feedback")
def assistant_feedback(task_id: str, req: FeedbackRequest):
    _load_or_404(task_id)
    learned = _assistant().feedback(task_id, rating=req.rating, correction=req.correction,
                                    preferences=req.preferences, undo=req.undo)
    return {"learned": learned}


@app.get("/assistant/learning")
def assistant_learning(user_id: str = "default"):
    """What the assistant has learned (lessons, playbooks, prompt candidates)."""
    return _assistant().learner.digest(user_id)


@app.delete("/assistant/lessons/{lesson_id}")
def assistant_forget_lesson(lesson_id: str):
    _assistant().store.delete("lessons", lesson_id)
    return {"deleted": lesson_id}


@app.get("/")
def health():
    logger.info("Health check requested.")
    return {"status": "ok", "message": "LLM + RAG API is running"}
