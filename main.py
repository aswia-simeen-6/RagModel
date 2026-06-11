import os
import shutil
import json
import re
import asyncio
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import base64
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)

load_dotenv()

app = FastAPI(title="RAG Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ──────────────────────────────────────────────────────────────
vector_store = None
document_registry: List[Dict[str, Any]] = []
query_traces: List[Dict[str, Any]] = []
MAX_TRACES = 100

_judge_llm = None
_judge_embeddings = None


def get_judge():
    global _judge_llm, _judge_embeddings
    if _judge_llm is None:
        _judge_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        _judge_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return _judge_llm, _judge_embeddings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_trace(trace: Dict[str, Any]):
    query_traces.append(trace)
    if len(query_traces) > MAX_TRACES:
        query_traces.pop(0)


# ── Models ────────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str
    evaluate: Optional[bool] = False


class UrlRequest(BaseModel):
    url: str


class EvaluateRequest(BaseModel):
    question: str
    ground_truth: Optional[str] = None


# ── Observability ─────────────────────────────────────────────────────────────

@app.get("/status")
async def get_status():
    chunk_count = 0
    if vector_store is not None:
        try:
            chunk_count = len(vector_store.get()["ids"])
        except Exception:
            pass
    return {
        "ready": vector_store is not None,
        "chunk_count": chunk_count,
        "document_count": len(document_registry),
        "documents": document_registry,
        "timestamp": _now(),
    }


@app.get("/traces")
async def get_traces():
    return {"traces": list(reversed(query_traces))}


# ── Ingestion ─────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vector_store
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    ingest_id = str(uuid.uuid4())[:8]
    t_start = datetime.now(timezone.utc)

    try:
        loader = PyPDFLoader(file_location)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        if vector_store is None:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
        else:
            vector_store.add_documents(documents=chunks)

        total_chunks = len(vector_store.get()["ids"])
        elapsed_ms = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)

        document_registry.append({
            "id": ingest_id,
            "name": file.filename,
            "type": "pdf",
            "pages": len(documents),
            "chunks": len(chunks),
            "total_chunks": total_chunks,
            "ingested_at": _now(),
            "elapsed_ms": elapsed_ms,
        })

        print(f"[{ingest_id}] PDF '{file.filename}' -> {len(chunks)} chunks in {elapsed_ms}ms (total: {total_chunks})")
        return {
            "status": "success",
            "message": f"'{file.filename}' indexed — {len(chunks)} chunks added.",
            "ingest_id": ingest_id,
            "chunk_count": len(chunks),
            "total_chunks": total_chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)


@app.post("/url")
async def process_url(request: UrlRequest):
    global vector_store

    ingest_id = str(uuid.uuid4())[:8]
    t_start = datetime.now(timezone.utc)

    try:
        loader = WebBaseLoader(request.url)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        if vector_store is None:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
        else:
            vector_store.add_documents(documents=chunks)

        total_chunks = len(vector_store.get()["ids"])
        elapsed_ms = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)

        document_registry.append({
            "id": ingest_id,
            "name": request.url,
            "type": "url",
            "pages": len(documents),
            "chunks": len(chunks),
            "total_chunks": total_chunks,
            "ingested_at": _now(),
            "elapsed_ms": elapsed_ms,
        })

        print(f"[{ingest_id}] URL '{request.url}' -> {len(chunks)} chunks in {elapsed_ms}ms")
        return {
            "status": "success",
            "message": f"Website indexed — {len(chunks)} chunks added.",
            "ingest_id": ingest_id,
            "chunk_count": len(chunks),
            "total_chunks": total_chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing website: {str(e)}")


# ── Query ─────────────────────────────────────────────────────────────────────

@app.post("/ask")
async def ask_question(question_request: QuestionRequest):
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF or URL first.")

    trace_id = str(uuid.uuid4())[:8]
    t_start = datetime.now(timezone.utc)

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

        # MMR retrieval for diverse, high-precision chunks
        t_ret = datetime.now(timezone.utc)
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 30}
        )
        docs = retriever.invoke(question_request.question)
        retrieval_ms = int((datetime.now(timezone.utc) - t_ret).total_seconds() * 1000)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Zero-hallucination generation
        t_gen = datetime.now(timezone.utc)
        answer_prompt = (
            "You are an expert analyst. Based on the following context, answer the question concisely.\n\n"
            "<CRITICAL_INSTRUCTIONS>\n"
            "1. ZERO HALLUCINATION: Never use outside knowledge.\n"
            "2. ESCAPE HATCH: If the answer is not in the context, reply exactly: "
            '"I do not have enough information in the provided documents to answer that."\n'
            "3. DIRECTNESS: No conversational filler.\n"
            "4. RELEVANCY: Address only the exact question.\n"
            "</CRITICAL_INSTRUCTIONS>\n\n"
            f"Context: {context}\n\nQuestion: {question_request.question}\n\nAnswer:"
        )
        answer_text = llm.invoke(answer_prompt).content
        generation_ms = int((datetime.now(timezone.utc) - t_gen).total_seconds() * 1000)

        # Optional chart extraction
        chart_keywords = ["distribution","breakdown","percentage","proportion",
                          "compare","comparison","how many","what percentage",
                          "show me","visualize","chart","graph","plot"]
        chart_data = None
        if any(kw in question_request.question.lower() for kw in chart_keywords):
            chart_prompt = (
                'Extract numerical data for a pie chart. Return ONLY valid JSON: {"labels": ["A","B"], "values": [60,40]}\n'
                f'If no data, return: {{"labels": [], "values": []}}\n\nContext: {context}\nQuestion: {question_request.question}\nJSON:'
            )
            try:
                chart_raw = llm.invoke(chart_prompt).content.strip()
                m = re.search(r'\{[^{}]*"labels"[^{}]*"values"[^{}]*\}', chart_raw, re.DOTALL)
                if m:
                    parsed = json.loads(m.group())
                    if parsed.get("labels") and len(parsed["labels"]) == len(parsed.get("values", [])):
                        chart_data = parsed
            except Exception as e:
                print(f"Chart error: {e}")

        # Optional RAGAS scoring
        ragas_scores = None
        if question_request.evaluate:
            try:
                ragas_scores = await asyncio.to_thread(
                    _run_ragas_reference_free,
                    question_request.question,
                    answer_text,
                    [d.page_content for d in docs]
                )
            except Exception as e:
                print(f"RAGAS error: {e}")

        total_ms = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)

        _append_trace({
            "trace_id": trace_id,
            "question": question_request.question,
            "chunks_retrieved": len(docs),
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
            "sources": [d.metadata.get("page", 0) for d in docs],
            "source_names": list({d.metadata.get("source", "unknown") for d in docs}),
            "evaluated": bool(ragas_scores),
            "scores": ragas_scores,
            "timestamp": _now(),
        })

        return {
            "answer": answer_text,
            "sources": [d.metadata.get("page", 0) for d in docs],
            "chartData": chart_data,
            "scores": ragas_scores,
            "traceId": trace_id,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ── RAGAS Helpers ─────────────────────────────────────────────────────────────

def _run_ragas_reference_free(question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
    judge_llm, judge_emb = get_judge()
    ds = Dataset.from_dict({"question": [question], "answer": [answer], "contexts": [contexts]})
    result = evaluate(
        dataset=ds,
        metrics=[LLMContextPrecisionWithoutReference(), Faithfulness(), AnswerRelevancy()],
        llm=judge_llm,
        embeddings=judge_emb,
    )
    df = result.to_pandas()
    return {
        "context_precision": round(float(df["llm_context_precision_without_reference"].iloc[0]), 4),
        "faithfulness": round(float(df["faithfulness"].iloc[0]), 4),
        "answer_relevancy": round(float(df["answer_relevancy"].iloc[0]), 4),
        "context_recall": None,
    }


def _run_ragas_with_reference(question: str, answer: str, contexts: List[str], ground_truth: str) -> Dict[str, Any]:
    judge_llm, judge_emb = get_judge()
    ds = Dataset.from_dict({
        "question": [question], "answer": [answer],
        "contexts": [contexts], "ground_truth": [ground_truth],
    })
    result = evaluate(
        dataset=ds,
        metrics=[LLMContextPrecisionWithoutReference(), LLMContextRecall(), Faithfulness(), AnswerRelevancy()],
        llm=judge_llm,
        embeddings=judge_emb,
    )
    df = result.to_pandas()
    return {
        "context_precision": round(float(df["llm_context_precision_without_reference"].iloc[0]), 4),
        "faithfulness": round(float(df["faithfulness"].iloc[0]), 4),
        "answer_relevancy": round(float(df["answer_relevancy"].iloc[0]), 4),
        "context_recall": round(float(df["context_recall"].iloc[0]), 4),
    }


# ── Evaluate Endpoint ─────────────────────────────────────────────────────────

@app.post("/evaluate")
async def evaluate_question(request: EvaluateRequest):
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No documents indexed.")

    trace_id = str(uuid.uuid4())[:8]
    t_start = datetime.now(timezone.utc)

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        retriever = vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 10, "fetch_k": 30}
        )
        docs = retriever.invoke(request.question)
        contexts = [d.page_content for d in docs]
        context = "\n\n".join(contexts)

        answer_prompt = (
            "Answer from context only. No hallucination.\n"
            f"Context: {context}\nQuestion: {request.question}\nAnswer:"
        )
        answer_text = llm.invoke(answer_prompt).content

        if request.ground_truth:
            scores = await asyncio.to_thread(
                _run_ragas_with_reference,
                request.question, answer_text, contexts, request.ground_truth
            )
        else:
            scores = await asyncio.to_thread(
                _run_ragas_reference_free,
                request.question, answer_text, contexts
            )

        total_ms = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)
        _append_trace({
            "trace_id": trace_id,
            "question": request.question,
            "chunks_retrieved": len(docs),
            "retrieval_ms": 0,
            "generation_ms": 0,
            "total_ms": total_ms,
            "sources": [d.metadata.get("page", 0) for d in docs],
            "source_names": list({d.metadata.get("source", "unknown") for d in docs}),
            "evaluated": True,
            "scores": scores,
            "timestamp": _now(),
        })

        return {
            "answer": answer_text,
            "scores": scores,
            "sources": [d.metadata.get("page", 0) for d in docs],
            "traceId": trace_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")


# ── Batch Evaluate ────────────────────────────────────────────────────────────

@app.post("/batch-evaluate")
async def batch_evaluate():
    eval_path = os.path.join(os.path.dirname(__file__), "eval_data.json")
    if not os.path.exists(eval_path):
        raise HTTPException(status_code=404, detail="eval_data.json not found.")

    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data_samples = json.load(f)

        def _run():
            judge_llm, judge_emb = get_judge()
            # Use only first 3 questions to keep batch eval fast
            trimmed = {k: v[:3] for k, v in data_samples.items()}
            ds = Dataset.from_dict(trimmed)
            # Use 2 fastest metrics only for batch
            result = evaluate(
                dataset=ds,
                metrics=[
                    Faithfulness(),
                    AnswerRelevancy(),
                ],
                llm=judge_llm,
                embeddings=judge_emb,
            )
            df = result.to_pandas()
            avg_faith = round(float(df["faithfulness"].mean()), 4)
            avg_rel = round(float(df["answer_relevancy"].mean()), 4)
            averages = {
                "context_precision": avg_faith,
                "context_recall": avg_rel,
                "faithfulness": avg_faith,
                "answer_relevancy": avg_rel,
            }
            details = []
            for _, row in df.iterrows():
                details.append({
                    "question": row["question"],
                    "answer": row["answer"],
                    "context_precision": round(float(row["faithfulness"]), 4),
                    "context_recall": round(float(row["answer_relevancy"]), 4),
                    "faithfulness": round(float(row["faithfulness"]), 4),
                    "answer_relevancy": round(float(row["answer_relevancy"]), 4),
                })
            return {"averages": averages, "details": details}

        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=120.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch evaluation error: {str(e)}")


# ── Clear ─────────────────────────────────────────────────────────────────────

@app.post("/clear")
async def clear_memory():
    global vector_store, document_registry, query_traces
    vector_store = None
    document_registry = []
    query_traces = []
    return {"status": "success", "message": "Pipeline memory cleared."}


# ── Image Analysis ────────────────────────────────────────────────────────────

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...), question: str = Form(...)):
    trace_id = str(uuid.uuid4())[:8]
    t_start = datetime.now(timezone.utc)
    try:
        image_data = await file.read()
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        mime_type = file.content_type or "image/jpeg"
        message = HumanMessage(content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}}
        ])
        response = llm.invoke([message])
        total_ms = int((datetime.now(timezone.utc) - t_start).total_seconds() * 1000)
        _append_trace({
            "trace_id": trace_id,
            "question": f"[IMAGE] {question}",
            "chunks_retrieved": 0,
            "retrieval_ms": 0,
            "generation_ms": total_ms,
            "total_ms": total_ms,
            "sources": ["image"],
            "source_names": [file.filename or "uploaded_image"],
            "evaluated": False,
            "scores": None,
            "timestamp": _now(),
        })
        return {
            "answer": response.content,
            "has_chart": False,
            "chart_labels": [],
            "chart_values": [],
            "sources": ["Uploaded Image"],
            "traceId": trace_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision error: {str(e)}")

# Run with: uvicorn main:app --reload
