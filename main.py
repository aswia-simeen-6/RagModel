import os
import shutil
import json
import re
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
import base64
from fastapi import Form
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
load_dotenv()  # Load environment variables from .env file

app = FastAPI()

# Enable CORS for Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to hold the vector store in memory (for demo purposes)
vector_store = None

# Shared judge LLM + embeddings for RAGAS evaluation (lazily initialized)
_judge_llm = None
_judge_embeddings = None

def get_judge():
    global _judge_llm, _judge_embeddings
    if _judge_llm is None:
        _judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        _judge_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return _judge_llm, _judge_embeddings

class QuestionRequest(BaseModel):
    question: str
    evaluate: Optional[bool] = False

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vector_store
    
    # 1. Save the file temporarily
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 2. Load and Extract Text
        loader = PyPDFLoader(file_location)
        documents = loader.load()
        
        # 3. Split Text into Chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # 4. Create or Update Vector Store
        # Create or Update Vector Store
        if vector_store is None:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            # Create a new Chroma database in memory
            vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
            
            item_count = len(vector_store.get()['ids'])
            print(f"🧠 NEW CHROMA MEMORY CREATED: {item_count} text chunks stored.")
        else:
            # Add new chunks to the existing Chroma database
            vector_store.add_documents(documents=chunks)
            item_count = len(vector_store.get()['ids'])
            print(f"🧠 CHROMA MEMORY EXPANDED: Database now holds {item_count} total chunks!")
        
        return {"status": "success", "message": "File processed and indexed successfully."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if os.path.exists(file_location):
            os.remove(file_location)
# Create a data model for the incoming URL
class UrlRequest(BaseModel):
    url: str
@app.post("/url")
async def process_url(request: UrlRequest):
    global vector_store
    
    try:
        # 1. Scrape the website
        loader = WebBaseLoader(request.url)
        documents = loader.load()
        
        # 2. Split Text into Chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # 3/4. Create or Update Vector Store
        if vector_store is None:
            # If no database exists yet, create a new one
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
            
            item_count = len(vector_store.get()['ids'])
            print(f"🧠 NEW CHROMA MEMORY CREATED: {item_count} text chunks stored.")
        else:
            # If a database already exists, just add the new chunks to it!
            vector_store.add_documents(documents=chunks)
        
        return {"status": "success", "message": "Website processed and indexed successfully."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing website: {str(e)}")
@app.post("/ask")
async def ask_question(question_request: QuestionRequest):
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=400, detail="Please upload a file first.")
    
    try:
        # 5. Setup RAG Chain
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
        # Get relevant documents
        retriever = vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 10, "fetch_k": 30}
        )
        docs = retriever.invoke(question_request.question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # First, get the answer
        answer_prompt = f"""You are an expert analyst.Based on the following context, answer the question concisely.

<CRITICAL_INSTRUCTIONS>
1. ZERO HALLUCINATION: You must not use any outside knowledge, whatsoever. 
2. THE ESCAPE HATCH: If the answer cannot be fully constructed from the Context, you must reply exactly: "I do not have enough information in the provided documents to answer that." Do not attempt to guess.
3. DIRECTNESS: Answer the user's question immediately and concisely. Do not add conversational filler (e.g., do not say "According to the provided text...").
4. RELEVANCY: Address the exact question asked. Do not provide tangential information.
</CRITICAL_INSTRUCTIONS>
Context: {context}

Question: {question_request.question}

Answer:"""
        
        answer_response = llm.invoke(answer_prompt)
        answer_text = answer_response.content
        
        # Check if the question asks for numerical data or distribution
        chart_keywords = ['distribution', 'breakdown', 'percentage', 'proportion', 'compare', 'comparison', 'how many', 'what percentage', 'show me', 'visualize', 'chart', 'graph', 'plot']
        should_extract_chart = any(keyword in question_request.question.lower() for keyword in chart_keywords)
        
        chart_data = None
        if should_extract_chart:
            # Try to extract chart data
            chart_prompt = f"""Based on the following context and question, extract numerical data suitable for a pie chart.
Return ONLY a JSON object with this exact format (no extra text):
{{
  "labels": ["Category1", "Category2", "Category3"],
  "values": [30, 45, 25]
}}

If no numerical distribution data is found, return: {{"labels": [], "values": []}}

Context: {context}

Question: {question_request.question}

JSON:"""
            
            try:
                chart_response = llm.invoke(chart_prompt)
                chart_text = chart_response.content.strip()
                
                # Extract JSON from response
                json_match = re.search(r'\{[^{}]*"labels"[^{}]*"values"[^{}]*\}', chart_text, re.DOTALL)
                if json_match:
                    chart_data = json.loads(json_match.group())
                    # Validate the data
                    if not chart_data.get('labels') or not chart_data.get('values'):
                        chart_data = None
                    elif len(chart_data['labels']) != len(chart_data['values']):
                        chart_data = None
            except Exception as e:
                print(f"Chart extraction error: {e}")
                chart_data = None
        
        ragas_scores = None
        if question_request.evaluate:
            try:
                ragas_scores = await asyncio.to_thread(
                    _run_ragas_reference_free,
                    question_request.question,
                    answer_text,
                    [doc.page_content for doc in docs]
                )
            except Exception as e:
                print(f"RAGAS evaluation error: {e}")
                ragas_scores = None

        return {
            "answer": answer_text,
            "sources": [doc.metadata.get('page', 0) for doc in docs],
            "chartData": chart_data,
            "scores": ragas_scores
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# ── RAGAS Helper ──────────────────────────────────────────────────────────────

def _run_ragas_reference_free(question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
    """Run reference-free RAGAS metrics in a blocking thread."""
    judge_llm, judge_embeddings = get_judge()
    ds = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    })
    result = evaluate(
        dataset=ds,
        metrics=[
            LLMContextPrecisionWithoutReference(),
            Faithfulness(),
            AnswerRelevancy(),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    df = result.to_pandas()
    return {
        "context_precision": round(float(df["llm_context_precision_without_reference"].iloc[0]), 4),
        "faithfulness": round(float(df["faithfulness"].iloc[0]), 4),
        "answer_relevancy": round(float(df["answer_relevancy"].iloc[0]), 4),
        "context_recall": None,
    }


def _run_ragas_with_reference(question: str, answer: str, contexts: List[str], ground_truth: str) -> Dict[str, Any]:
    """Run all RAGAS metrics (including recall which needs ground_truth) in a blocking thread."""
    judge_llm, judge_embeddings = get_judge()
    ds = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth],
    })
    result = evaluate(
        dataset=ds,
        metrics=[
            LLMContextPrecisionWithoutReference(),
            LLMContextRecall(),
            Faithfulness(),
            AnswerRelevancy(),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    df = result.to_pandas()
    return {
        "context_precision": round(float(df["llm_context_precision_without_reference"].iloc[0]), 4),
        "faithfulness": round(float(df["faithfulness"].iloc[0]), 4),
        "answer_relevancy": round(float(df["answer_relevancy"].iloc[0]), 4),
        "context_recall": round(float(df["context_recall"].iloc[0]), 4),
    }


# ── Evaluate Endpoint ─────────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    question: str
    ground_truth: Optional[str] = None


@app.post("/evaluate")
async def evaluate_question(request: EvaluateRequest):
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=400, detail="Please upload a file first.")

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 30}
        )
        docs = retriever.invoke(request.question)
        contexts = [doc.page_content for doc in docs]
        context = "\n\n".join(contexts)

        answer_prompt = f"""You are an expert analyst. Based on the following context, answer the question concisely.

<CRITICAL_INSTRUCTIONS>
1. ZERO HALLUCINATION: You must not use any outside knowledge, whatsoever.
2. THE ESCAPE HATCH: If the answer cannot be fully constructed from the Context, you must reply exactly: "I do not have enough information in the provided documents to answer that."
3. DIRECTNESS: Answer the user's question immediately and concisely.
4. RELEVANCY: Address the exact question asked.
</CRITICAL_INSTRUCTIONS>
Context: {context}

Question: {request.question}

Answer:"""

        answer_response = llm.invoke(answer_prompt)
        answer_text = answer_response.content

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

        return {
            "answer": answer_text,
            "scores": scores,
            "sources": [doc.metadata.get('page', 0) for doc in docs],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating question: {str(e)}")


# ── Batch Evaluate Endpoint ───────────────────────────────────────────────────

@app.post("/batch-evaluate")
async def batch_evaluate():
    eval_path = os.path.join(os.path.dirname(__file__), "eval_data.json")
    if not os.path.exists(eval_path):
        raise HTTPException(status_code=404, detail="eval_data.json not found.")

    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data_samples = json.load(f)

        def _run_batch():
            judge_llm, judge_embeddings = get_judge()
            ds = Dataset.from_dict(data_samples)
            result = evaluate(
                dataset=ds,
                metrics=[
                    LLMContextPrecisionWithoutReference(),
                    LLMContextRecall(),
                    Faithfulness(),
                    AnswerRelevancy(),
                ],
                llm=judge_llm,
                embeddings=judge_embeddings,
            )
            df = result.to_pandas()
            averages = {
                "context_precision": round(float(df["llm_context_precision_without_reference"].mean()), 4),
                "context_recall": round(float(df["context_recall"].mean()), 4),
                "faithfulness": round(float(df["faithfulness"].mean()), 4),
                "answer_relevancy": round(float(df["answer_relevancy"].mean()), 4),
            }
            details = []
            for _, row in df.iterrows():
                details.append({
                    "question": row["question"],
                    "answer": row["answer"],
                    "context_precision": round(float(row["llm_context_precision_without_reference"]), 4),
                    "context_recall": round(float(row["context_recall"]), 4),
                    "faithfulness": round(float(row["faithfulness"]), 4),
                    "answer_relevancy": round(float(row["answer_relevancy"]), 4),
                })
            return {"averages": averages, "details": details}

        result = await asyncio.to_thread(_run_batch)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch evaluation error: {str(e)}")


@app.post("/clear")
async def clear_memory():
    global vector_store
    
    # Delete the vector database from memory
    vector_store = None
    
    return {"status": "success", "message": "Memory cleared successfully."}
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...), question: str = Form(...)):
    try:
        # 1. Read the image and convert it to Base64 (so the AI can "see" it over the API)
        image_data = await file.read()
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        
        # 2. Set up the LLM (Gemini 2.5 Flash natively supports vision!)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
        # 3. Create a multimodal message (combining text and the image)
        message = HumanMessage(
            content=[
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:{file.content_type};base64,{image_b64}"}}
            ]
        )
        
        # 4. Ask Gemini!
        response = llm.invoke([message])
        
        # 5. Return the response matching our Angular interface
        return {
            "answer": response.content,
            "has_chart": False,
            "chart_labels": [],
            "chart_values": [],
            "sources": ["Uploaded Image"]
        }
        
    except Exception as e:
        print(f"Vision Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")
# Run with: uvicorn main:app --reload