import os
import json
from dotenv import load_dotenv
from datasets import Dataset
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
load_dotenv()  # Load environment variables from .env file
# 1. Setup the "Judge" LLM and Embeddings using Gemini
# We use temperature=0 so the judge is highly consistent and doesn't hallucinate its grades
judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
judge_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 2. Load the Test Data from our JSON text file
print("📂 Loading test data from eval_data.json...")
try:
    with open('eval_data.json', 'r', encoding='utf-8') as file:
        data_samples = json.load(file)
except FileNotFoundError:
    print("❌ Error: Could not find 'eval_data.json'. Make sure it is in the same folder!")
    exit()

# 3. Convert the dictionary into a HuggingFace Dataset (required by Ragas)
dataset = Dataset.from_dict(data_samples)

# 4. Run the Evaluation
print("🧠 Initializing Ragas Evaluation Judge...")
print("⏳ Grading your RAG pipeline. This may take a minute...\n")
results = evaluate(
    dataset=dataset,
    metrics=[
        LLMContextPrecisionWithoutReference(), # Did ChromaDB rank the best chunks at the top?
        LLMContextRecall(),                    # Did ChromaDB find all the necessary info to answer the question?
        Faithfulness(),                        # Did the LLM make anything up? (Hallucination check)
        AnswerRelevancy()                      # Did the LLM actually answer the user's specific question?
    ],
    llm=judge_llm,
    embeddings=judge_embeddings
)

# 5. Print the Scorecard
print("✅ EVALUATION COMPLETE! Here is your scorecard:")
print("-" * 40)
print(results)