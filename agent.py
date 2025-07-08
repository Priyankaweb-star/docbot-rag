# agent.py

from tools import google_tool
from transformers import pipeline
import os

def answer_from_pdf(retriever, query):
    docs = retriever.get_relevant_documents(query)

    if not docs:
        return {"result": "I don't know", "source_documents": []}

    context = "\n\n".join([doc.page_content for doc in docs])

    # ✅ Use proper QA model
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

    result = qa_pipeline({
        "context": context,
        "question": query
    })

    return {
        "result": result.get("answer", "I don't know"),
        "source_documents": docs
    }


def answer_from_google(query):
    """Fallback using Serper-based search"""
    return google_tool.run(query)

def is_weak_answer(answer):
    """Define when to fallback to Google"""
    if not answer:
        return True
    if len(answer.strip()) < 40:
        return True
    # Add more rules if needed (e.g., check for "I don't know")
    return False
