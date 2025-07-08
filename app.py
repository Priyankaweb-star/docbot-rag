# app.py

import streamlit as st
from utils import get_retriever_from_pdf
from agent import answer_from_pdf, is_weak_answer, answer_from_google
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="📄 Ask your PDF with Google Fallback", layout="wide")
st.title("📘 Ask your PDF + 🌍 Google (Smart Fallback)")

pdf_file = st.file_uploader("📂 Upload PDF file", type=["pdf"])
query = st.text_input("🔍 Enter your question")
source_choice = st.radio("🧭 Select source for answer", ["PDF", "Google"])

if query and source_choice == "PDF" and pdf_file:
    with st.spinner("📚 Reading PDF..."):
        with open("uploaded.pdf", "wb") as f:
            f.write(pdf_file.read())

        retriever = get_retriever_from_pdf("uploaded.pdf")
        result = answer_from_pdf(retriever, query)
        answer = result.get("result", "").strip()
        sources = result.get("source_documents", [])

    st.markdown("### 📄 Answer from PDF:")
    st.success(answer or "❌ No answer returned.")

    if sources:
        with st.expander("📘 Source Chunks"):
            for i, doc in enumerate(sources, 1):
                st.markdown(f"**{i}.** {doc.page_content[:300]}...")

    if is_weak_answer(answer):
        st.warning("⚠️ This answer seems vague or incomplete.")
        if st.button("🔄 Try with Google instead"):
            with st.spinner("🌐 Searching Google..."):
                g_answer = answer_from_google(query)
            st.markdown("### 🔍 Google Result")
            st.info(g_answer)

elif query and source_choice == "Google":
    with st.spinner("🌐 Searching Google..."):
        g_answer = answer_from_google(query)
    st.markdown("### 🔍 Google Result")
    st.info(g_answer)

elif query and source_choice == "PDF" and not pdf_file:
    st.warning("📁 Please upload a PDF first.")

st.markdown("---")
st.caption("Built with ❤️ using LangChain, HuggingFace, Serper.dev, and Streamlit")
