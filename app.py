

from flask import Flask, render_template, request

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

app = Flask(__name__)

embeddings = download_hugging_face_embeddings()

docsearch = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = OllamaLLM(
    model="llama3"
)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():

    question = request.form["msg"]

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
{system_prompt}

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return str(response)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )