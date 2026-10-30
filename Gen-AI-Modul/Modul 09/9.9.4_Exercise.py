import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

load_dotenv()

# Setup Vector Store & Retriever
docs = [
    Document(page_content="RAG stands for Retrieval-Augmented Generation. It connects LLMs to external knowledge bases."),
    Document(page_content="The retrieval step uses embeddings and vector search to find relevant documents for a given query."),
    Document(page_content="The generation step takes the retrieved context and the user question to produce a grounded answer."),
]

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Python For GenAI",
    },
)

# 1. Chain Pembuat Jawaban RAG
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using ONLY the provided context.\n\nContext:\n{context}"),
    ("human", "{question}"),
])
answer_chain = rag_prompt | llm | StrOutputParser()

# 2. Chain Penilai Groundedness (Score 0-10)
class GroundednessEvaluation(BaseModel):
    score: int = Field(description="Groundedness score from 0 to 10 based on context")
    reason: str = Field(description="Brief explanation for the score")

eval_prompt = ChatPromptTemplate.from_messages([
    ("system", "Score the groundedness of the generated answer against the provided context on a scale of 0 to 10. 10 means fully grounded in context, 0 means completely fabricated or ungrounded."),
    ("human", "Context:\n{context}\n\nGenerated Answer:\n{answer}"),
])

eval_chain = eval_prompt | llm.with_structured_output(GroundednessEvaluation)

def rag_with_quality_filter(question: str):
    # Step A: Retrieve Context
    retrieved_docs = retriever.invoke(question)
    context_str = "\n\n".join(d.page_content for d in retrieved_docs)
    
    # Step B: Generate Answer
    answer = answer_chain.invoke({"context": context_str, "question": question})
    
    # Step C: Evaluate Groundedness
    evaluation = eval_chain.invoke({"context": context_str, "answer": answer})
    
    return {
        "answer": answer,
        "groundedness_score": evaluation.score,
        "reason": evaluation.reason
    }

# Pengujian Script
question = "What does RAG stand for and how does retrieval work?"
result = rag_with_quality_filter(question)

print(f"Q: {question}")
print(f"A: {result['answer']}")
print(f"\nGroundedness Score: {result['groundedness_score']}/10")
print(f"Alasan Evaluasi: {result['reason']}")