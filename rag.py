from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)

from config import *
from prompts import (
    CONTEXTUALIZE_Q_PROMPT,
    QA_PROMPT,
)


class RAGPipeline:

    def __init__(self):

        print("=" * 60)
        print("Initializing HCL AI Employee Assistant...")
        print("=" * 60)

        # --------------------------------------------------
        # Embedding Model
        # --------------------------------------------------

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        print("Embedding Model Loaded.")

        # --------------------------------------------------
        # Load FAISS
        # --------------------------------------------------

        self.vectorstore = FAISS.load_local(
            folder_path="vectorstore",
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

        print("Vector Database Loaded.")

        # --------------------------------------------------
        # Retriever
        # --------------------------------------------------

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": TOP_K
            }
        )

        print("Retriever Created.")

        # --------------------------------------------------
        # LLM
        # --------------------------------------------------

        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=MODEL_NAME
        )

        print("LLM Loaded.")

        # --------------------------------------------------
        # History Aware Retriever
        # --------------------------------------------------

        self.history_aware_retriever = create_history_aware_retriever(

            llm=self.llm,

            retriever=self.retriever,

            prompt=CONTEXTUALIZE_Q_PROMPT

        )

        print("History Aware Retriever Ready.")

        # --------------------------------------------------
        # QA Chain
        # --------------------------------------------------

        self.document_chain = create_stuff_documents_chain(

            llm=self.llm,

            prompt=QA_PROMPT

        )

        print("Document Chain Created.")

        # --------------------------------------------------
        # Final Retrieval Chain
        # --------------------------------------------------

        self.chain = create_retrieval_chain(

            self.history_aware_retriever,

            self.document_chain

        )

        print("Conversational RAG Ready.")
        print("=" * 60)

    # ======================================================
    # Ask Question
    # ======================================================

    def ask(self, question, chat_history):

        response = self.chain.invoke(

            {

                "input": question,

                "chat_history": chat_history

            }

        )

        sources = []

        seen = set()

        for doc in response["context"]:

            key = (
                doc.metadata.get("source"),
                doc.metadata.get("page")
            )

            if key in seen:
                continue

            seen.add(key)

            sources.append(

                {

                    "source": doc.metadata.get(
                        "source",
                        "Unknown"
                    ),

                    "page": doc.metadata.get(
                        "page",
                        "N/A"
                    ),

                    "department": doc.metadata.get(
                        "department",
                        "Unknown"
                    ),

                    "category": doc.metadata.get(
                        "category",
                        "Unknown"
                    ),

                    "version": doc.metadata.get(
                        "version",
                        "Unknown"
                    )

                }

            )

        return {

            "answer": response["answer"],

            "sources": sources

        }