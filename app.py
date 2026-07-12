import streamlit as st

from rag import RAGPipeline
from memory import ConversationMemory

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="HCL AI Employee Assistant",
    page_icon="🤖",
    layout="wide"
)

# ==========================================================
# Load RAG Pipeline
# ==========================================================

@st.cache_resource
def load_rag():

    return RAGPipeline()


rag = load_rag()

# ==========================================================
# Session State
# ==========================================================

if "messages" not in st.session_state:

    st.session_state.messages = [

        {
            "role": "assistant",
            "content":
            """
👋 Hello!

I am your **HCL AI Employee Assistant**.

I can help you with:

• Employee Handbook

• HR Policies

• Leave Policy

• Security Guidelines

• Travel Policy

• Coding Standards

Ask me anything.
"""
        }

    ]


if "memory" not in st.session_state:

    st.session_state.memory = ConversationMemory()

# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:

    st.title("🏢 HCL AI Assistant")

    st.markdown("---")

    st.subheader("📚 Knowledge Base")

    st.success("Employee Handbook")

    st.success("HR Policy")

    st.success("Leave Policy")

    st.success("Security Guidelines")

    st.success("Travel Policy")

    st.success("Coding Standards")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = [

            {
                "role": "assistant",
                "content":
                """
👋 Hello!

I am your **HCL AI Employee Assistant**.

How can I help you today?
"""
            }

        ]

        st.session_state.memory.clear()

        st.rerun()

# ==========================================================
# Title
# ==========================================================

st.title("🤖 HCL AI Employee Assistant")

st.caption(
    "Enterprise Conversational RAG using LangChain + FAISS + Groq"
)

st.markdown("---")

# ==========================================================
# Show Previous Messages
# ==========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ==========================================================
# Chat Input
# ==========================================================

question = st.chat_input(
    "Ask a question..."
)

if question:

    # -----------------------------------
    # Show User Message
    # -----------------------------------

    st.session_state.messages.append(

        {
            "role": "user",
            "content": question
        }

    )

    st.session_state.memory.add_user_message(
        question
    )

    with st.chat_message("user"):

        st.markdown(question)

    # -----------------------------------
    # Ask RAG
    # -----------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = rag.ask(

                question,

                st.session_state.memory.get_history()

            )

            answer = response["answer"]

            sources = response["sources"]

            st.markdown(answer)

            # -----------------------------------
            # Sources
            # -----------------------------------

            if sources:

                st.markdown("---")

                with st.expander("📄 Source Documents"):

                    shown = set()

                    for source in sources:

                        key = (

                            source["source"],

                            source["page"]

                        )

                        if key in shown:
                            continue

                        shown.add(key)

                        st.markdown(
f"""
### 📄 {source['source']}

- **Page :** {source['page'] + 1}

- **Department :** {source['department']}

- **Category :** {source['category']}

- **Version :** {source['version']}

---
"""
                        )

    # -----------------------------------
    # Save AI Response
    # -----------------------------------

    st.session_state.memory.add_ai_message(
        answer
    )

    st.session_state.messages.append(

        {

            "role": "assistant",

            "content": answer

        }

    )