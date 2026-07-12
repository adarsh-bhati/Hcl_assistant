from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

# ==========================================================
# Prompt 1
# Convert Follow-up Question → Standalone Question
# ==========================================================

CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages(

    [

        (
            "system",

            """
Given the chat history and the latest user question,
rewrite the question so that it becomes a standalone question.

Do NOT answer the question.

Only rewrite it if needed.

If it is already a complete question,
return it unchanged.
"""
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        (
            "human",
            "{input}"
        )

    ]

)

# ==========================================================
# Prompt 2
# Final QA Prompt
# ==========================================================

QA_PROMPT = ChatPromptTemplate.from_messages(

    [

        (

            "system",

            """
You are HCL AI Employee Assistant.

Answer ONLY from the provided company documents.

Rules:

1. Never make up information.

2. If the answer is unavailable,
reply:

'I could not find this information in the company documents.'

3. Keep answers short and professional.

4. Mention policy names whenever possible.

Context:

{context}
"""
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        (
            "human",
            "{input}"
        )

    ]

)