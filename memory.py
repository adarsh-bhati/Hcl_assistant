from langchain_core.messages import (
    HumanMessage,
    AIMessage
)


class ConversationMemory:

    def __init__(self):

        self.chat_history = []

    def add_user_message(self, message):

        self.chat_history.append(
            HumanMessage(
                content=message
            )
        )

    def add_ai_message(self, message):

        self.chat_history.append(
            AIMessage(
                content=message
            )
        )

    def get_history(self):

        return self.chat_history

    def clear(self):

        self.chat_history = []