"""
OpenAI-powered farming chatbot.
The API key is read from the .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DEFAULT_MODEL = "gpt-6-luna"

SYSTEM_PROMPT = (
    "You are AI Farmer Assistant, a friendly and knowledgeable agriculture expert. "
    "You help farmers with crop selection, soil health, fertilizers, irrigation, pests, "
    "plant diseases, weather planning, seasons, and farming best practices. "
    "Use simple, clear language and short practical steps. "
    "If a question is not related to farming, agriculture, weather, or rural livelihood, "
    "politely say you can only help with farming topics. "
    "For serious pesticide, chemical, or disease problems, remind the farmer to also "
    "consult a local agriculture officer. "
    "Reply in the same language the farmer uses."
)


def get_api_key():
    key = os.getenv("OPENAI_API_KEY")

    if key:
        key = key.strip().strip('"').strip("'")

    return key or None


class FarmerChatbot:

    def __init__(self):
        api_key = get_api_key()

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please add it to your .env file."
            )

        self.model_name = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(api_key=api_key)

    def ask(self, question, context=""):
        question = (question or "").strip()

        if not question:
            return "Please type a question."

        if context:
            message = (
                f"Background information (use it only if relevant):\n{context}\n\n"
                f"Farmer's question: {question}"
            )
        else:
            message = question

        try:
            response = self.client.responses.create(
                model=self.model_name,
                instructions=SYSTEM_PROMPT,
                input=message,
            )

            text = response.output_text

            if not text:
                return "I could not generate an answer. Please try again."

            return text

        except Exception as error:
            return f"Sorry, I could not reach the OpenAI service: {error}"