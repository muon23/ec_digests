import logging
import os
from typing import Any, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from llms.Llm import Llm


class GeminiLlm(Llm):
    SUPPORTED_MODELS = [
        "gemini-2.5-pro-exp-03-25",
        "gemini-2.0-flash",
        "gemini-2.0-flash-thinking-exp-01-21",
    ]

    MODEL_ALIASES = {
        "gemini-2.5": "gemini-2.5-pro-exp-03-25",
        "gemini-2": "gemini-2.0-flash",
        "gemini-2t": "gemini-2.0-flash-thinking-exp-01-21"
    }

    __MODEL_TOKEN_LIMITS = {
        "gemini-2.5-pro-exp-03-25": 1_048_576,
        "gemini-2.0-flash": 1_048_576,
        "gemini-2.0-flash-thinking-exp-01-21": 1_048_576,
    }

    def __init__(self, model_name: str = "gpt-4", model_key: str = None, **kwargs):
        self.model_name = model_name
        self.role_names = ["user", "user", "model"]

        if self.model_name in self.MODEL_ALIASES:
            self.model_name = self.MODEL_ALIASES[self.model_name]

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"LLM model {model_name} not supported")

        logging.info(f"Using {self.model_name}")

        self.model_key = model_key if model_key else os.environ.get("GEMINI_API_KEY", None)
        if not self.model_key:
            raise RuntimeError(f"Gemini API key not provided")

        self.llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.model_key, **kwargs)

        super().__init__(llm=self.llm, role_names={self.Role.SYSTEM: "user", self.Role.AI: "model"})

    def clean_up_response(self, response: Any) -> dict:
        if isinstance(response, AIMessage):
            return {
                "content": response.content,
                "metadata": response
            }

        else:
            raise TypeError(f"Unsupported return type for GptLlm.invoke() (was {type(response)})")

    def get_num_tokens(self, text: str) -> int:
        return self.llm.get_num_tokens(text)

    def get_max_tokens(self) -> int:
        return self.__MODEL_TOKEN_LIMITS.get(self.model_name, 100_000)

    @classmethod
    def get_supported_models(cls) -> List[str]:
        return list(cls.MODEL_ALIASES.keys()) + cls.SUPPORTED_MODELS

    def as_runnable(self) -> Runnable:
        return self.llm

    def as_language_model(self) -> BaseLanguageModel:
        return self.llm

