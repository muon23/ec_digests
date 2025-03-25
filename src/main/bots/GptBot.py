import logging
import os
from typing import Any, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

from bots.Bot import Bot


class GptBot(Bot):
    SUPPORTED_MODELS = [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4o-2024-08-06",
        "gpt-4o",
        "o1-preview", "o1", "gpt-4o-1",
        "o3-mini", "gpt-4o-3",
        "gpt-4.5-preview",
    ]

    MODEL_ALIASES = {
        "gpt-4.5": "gpt-4.5-preview",
        "gpt-4o+": "gpt-4o-2024-08-06",
        "gpt-o1": "o1",
        "gpt-o3": "o3-mini",
        "gpt-3.5": "gpt-3.5-turbo",
    }

    __MODEL_TOKEN_LIMITS = {
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16_000
    }

    def __init__(self, model_name: str = "gpt-4", model_key: str = None, **kwargs):
        self.model_name = model_name
        self.role_names = ["system", "user", "assistant"]

        if self.model_name in self.MODEL_ALIASES:
            self.model_name = self.MODEL_ALIASES[self.model_name]

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"LLM model {model_name} not supported")

        if self.model_name in ["o1", "o1-preview", "o3-mini"]:
            kwargs["temperature"] = 1
            role_names = {Bot.Role.SYSTEM: "user"}
        else:
            role_names = {}

        logging.info(f"Using {self.model_name}")

        self.model_key = model_key if model_key else os.environ.get("OPENAI_API_KEY", None)
        if not self.model_key:
            raise RuntimeError(f"OpenAI API key not provided")

        llm = ChatOpenAI(model_name=self.model_name, openai_api_key=self.model_key, **kwargs)

        super().__init__(llm=llm, role_names=role_names)

    def clean_up_response(self, response: Any) -> dict:
        if isinstance(response, AIMessage):
            return {
                "content": response.content,
                "metadata": response
            }

        else:
            raise TypeError(f"Unsupported return type for GptBot.react() (was {type(response)})")

    def get_num_tokens(self, text: str) -> int:
        return ChatOpenAI(temperature=0).get_num_tokens(text)

    def get_max_tokens(self) -> int:
        return self.__MODEL_TOKEN_LIMITS.get(self.model_name, 100_000)

    @classmethod
    def get_supported_models(cls) -> List[str]:
        return list(cls.MODEL_ALIASES.keys()) + cls.SUPPORTED_MODELS
