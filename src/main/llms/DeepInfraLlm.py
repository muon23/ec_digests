import logging
import os
from typing import Any, List

from llms.Llm import Llm
from llms.DeepInfraChatRunnable import DeepInfraChatRunnable


class DeepInfraLlm(Llm):

    SUPPORTED_MODELS = [
        "Sao10K/L3.3-70B-Euryale-v2.3",
        "meta-llama/Llama-3.3-70B-Instruct",
        "microsoft/WizardLM-2-8x22B",
        "google/gemini-2.0-flash-001",
    ]

    MODEL_ALIASES = {
        "euryale": "Sao10K/L3.3-70B-Euryale-v2.3",
        "llama-3": "meta-llama/Llama-3.3-70B-Instruct",
        "wizardlm-2": "microsoft/WizardLM-2-8x22B",
        "gemini-2": "google/gemini-2.0-flash-001",
    }

    __MODEL_TOKEN_LIMITS = {
        "Sao10K/L3.3-70B-Euryale-v2.3": 131_072,
        "meta-llama/Llama-3.3-70B-Instruct": 128_000,
        "microsoft/WizardLM-2-8x22B": 65_536,
        "google/gemini-2.0-flash-001": 1_000_000,
    }

    __API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

    def __init__(self, model_name: str = "llama-2", model_key: str = None, **kwargs):
        self.model_name = model_name
        self.role_names = ["system", "user", "assistant"]

        if self.model_name in self.MODEL_ALIASES:
            self.model_name = self.MODEL_ALIASES[self.model_name]

        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"LLM model {model_name} not supported")

        logging.info(f"Using {self.model_name}")

        self.model_key = model_key if model_key else os.environ.get("DEEPINFRA_API_KEY", None)
        if not self.model_key:
            raise RuntimeError(f"DeepInfra API key is not provided")

        self.llm = DeepInfraChatRunnable(
            model_name=self.model_name,
            api_key=self.model_key,
            api_url=self.__API_URL,
        )

        super().__init__(llm=self.llm)

    def clean_up_response(self, response: Any) -> dict:
        if isinstance(response, str):
            return {
                "content": response,
                "metadata": {}
            }

        else:
            raise TypeError(f"Unsupported return type for GptBot.react() (was {type(response)})")

    def get_max_tokens(self) -> int:
        return self.__MODEL_TOKEN_LIMITS.get(self.model_name, 1000)

    @classmethod
    def get_supported_models(cls) -> List[str]:
        return list(cls.MODEL_ALIASES.keys()) + cls.SUPPORTED_MODELS
