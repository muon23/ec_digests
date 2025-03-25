from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, Runnable, RunnableConfig
from transformers import AutoTokenizer


class Bot(ABC):
    class Role(Enum):
        SYSTEM = 0
        HUMAN = 1
        AI = 2

    def __init__(self, llm: Runnable, role_names: dict = None):
        self.llm = llm
        self.role_names = {
            self.Role.SYSTEM: "system",
            self.Role.HUMAN: "user",
            self.Role.AI: "assistant",
        }
        if role_names:
            for r in role_names:
                self.role_names[r] = role_names[r]

        # For counting tokens
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def react(self, prompt: Sequence[tuple[Role | str, str] | str] | str, **kwargs) -> dict:
        # Format the prompt
        prompt = self.preprocess_prompt(prompt)

        # Prompt template parameters
        arguments = kwargs.get("arguments", {})

        # Task, e.g., chat or completion
        task = kwargs.get("task", self.get_default_task())

        # Create a chain and run
        chain = RunnableSequence(prompt | self.llm)
        config = RunnableConfig(metadata={"task": task})
        response = chain.invoke(input=arguments, config=config, **kwargs)

        return self.clean_up_response(response)

    def preprocess_prompt(self, prompt: Sequence[tuple[Role | str, str] | str] | str) -> ChatPromptTemplate:
        # Reformat the prompt
        if isinstance(prompt, str):
            return ChatPromptTemplate(messages=[prompt])
        else:
            messages = []
            for msg in prompt:
                role_name = self.role_names[msg[0]] if isinstance(msg[0], self.Role) else msg[0]
                messages.append((role_name, msg[1]))
            return ChatPromptTemplate(messages=messages)

    @abstractmethod
    def clean_up_response(self, response: Any) -> dict:
        pass

    def get_num_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @abstractmethod
    def get_max_tokens(self) -> int:
        pass

    def get_default_task(self) -> str:
        return "chat"

    @classmethod
    @abstractmethod
    def get_supported_models(cls) -> List[str]:
        pass
