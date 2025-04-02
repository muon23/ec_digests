from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from langchain_core.prompts import PromptTemplate

from llms import Llm


class Extractor(ABC):
    """
    The text below is extracted from a {article_type}.
    You are to extract a list of items and their values.  List only within the scope of {scope}.
    Use JSON format with the following fields:
    - entity: (e.g., IBM total, AWS division)
    - item: (e.g., revenue, net income, gross profit margin)
    - value: (e.g., $1,000,000, 75%) (use negative value for down trend)
    - unit: (e.g., USD, ea, %)
    - comments: (optional. for any supporting information, such as the contributing factor affecting the value)
    Output only the list.
    ===
    {text}
    """

    @dataclass
    class SchemaSpec:
        name: str
        description: str
        example: str
        optional: bool

    DEFAULT_PROMPT = """
    The text below is extracted from a {article_name}.
    You are to extract a list of items and their values.  {scope}
    Use JSON format with the following fields:
    {schema}
    Output only the list.
    ===
    {text}
    """

    DEFAULT_SCHEMA = [

    ]

    def __init__(
            self,
            llm: Llm,
            prompt: str = DEFAULT_PROMPT,
            schema: Sequence[SchemaSpec] = None
    ):
        self.llm = llm
        self.prompt = PromptTemplate(prompt)
        self.schema = schema or self.DEFAULT_SCHEMA
    
    @abstractmethod
    def extract(self, file: str, description: str = None) -> pd.DataFrame:
        pass
