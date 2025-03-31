from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Any

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from llms import Llm


class AgentWrapper(ABC):
    @dataclass
    class ToolSpec:
        name: str
        func: Callable[["AgentWrapper", ...], Any] = None
        description: str = None

    def __init__(self, llm: Llm, prompt: PromptTemplate):
        self.llm = llm
        self.prompt = prompt

        self.tool_specs: List[AgentWrapper.ToolSpec] = []
        self.executor: AgentExecutor | None = None

    def replace_tool(self, name: str, func: Callable = None, description: str = None):
        if self.executor:
            raise RuntimeError("Cannot replace a tool after the agent executor has already started")

        for t in self.tool_specs:
            if t.name == name:
                t.func = func or t.func
                t.description = description or t.description

    def add_tool(self, name: str, func: Callable = None, description: str = None):
        if self.executor:
            raise RuntimeError("Cannot add a tool after the agent executor has already started")

        self.tool_specs.append(self.ToolSpec(name, func, description))

    def remove_tools(self, names: List[str]):
        if self.executor:
            raise RuntimeError("Cannot remove tools after the agent executor has already started")

        for t in self.tool_specs:
            if t.name in names:
                self.tool_specs.remove(t)

    def get_tool_names(self):
        return [t.name for t in self.tool_specs]

    def _bind_tools(self) -> List[Tool]:
        return [
            Tool(
                name=ts.name,
                func=partial(ts.func, self),
                description=ts.description,
            )
            for ts in self.tool_specs
        ]

    def _create_agent_chain(self, **kwargs):
        # creating tools from the tool specs
        tools = self._bind_tools()

        # create the agent
        agent = create_react_agent(
            llm=self.llm.as_language_model(),
            tools=tools,
            prompt=self.prompt,
        )

        # Wrap the agent in an executor
        self.executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, **kwargs)

    def invoke(self, query: str, context="", **kwargs) -> dict:
        # Create agent if not yet created
        self.executor or self._create_agent_chain(**kwargs)

        # Turn on/off verbose per invocation
        self.executor.verbose = kwargs.get("verbose", self.executor.verbose)

        # Run the agent
        result = self.executor.invoke({
            "input": query,
            "context": context
        })

        return result
