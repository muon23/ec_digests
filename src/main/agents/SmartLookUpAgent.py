import logging
from typing import List

from langchain_core.prompts import PromptTemplate

from agents.AgentWrapper import AgentWrapper
from llms import Llm
from util import TableIndexer


class SmartLookUpAgent(AgentWrapper):
    DEFAULT_PROMPT = """
        You are a {agent_role}. You have access to the following tool_specs:
        {tools}
    
        The tool_specs you can use are: {tool_names}
    
        Question: {input}
        Current Context: {context}
        Previous Steps: {agent_scratchpad}
    
        {instructions}
        
        Respond in the following format:
        Thought: [Your reasoning process]
        Action: [The tool to use, if applicable]
        Action Input: [Your input to the tool, if applicable]
    
        OR
    
        If you have a final answer:
        Thought: [Your reasoning process]
        Final Answer: [{final_answer_type}]
        """

    DEFAULT_AGENT_ROLE = "AI agent tasked with answering questions"

    CONCISE_FINAL_ANSWER = "Concise answer without descriptions"
    COMPLETE_FINAL_ANSWER = "Your complete response"

    DEFAULT_INSTRUCTIONS = """
        Follow these steps:
        1. Determine if the question can be answered directly or if it requires additional information.
        2. If additional information is needed, decide whether to look it up or break the question into sub-questions.
        3. If a tool fails to provide useful results after {max_tool_attempts} attempts, stop using it and try another tool.
        4. If no tool can provide the necessary information, explain why the question cannot be answered and provide suggestions for how to find the answer.
        5. Combine all the information to provide a final answer.

        Important:
        - If you have enough information to answer the question, provide a **Final Answer** and nothing else.
        - If you need more information, specify an **Action** and an **Action Input**, and do not provide a final answer.
    """

    DEFAULT_MAX_TOOL_ATTEMPTS = 2

    @staticmethod
    def look_up_database(agent: "SmartLookUpAgent", query: str) -> str:
        logging.info(f"\n*** search: {query}")
        response = agent.table.query(query)
        logging.info(f"\n*** search response: {response}")
        return response

    @staticmethod
    def decompose_question(agent: "SmartLookUpAgent", query: str) -> List[str]:
        logging.info(f"\n*** decompose: {query}")

        prompt = f"""
            Break up the following question into multiple simple questions, each on its own line.
            ===
            {query}
            """
        response = agent.llm.invoke(prompt)["content"]

        logging.info(response)
        return response.split("\n")

    def __init__(
            self,
            llm: Llm,
            table: TableIndexer,
            prompt: str = DEFAULT_PROMPT,
            concise_final_answer: bool = True,
            **kwargs
    ):

        # Manipulate prompt template
        self.prompt = PromptTemplate.from_template(prompt or self.DEFAULT_PROMPT).partial(
            final_answer_type=self.CONCISE_FINAL_ANSWER if concise_final_answer else self.COMPLETE_FINAL_ANSWER
        )

        kwargs.setdefault("agent_role", self.DEFAULT_AGENT_ROLE)

        instructions = PromptTemplate.from_template(self.DEFAULT_INSTRUCTIONS).format(
            max_tool_attempts=kwargs.get("max_tool_attempts", self.DEFAULT_MAX_TOOL_ATTEMPTS),
        )
        kwargs.setdefault("instructions", instructions)

        prompt_arguments = {p: kwargs[p] for p in kwargs.keys() if p in self.prompt.input_variables}
        self.prompt = self.prompt.partial(**prompt_arguments)

        self.verbose = kwargs.get("verbose", True)

        # Initialize the super class
        super().__init__(llm, self.prompt)

        # Add default tools
        self.add_tool(
            name="look_up_database",
            func=self.look_up_database,
            description="Look up specific information in the database.  Only capable of one single answer."
        )
        self.add_tool(
            name="decompose_question",
            func=self.decompose_question,
            description="Break down complex questions into smaller sub-questions."
        )

        # Data member initialization
        self.table: TableIndexer = table


