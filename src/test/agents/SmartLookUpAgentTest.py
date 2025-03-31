import unittest

import llms
from agents.SmartLookUpAgent import SmartLookUpAgent


class SmartLookUpAgentTest(unittest.TestCase):

    @staticmethod
    def mocked_look_up_database(agent: "SmartLookUpAgent", query: str) -> str:
        print(f"\n*** search: {query}")
        has_linked_in = "LinkedIn" in query
        has_microsoft = "Microsoft" in query
        q1 = "1Q25" in query or "Q1 2025" in query
        q2 = "2Q25" in query or "Q2 2025" in query

        if has_microsoft and has_linked_in:
            response = "not found"
        elif q1 and q2:
            response = "not found"
        elif has_microsoft:
            if q1:
                response = "600 billion USD"
            elif q2:
                response = "500 billion USD"
            else:
                response = "not found"
        elif has_linked_in:
            if q1:
                response = "50 billion USD"
            elif q2:
                response = "60 billion USD"
            else:
                response = "not found"
        else:
            response = "not found"
        print(f"\n*** search response: {response}")

        return response

    def test_something(self):
        # llm = llms.of("llama-3")
        # llm = llms.of("gpt-4o")
        # llm = llms.of("deepseek-v3")
        llm = llms.of("gemini-2t")
        table = ...
        agent = SmartLookUpAgent(llm, table)
        agent.replace_tool("look_up_database", func=self.mocked_look_up_database)

        result = agent.invoke(
            "How much percentage does LinkedIn contribute to overall combined revenues for Microsoft in the last two quarters?",
            context="Today: 2Q25",
            verbose=True,
            temperature=0.4,
        )

        print(result)
        self.assertIn("10%", result["output"])


if __name__ == '__main__':
    unittest.main()
