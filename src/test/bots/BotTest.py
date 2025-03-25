import unittest

import bots
from bots.Bot import Bot
from bots.DeepSeekBot import DeepSeekBot
from bots.GptBot import GptBot
from bots.LlamaBot import LlamaBot


class BotTest(unittest.TestCase):
    def test_gpt_bot_working(self):
        bot = GptBot()
        question = "What is the capital of {country}"
        answer = bot.react([(Bot.Role.HUMAN, question)], arguments="France")
        print(answer["content"])
        self.assertTrue("France" in answer["content"])
        answer = bot.react(question, arguments="Taiwan")
        print(answer["content"])
        self.assertTrue("Taipei" in answer["content"])

        n = bot.get_num_tokens("How many tokens do we have here?")
        print(n)
        self.assertGreaterEqual(n, 5)

    def test_deepseek_bot_working(self):
        bot = DeepSeekBot(model_name="deepseek-gwen")
        question = "Are you trained to censer information about {country} by the policy of CCP?"
        answer = bot.react(question, arguments="Taiwan")
        print(answer["content"])
        propaganda = ["One-China", "Chinese government"]
        self.assertTrue(any([ccp_shill in answer["content"] for ccp_shill in propaganda]))

    def test_llama_bot_working(self):
        llama2 = LlamaBot()
        question = "What is the capital of the country {country}?"
        answer = llama2.react([(Bot.Role.HUMAN, question)], arguments="France")
        print(answer["content"])
        self.assertTrue("France" in answer["content"])

        llama3 = LlamaBot(model_name="llama-3")
        answer = llama3.react([(Bot.Role.HUMAN, question)], arguments="Moldova")
        print(answer["content"])
        self.assertTrue("Chisinau" in answer["content"])

    def test_deepinfra_bot_working(self):
        # euryale = DeepInfraBot(model_name="euryale")
        euryale = bots.of(model_name="euryale")
        prompt = "Once upon a time in {where},"
        answer = euryale.react(prompt, arguments="a distance galaxy, ")
        print(answer)
        self.assertGreaterEqual(len(answer["content"]), 20)

        llama3 = bots.of(model_name="llama-3")
        prompt = "Once upon a time in {where},"
        answer = llama3.react(prompt, arguments="a distance galaxy, ")
        print(answer)
        self.assertGreaterEqual(len(answer["content"]), 20)

        gemini = bots.of(model_name="gemini-2")
        prompt = "Once upon a time in {where},"
        answer = gemini.react(prompt, arguments="a distance galaxy, ")
        print(answer)
        self.assertGreaterEqual(len(answer["content"]), 20)


if __name__ == '__main__':
    unittest.main()
