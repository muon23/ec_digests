from llms.Llm import Llm
from llms.DeepInfraLlm import DeepInfraLlm
from llms.DeepSeekLlm import DeepSeekLlm
from llms.GptLlm import GptLlm
from llms.LlamaLlm import LlamaBot


def of(model_name: str, **kwargs) -> Llm:
    bots = [GptLlm, DeepInfraLlm, LlamaBot, DeepSeekLlm]

    for bot in bots:
        if model_name in bot.get_supported_models():
            return bot(model_name, **kwargs)

    raise RuntimeError(f"Model {model_name} not supported.")

