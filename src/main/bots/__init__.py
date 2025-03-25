from bots.Bot import Bot
from bots.DeepInfraBot import DeepInfraBot
from bots.DeepSeekBot import DeepSeekBot
from bots.GptBot import GptBot
from bots.LlamaBot import LlamaBot


def of(model_name: str, **kwargs) -> Bot:
    bots = [GptBot, DeepInfraBot, LlamaBot, DeepSeekBot]

    for bot in bots:
        if model_name in bot.get_supported_models():
            return bot(model_name, **kwargs)

    raise RuntimeError(f"Model {model_name} not supported.")

