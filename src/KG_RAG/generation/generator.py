from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage


class Generator:
    def __init__(self, config):
        self.config = config
        self.model = OllamaLLM(model=config["model"])
        self.model.temperature = config["temperature"]
        self.system_message = config["prompts"].get("system", None)
        
    def generate(self, prompt):
        messages = [
            SystemMessage(self.config["prompts"].get("system", None)),
            HumanMessage(prompt)
        ]
        response = self.model.invoke(messages)
        return response