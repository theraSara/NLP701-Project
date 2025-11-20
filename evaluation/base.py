from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    def __init__(self, method_name: str, dataset_name: str, config: dict):
        self.method_name = method_name
        self.dataset_name = dataset_name
        self.config = config
    
    @abstractmethod
    def evaluate(self, explanations: list[dict]) -> dict:
        pass
    
    def log(self, message: str):
        print(f"[{self.method_name.upper()}:{self.dataset_name}] {message}")