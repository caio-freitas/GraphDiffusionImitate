from abc import ABC, abstractmethod
from typing import Dict
from imitation.agent.base_agent import BaseAgent

class BaseRunner(ABC):
    def __init__(self, output_dir) -> None:
        self.output_dir = output_dir
        self.env = None
    
    @abstractmethod
    def run(self, agent: BaseAgent) -> Dict:
        raise NotImplementedError()
    
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()