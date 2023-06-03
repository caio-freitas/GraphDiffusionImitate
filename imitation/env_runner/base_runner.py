from typing import Dict
from imitation.policy.base_policy import BasePolicy

class BaseRunner:
    def __init__(self, output_dir) -> None:
        self.output_dir = output_dir

    def run(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()
    def reset(self) -> None:
        raise NotImplementedError()