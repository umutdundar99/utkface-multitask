from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    CONTRASTIVE = "contrastive"
    MULTITASK = "multitask"


@dataclass
class GlobalConfig:
    task: TaskType = TaskType.CONTRASTIVE
