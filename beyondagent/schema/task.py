from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class Task(BaseModel):
    task_id: str = Field(default=...)

    env_type: str = Field(default="appworld")

    metadata: dict = Field(default_factory=dict)

    query: List | str = Field(default="")


class TaskObjective(BaseModel):
    task:Task=Field(...,description="task")
    description:str=Field(...,description="description of the task")
    ground_truth:str=Field(...,description="ground truth")
    confidence:Optional[float]=Field(None,description="confidence")
    reward:Optional[float]=Field(None,description="reward")
    
    @property
    def objective(self):
        return self.task.query