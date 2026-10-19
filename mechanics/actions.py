from typing import Literal, Optional
from pydantic import BaseModel, Field

from mechanics.expr import Formula


class DifficultyLevel(BaseModel):
    """
    guidance: Benchmark for a specific difficulty level.
    """
    level: int = Field(ge=0)
    name: str
    guidance: str = ""


class EffortRule(BaseModel):
    cost_expr: Formula
    adjustment_key: str = "effort"
    max_levels_path: Optional[str] = None
    edge_path_template: Optional[str] = None


class TaskAction(BaseModel):
    """
    guidance: GM procedure for this action, injected into the analysis prompt.
    """
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    kind: Literal["task"] = "task"
    name: str
    description: str = ""
    guidance: str = ""
    pools: list[str] = Field(default_factory=list)
    effort: Optional[EffortRule] = None


class ActionsConfig(BaseModel):
    version: int = 1
    difficulty_scale: list[DifficultyLevel] = Field(default_factory=list)
    actions: list[TaskAction] = Field(default_factory=list)
