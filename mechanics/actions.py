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
    refund_on_20: bool = False
    damage_per_level: int = 0


class TaskAction(BaseModel):
    """
    guidance: GM procedure for this action, injected into the analysis prompt.
    """
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    kind: Literal["task"] = "task"
    role: Literal["general", "attack", "defend"] = "general"
    name: str
    description: str = ""
    guidance: str = ""
    pools: list[str] = Field(default_factory=list)
    effort: Optional[EffortRule] = None
    default_damage: int = 0
    weapon_damage: dict[str, int] = Field(default_factory=dict)
    damage_expr: Optional[Formula] = None
    roll_damage_bonuses: dict[str, int] = Field(default_factory=dict)
    critical_damage_expr: Optional[Formula] = None
    difficulty_expr: Optional[Formula] = None
    npc_health_expr: Optional[Formula] = None
    npc_armor_expr: Optional[Formula] = None
    default_incoming_damage_expr: Optional[Formula] = None
    impaired_max_roll_bonus: Optional[int] = None
    impaired_suppressed_special_rolls: list[str] = Field(default_factory=list)
    special_rolls: dict[str, str] = Field(default_factory=dict)
    special_roll_labels: dict[str, str] = Field(default_factory=dict)


class ActionsConfig(BaseModel):
    version: int = 1
    difficulty_scale: list[DifficultyLevel] = Field(default_factory=list)
    actions: list[TaskAction] = Field(default_factory=list)
