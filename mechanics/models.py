from enum import Enum
from typing import Literal, Annotated, Union
from pydantic import BaseModel, Field


class Outcome(str, Enum):
    CRITICAL_SUCCESS = "CRITICAL_SUCCESS"
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILURE = "FAILURE"
    CRITICAL_FAILURE = "CRITICAL_FAILURE"


class Manifest(BaseModel):
    id: str
    name: str
    version: str
    language: str
    family: str
    source_pdfs: list[str]


# --- Resolution Config Models ---

class TierDef(BaseModel):
    id: str
    min: int | None = None
    max: int | None = None
    outcome: Outcome
    label: str


class D20VsTargetConfig(BaseModel):
    family: Literal["D20VsTarget"]
    advantage_enabled: bool = False
    advantage_dice: int = 2
    adv_dis_cancel: bool = True
    critical_success_on: int = 20
    critical_failure_on: int = 1


class PbtA2d6Config(BaseModel):
    family: Literal["PbtA2d6"]
    tiers: list[TierDef]


class StepTargetD20Config(BaseModel):
    family: Literal["StepTargetD20"]
    multiplier: int
    min_difficulty: int
    max_difficulty: int
    auto_success_at: int
    max_total_reduction: int | None = None


class DicePoolSuccessConfig(BaseModel):
    family: Literal["DicePoolSuccess"]
    default_pool_size: int = 1
    dice_faces: int
    success_threshold: int
    botch_rule: bool = False
    botch_threshold: int | None = None


ResolutionConfig = Annotated[
    Union[D20VsTargetConfig, PbtA2d6Config, StepTargetD20Config, DicePoolSuccessConfig],
    Field(discriminator="family")
]


# --- Resources Config Models ---

class RecoveryRule(BaseModel):
    trigger: str
    mode: Literal["full", "fixed", "percent"]
    amount: int | None = None

    def model_post_init(self, __context):
        if self.mode in ("fixed", "percent") and self.amount is None:
            raise ValueError("amount must be provided when mode is 'fixed' or 'percent'")
        if self.mode == "percent" and self.amount is not None:
            if not (0 <= self.amount <= 100):
                raise ValueError("amount must be between 0 and 100 when mode is 'percent'")


class PoolDef(BaseModel):
    id: str
    name: str
    kind: Literal["health", "counter", "track"]
    min_value: int = 0
    current_path: str
    max_value: int | None = None
    max_path: str | None = None
    recovery: list[RecoveryRule]

    # Exactly one of max_value or max_path must be provided
    def model_post_init(self, __context):
        if (self.max_value is None) == (self.max_path is None):
            raise ValueError("Exactly one of max_value or max_path must be provided")


class PoolGroupDef(BaseModel):
    id: str
    name: str
    path: str
    recovery: list[RecoveryRule]


class ResourcesConfig(BaseModel):
    recovery_triggers: list[dict[str, str]]
    pools: list[PoolDef]
    pool_groups: list[PoolGroupDef]


# --- Resolution Request/Result ---

class ResolutionRequest(BaseModel):
    modifier: int = 0
    difficulty: int | None = None
    advantage: Literal["none", "advantage", "disadvantage"] = "none"
    step_adjustments: dict[str, int] = Field(default_factory=dict)
    roll_bonus: int = 0


class ResolutionResult(BaseModel):
    rolled: list[int]
    total: int
    outcome: Outcome
    tier_id: str | None = None
    tier_label: str | None = None
    margin: int | None = None
    flags: list[str] = Field(default_factory=list)

    # Extra fields useful for specific mechanics (e.g. StepTargetD20)
    difficulty_initial: int | None = None
    difficulty_final: int | None = None
    target: int | None = None
    warnings: list[str] = Field(default_factory=list)
