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

    @classmethod
    def template(cls, pack_id: str, family: str) -> "Manifest":
        return cls(
            id=pack_id,
            name="TODO",
            version="1.0.0",
            language="fr",
            family=family,
            source_pdfs=["TODO"]
        )


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
    critical_success_on: int = 20
    critical_failure_on: int = 1

    @classmethod
    def template(cls) -> "D20VsTargetConfig":
        return cls(
            family="D20VsTarget",
            advantage_enabled=True,
            advantage_dice=2,
            critical_success_on=20,
            critical_failure_on=1
        )


class PbtA2d6Config(BaseModel):
    family: Literal["PbtA2d6"]
    tiers: list[TierDef]

    @classmethod
    def template(cls) -> "PbtA2d6Config":
        return cls(
            family="PbtA2d6",
            tiers=[
                TierDef(id="miss", max=6, outcome=Outcome.FAILURE, label="TODO"),
                TierDef(id="weak", min=7, max=9, outcome=Outcome.PARTIAL_SUCCESS, label="TODO"),
                TierDef(id="strong", min=10, outcome=Outcome.SUCCESS, label="TODO")
            ]
        )


class StepTargetD20Config(BaseModel):
    family: Literal["StepTargetD20"]
    multiplier: int
    min_difficulty: int
    max_difficulty: int
    auto_success_at: int
    max_total_reduction: int | None = None

    @classmethod
    def template(cls) -> "StepTargetD20Config":
        return cls(
            family="StepTargetD20",
            multiplier=3,
            min_difficulty=0,
            max_difficulty=10,
            auto_success_at=0
        )


class DicePoolSuccessConfig(BaseModel):
    family: Literal["DicePoolSuccess"]
    default_pool_size: int = 1
    dice_faces: int
    success_threshold: int
    botch_rule: bool = False
    botch_threshold: int | None = None
    botch_condition: Literal["more_botches_than_successes", "any_botch"] = "more_botches_than_successes"

    @classmethod
    def template(cls) -> "DicePoolSuccessConfig":
        return cls(
            family="DicePoolSuccess",
            default_pool_size=1,
            dice_faces=6,
            success_threshold=5,
            botch_rule=True,
            botch_threshold=1,
            botch_condition="more_botches_than_successes"
        )


FAMILIES = {
    "D20VsTarget": D20VsTargetConfig,
    "PbtA2d6": PbtA2d6Config,
    "StepTargetD20": StepTargetD20Config,
    "DicePoolSuccess": DicePoolSuccessConfig,
}


ResolutionConfig = Annotated[
    Union[D20VsTargetConfig, PbtA2d6Config, StepTargetD20Config, DicePoolSuccessConfig],
    Field(discriminator="family")
]


# --- Resources Config Models ---

class RecoveryTrigger(BaseModel):
    id: str
    name: str

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
    recovery_triggers: list[RecoveryTrigger]
    pools: list[PoolDef]
    pool_groups: list[PoolGroupDef]

    def model_post_init(self, __context):
        # Validate that all triggers in recovery rules exist in recovery_triggers
        valid_trigger_ids = {t.id for t in self.recovery_triggers}

        for pool in self.pools:
            for rule in pool.recovery:
                if rule.trigger not in valid_trigger_ids:
                    raise ValueError(f"RecoveryRule trigger '{rule.trigger}' in pool '{pool.id}' is not defined in recovery_triggers.")

        for group in self.pool_groups:
            for rule in group.recovery:
                if rule.trigger not in valid_trigger_ids:
                    raise ValueError(f"RecoveryRule trigger '{rule.trigger}' in pool group '{group.id}' is not defined in recovery_triggers.")

    @classmethod
    def template(cls) -> "ResourcesConfig":
        return cls(
            recovery_triggers=[RecoveryTrigger(id="TODO", name="TODO")],
            pools=[
                PoolDef(
                    id="hp",
                    name="TODO",
                    kind="health",
                    min_value=0,
                    current_path="TODO",
                    max_path="TODO",
                    recovery=[RecoveryRule(trigger="TODO", mode="full")]
                )
            ],
            pool_groups=[
                PoolGroupDef(
                    id="spells",
                    name="TODO",
                    path="TODO",
                    recovery=[RecoveryRule(trigger="TODO", mode="full")]
                )
            ]
        )


# --- Triggers Config Models ---

class TriggerRule(BaseModel):
    id: str
    kind: Literal["consume", "recover"]
    target: str | None = None
    trigger: str | None = None
    amount: int | None = None
    key_regex: dict[str, str] | None = None
    key_default: str | None = None
    key_template: str | None = None
    keywords: dict[str, list[str]]

    def model_post_init(self, __context):
        if self.kind == "consume" and self.target is None:
            raise ValueError("target must be provided when kind is 'consume'")
        if self.kind == "recover" and self.trigger is None:
            raise ValueError("trigger must be provided when kind is 'recover'")
        if self.key_template is not None and "{key}" not in self.key_template:
            raise ValueError("key_template must contain '{key}' if provided")


class TriggersConfig(BaseModel):
    version: int
    rules: list[TriggerRule]


# --- Resolution Request/Result ---

class ResolutionRequest(BaseModel):
    modifier: int = 0
    difficulty: int | None = None
    advantage: Literal["none", "advantage", "disadvantage"] = "none"
    step_adjustments: dict[str, int] = Field(default_factory=dict)
    roll_bonus: int = 0
    critical_applies: bool = True
    pool_size: int | None = None


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
