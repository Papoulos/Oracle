import random
from typing import Any

from mechanics.models import (
    ResolutionConfig,
    ResolutionRequest,
    ResolutionResult,
    Outcome,
    D20VsTargetConfig,
    PbtA2d6Config,
    StepTargetD20Config,
    DicePoolSuccessConfig,
)


def resolve(
    config: ResolutionConfig,
    request: ResolutionRequest,
    rng: random.Random
) -> ResolutionResult:
    """Resolves a check based on the configuration family."""
    if isinstance(config, D20VsTargetConfig):
        return _resolve_d20_vs_target(config, request, rng)
    elif isinstance(config, PbtA2d6Config):
        return _resolve_pbta_2d6(config, request, rng)
    elif isinstance(config, StepTargetD20Config):
        return _resolve_step_target_d20(config, request, rng)
    elif isinstance(config, DicePoolSuccessConfig):
        return _resolve_dice_pool_success(config, request, rng)
    else:
        raise ValueError(f"Unknown config family: {config.family}")


def _resolve_d20_vs_target(
    config: D20VsTargetConfig,
    request: ResolutionRequest,
    rng: random.Random
) -> ResolutionResult:
    warnings = []

    # Handle advantage
    advantage = request.advantage
    if advantage != "none" and not config.advantage_enabled:
        warnings.append("Advantage/disadvantage not enabled for this family.")
        advantage = "none"

    if advantage == "advantage":
        rolls = [rng.randint(1, 20) for _ in range(config.advantage_dice)]
        best_roll = max(rolls)
        rolled = rolls
    elif advantage == "disadvantage":
        rolls = [rng.randint(1, 20) for _ in range(config.advantage_dice)]
        best_roll = min(rolls)
        rolled = rolls
    else:
        best_roll = rng.randint(1, 20)
        rolled = [best_roll]

    total = best_roll + request.modifier + request.roll_bonus
    flags = []

    if request.difficulty is None:
        raise ValueError("Difficulty is required for D20VsTarget resolution.")
    target = request.difficulty

    if request.critical_applies and best_roll >= config.critical_success_on:
        flags.append("critical_success")
        outcome = Outcome.CRITICAL_SUCCESS
    elif request.critical_applies and best_roll <= config.critical_failure_on:
        flags.append("critical_failure")
        outcome = Outcome.CRITICAL_FAILURE
    else:
        if total >= target:
            outcome = Outcome.SUCCESS
        else:
            outcome = Outcome.FAILURE

    margin = total - target

    return ResolutionResult(
        rolled=rolled,
        total=total,
        outcome=outcome,
        margin=margin,
        flags=flags,
        warnings=warnings
    )


def _resolve_pbta_2d6(
    config: PbtA2d6Config,
    request: ResolutionRequest,
    rng: random.Random
) -> ResolutionResult:
    rolled = [rng.randint(1, 6), rng.randint(1, 6)]
    total = sum(rolled) + request.modifier + request.roll_bonus

    outcome = Outcome.FAILURE
    tier_id = None
    tier_label = None

    for tier in config.tiers:
        min_val = tier.min if tier.min is not None else float('-inf')
        max_val = tier.max if tier.max is not None else float('inf')

        if min_val <= total <= max_val:
            outcome = tier.outcome
            tier_id = tier.id
            tier_label = tier.label
            break

    return ResolutionResult(
        rolled=rolled,
        total=total,
        outcome=outcome,
        tier_id=tier_id,
        tier_label=tier_label
    )


def _resolve_step_target_d20(
    config: StepTargetD20Config,
    request: ResolutionRequest,
    rng: random.Random
) -> ResolutionResult:
    if request.difficulty is None:
        raise ValueError("Difficulty is required for StepTargetD20 resolution.")
    difficulty_initial = request.difficulty

    total_reduction = sum(request.step_adjustments.values())
    if config.max_total_reduction is not None:
        if total_reduction < -config.max_total_reduction:
            total_reduction = -config.max_total_reduction

    difficulty_final = difficulty_initial + total_reduction

    # Clamp difficulty
    difficulty_final = max(config.min_difficulty, min(config.max_difficulty, difficulty_final))

    target = difficulty_final * config.multiplier

    if difficulty_final == config.auto_success_at:
        return ResolutionResult(
            rolled=[],
            total=0,
            outcome=Outcome.SUCCESS,
            difficulty_initial=difficulty_initial,
            difficulty_final=difficulty_final,
            target=target,
            margin=None
        )

    roll = rng.randint(1, 20)
    total = roll + request.modifier + request.roll_bonus

    margin = total - target
    outcome = Outcome.SUCCESS if total >= target else Outcome.FAILURE

    return ResolutionResult(
        rolled=[roll],
        total=total,
        outcome=outcome,
        difficulty_initial=difficulty_initial,
        difficulty_final=difficulty_final,
        target=target,
        margin=margin
    )


def _resolve_dice_pool_success(
    config: DicePoolSuccessConfig,
    request: ResolutionRequest,
    rng: random.Random
) -> ResolutionResult:
    pool_size = request.pool_size if request.pool_size is not None else config.default_pool_size

    rolled = [rng.randint(1, config.dice_faces) for _ in range(pool_size)]

    successes = sum(1 for r in rolled if r >= config.success_threshold)
    botches = sum(1 for r in rolled if r <= config.botch_threshold) if config.botch_rule and config.botch_threshold else 0

    flags = []
    if config.botch_rule:
        botch_triggered = False
        if config.botch_condition == "more_botches_than_successes":
            botch_triggered = botches > successes
        elif config.botch_condition == "any_botch":
            botch_triggered = botches > 0

        if botch_triggered:
            outcome = Outcome.CRITICAL_FAILURE
            flags.append("botch")
        elif successes > 0:
            outcome = Outcome.SUCCESS
        else:
            outcome = Outcome.FAILURE
    else:
        if successes > 0:
            outcome = Outcome.SUCCESS
        else:
            outcome = Outcome.FAILURE

    # Difficulty could represent required successes
    required_successes = request.difficulty if request.difficulty is not None else 1

    if outcome == Outcome.SUCCESS and successes < required_successes:
         outcome = Outcome.FAILURE

    return ResolutionResult(
        rolled=rolled,
        total=successes,
        outcome=outcome,
        margin=successes - required_successes if request.difficulty is not None else successes,
        flags=flags
    )
