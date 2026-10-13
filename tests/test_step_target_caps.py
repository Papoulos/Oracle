import random
from pathlib import Path
import pytest
from mechanics.models import (
    StepTargetD20Config,
    ResolutionRequest,
    Outcome,
    D20VsTargetConfig,
    PbtA2d6Config,
    DicePoolSuccessConfig,
)
from mechanics.resolution import resolve
from systems.loader import load_pack

def test_step_target_caps_basic_limit():
    """difficulté 4, skill -3 -> plafonné à -2 : difficulté finale 2, cible 6, un warning 'skill capped'"""
    config = StepTargetD20Config(
        family="StepTargetD20",
        multiplier=3,
        min_difficulty=0,
        max_difficulty=10,
        auto_success_at=0,
        max_total_reduction=10,
        adjustment_caps={"skill": 2, "asset": 2, "effort": 6}
    )
    request = ResolutionRequest(
        difficulty=4,
        step_adjustments={"skill": -3}
    )
    rng = random.Random(42)
    result = resolve(config, request, rng)

    assert result.difficulty_final == 2
    assert result.target == 6
    assert "skill capped at 2" in result.warnings
    assert result.impossible is False
    assert result.outcome == Outcome.FAILURE


def test_step_target_max_total_reduction():
    """difficulté 6, skill -2, asset -2, effort -6 -> total -10 mais max_total_reduction 10 : difficulté finale 0,
       réussite sans jet"""
    config = StepTargetD20Config(
        family="StepTargetD20",
        multiplier=3,
        min_difficulty=0,
        max_difficulty=10,
        auto_success_at=0,
        max_total_reduction=10,
        adjustment_caps={"skill": 2, "asset": 2, "effort": 6}
    )
    request = ResolutionRequest(
        difficulty=6,
        step_adjustments={"skill": -2, "asset": -2, "effort": -6}
    )
    rng = random.Random(42)
    result = resolve(config, request, rng)

    assert result.difficulty_final == 0
    assert result.target == 0
    assert result.warnings == []  # No single cap is exceeded
    assert result.rolled == []
    assert result.total == 0
    assert result.outcome == Outcome.SUCCESS
    assert result.impossible is False


def test_step_target_caps_exact_and_positive():
    """effort -7 -> plafonné à -6 ; une valeur positive "hindrance": +1 n'est jamais plafonnée"""
    config = StepTargetD20Config(
        family="StepTargetD20",
        multiplier=3,
        min_difficulty=0,
        max_difficulty=10,
        auto_success_at=0,
        max_total_reduction=10,
        adjustment_caps={"skill": 2, "asset": 2, "effort": 6, "hindrance": 0}
    )
    request = ResolutionRequest(
        difficulty=4,
        step_adjustments={"effort": -7, "hindrance": 1}
    )
    rng = random.Random(42)
    result = resolve(config, request, rng)

    # total_reduction: effort cap = -6, hindrance = 1 -> -5
    assert result.difficulty_final == max(0, 4 - 5)  # 0
    assert result.target == 0
    assert "effort capped at 6" in result.warnings
    assert "hindrance capped at 0" not in result.warnings # hindrance is positive, so it's not capped
    assert result.outcome == Outcome.SUCCESS
    assert result.impossible is False


def test_step_target_impossible_no_modifier():
    """difficulté 7 sans ajustement : impossible True, rolled [], FAILURE, et le rng n'a pas avancé"""
    config = StepTargetD20Config(
        family="StepTargetD20",
        multiplier=3,
        min_difficulty=0,
        max_difficulty=10,
        auto_success_at=0,
        max_total_reduction=10,
        adjustment_caps={"skill": 2, "asset": 2, "effort": 6}
    )
    request = ResolutionRequest(difficulty=7)
    rng = random.Random(42)

    # Save the next random value to verify RNG isn't consumed
    test_rng = random.Random(42)
    expected_roll = test_rng.randint(1, 20)

    result = resolve(config, request, rng)

    assert result.difficulty_final == 7
    assert result.target == 21
    assert result.impossible is True
    assert result.rolled == []
    assert result.total == 0
    assert result.outcome == Outcome.FAILURE

    # Verify RNG did not advance
    assert rng.randint(1, 20) == expected_roll


def test_step_target_not_impossible_with_modifier():
    """difficulté 7 avec modifier +1 : cible 21 <= 20 + 1 -> le dé est lancé, impossible False"""
    config = StepTargetD20Config(
        family="StepTargetD20",
        multiplier=3,
        min_difficulty=0,
        max_difficulty=10,
        auto_success_at=0,
        max_total_reduction=10,
        adjustment_caps={"skill": 2, "asset": 2, "effort": 6}
    )
    request = ResolutionRequest(difficulty=7, modifier=1)
    rng = random.Random(42) # 4
    result = resolve(config, request, rng)

    assert result.target == 21
    assert result.impossible is False
    assert result.rolled == [4]
    assert result.total == 5
    assert result.outcome == Outcome.FAILURE


def test_existing_packs_compatibility():
    """packs existants (dnd5e_srd, fixture step_target_pack) : résultats identiques"""
    # 1. dnd5e_srd (D20VsTarget)
    fixtures_dir = str(Path(__file__).parent / "fixtures")
    systems_dir = str(Path(__file__).parent.parent / "systems")

    dnd5e_pack = load_pack("dnd5e_srd", base_dir=systems_dir)
    request_dnd = ResolutionRequest(difficulty=15, modifier=2)
    rng_dnd = random.Random(42) # roll = 4
    result_dnd = resolve(dnd5e_pack.resolution, request_dnd, rng_dnd)

    assert result_dnd.rolled == [4]
    assert result_dnd.total == 6
    assert result_dnd.outcome == Outcome.FAILURE
    assert result_dnd.warnings == []
    assert result_dnd.impossible is False

    # 2. step_target_pack fixture (StepTargetD20 with no adjustment caps defined)
    step_pack = load_pack("step_target_pack", base_dir=fixtures_dir)
    request_step = ResolutionRequest(difficulty=5, step_adjustments={"skill": -1})
    rng_step = random.Random(42) # roll = 4
    result_step = resolve(step_pack.resolution, request_step, rng_step)

    # difficulty 5 - 1 = 4 -> target 12
    assert result_step.target == 12
    assert result_step.rolled == [4]
    assert result_step.total == 4
    assert result_step.outcome == Outcome.FAILURE
    assert result_step.warnings == []
    assert result_step.impossible is False


def test_adjustment_caps_validation_error():
    """adjustment_caps avec une valeur négative : erreur de validation"""
    with pytest.raises(ValueError, match="adjustment_cap for 'skill' must be >= 0"):
        StepTargetD20Config(
            family="StepTargetD20",
            multiplier=3,
            min_difficulty=0,
            max_difficulty=10,
            auto_success_at=0,
            adjustment_caps={"skill": -1}
        )


def test_other_families_impossible_is_false():
    """Other resolvers maintain impossible=False"""
    request = ResolutionRequest(difficulty=15)
    rng = random.Random(42)

    d20_config = D20VsTargetConfig.template()
    res_d20 = resolve(d20_config, request, rng)
    assert res_d20.impossible is False

    pbta_config = PbtA2d6Config.template()
    res_pbta = resolve(pbta_config, request, rng)
    assert res_pbta.impossible is False

    dice_config = DicePoolSuccessConfig.template()
    res_dice = resolve(dice_config, request, rng)
    assert res_dice.impossible is False
