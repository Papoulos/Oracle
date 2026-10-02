import random
import pytest

from mechanics.models import (
    D20VsTargetConfig,
    PbtA2d6Config,
    StepTargetD20Config,
    DicePoolSuccessConfig,
    TierDef,
    Outcome,
    ResolutionRequest
)
from mechanics.resolution import resolve


def test_d20_vs_target_success():
    config = D20VsTargetConfig(family="D20VsTarget")
    rng = random.Random(42) # Gives 4

    request = ResolutionRequest(modifier=12, difficulty=15)
    result = resolve(config, request, rng)

    assert result.rolled == [4]
    assert result.total == 16
    assert result.outcome == Outcome.SUCCESS
    assert result.margin == 1
    assert result.flags == []


def test_d20_vs_target_advantage():
    config = D20VsTargetConfig(family="D20VsTarget", advantage_enabled=True)
    rng = random.Random(42) # Rolls 4, 1

    request = ResolutionRequest(modifier=0, difficulty=4, advantage="advantage")
    result = resolve(config, request, rng)

    assert result.rolled == [4, 1]
    assert result.total == 4
    assert result.outcome == Outcome.SUCCESS


def test_d20_vs_target_critical():
    config = D20VsTargetConfig(family="D20VsTarget")
    rng = random.Random(60) # Gives 10, let's find a seed for 20: 8 gives 5, let's mock it
    # Find a seed that gives 20
    # using a simple loop below if we had access, but let's just mock or find one
    # seed 9 gives 10... let's just use a fixed mock for rng to be robust
    class MockRNG:
        def randint(self, a, b):
            return 20
    rng = MockRNG()

    request = ResolutionRequest(modifier=0, difficulty=25)
    result = resolve(config, request, rng)

    assert result.rolled == [20]
    assert result.total == 20
    assert result.outcome == Outcome.CRITICAL_SUCCESS
    assert "critical_success" in result.flags


def test_pbta_2d6():
    config = PbtA2d6Config(
        family="PbtA2d6",
        tiers=[
            TierDef(id="miss", max=6, outcome=Outcome.FAILURE, label="Échec"),
            TierDef(id="weak", min=7, max=9, outcome=Outcome.PARTIAL_SUCCESS, label="Partiel"),
            TierDef(id="strong", min=10, outcome=Outcome.SUCCESS, label="Total")
        ]
    )
    rng = random.Random(1) # Rolls 2, 5 -> 7

    request = ResolutionRequest(modifier=1)
    result = resolve(config, request, rng)

    assert result.rolled == [2, 5]
    assert result.total == 8
    assert result.outcome == Outcome.PARTIAL_SUCCESS
    assert result.tier_id == "weak"


def test_step_target_d20():
    config = StepTargetD20Config(
        family="StepTargetD20",
        multiplier=3,
        min_difficulty=0,
        max_difficulty=10,
        auto_success_at=0
    )
    rng = random.Random(42) # Gives 4

    request = ResolutionRequest(
        difficulty=4,
        step_adjustments={"skill": -1, "asset": -1, "effort": -1}
    )
    result = resolve(config, request, rng)

    assert result.difficulty_initial == 4
    assert result.difficulty_final == 1
    assert result.target == 3
    assert result.rolled == [4]
    assert result.total == 4
    assert result.outcome == Outcome.SUCCESS
    assert result.margin == 1


def test_step_target_auto_success():
    config = StepTargetD20Config(
        family="StepTargetD20",
        multiplier=3,
        min_difficulty=0,
        max_difficulty=10,
        auto_success_at=0
    )
    rng = random.Random(42)

    request = ResolutionRequest(
        difficulty=2,
        step_adjustments={"skill": -2}
    )
    result = resolve(config, request, rng)

    assert result.difficulty_initial == 2
    assert result.difficulty_final == 0
    assert result.target == 0
    assert result.rolled == []
    assert result.total == 0
    assert result.outcome == Outcome.SUCCESS


def test_dice_pool_success():
    config = DicePoolSuccessConfig(
        family="DicePoolSuccess",
        dice_faces=10,
        success_threshold=8,
        botch_rule=True,
        botch_threshold=1
    )
    rng = random.Random(10) # 3 rolls: 10, 1, 7

    request = ResolutionRequest(pool_size=3, difficulty=1) # 3 dice
    result = resolve(config, request, rng)

    # 10, 1, 7 -> 1 success, 1 botch. successes (1) is not < botches (1), so not botch.
    # To test botch let's mock it
    class MockBotchRNG:
        def randint(self, a, b):
            self.rolls = getattr(self, 'rolls', [1, 1, 9])
            return self.rolls.pop(0)

    request = ResolutionRequest(pool_size=3, difficulty=1)
    result = resolve(config, request, MockBotchRNG())

    assert result.rolled == [1, 1, 9]
    assert result.total == 1
    assert result.outcome == Outcome.CRITICAL_FAILURE
    assert "botch" in result.flags

def test_d20_vs_target_no_difficulty():
    config = D20VsTargetConfig(family="D20VsTarget")
    rng = random.Random(42)
    request = ResolutionRequest(modifier=0)
    with pytest.raises(ValueError):
        resolve(config, request, rng)

def test_step_target_d20_no_difficulty():
    config = StepTargetD20Config(family="StepTargetD20", multiplier=3, min_difficulty=0, max_difficulty=10, auto_success_at=0)
    rng = random.Random(42)
    request = ResolutionRequest(modifier=0)
    with pytest.raises(ValueError):
        resolve(config, request, rng)

def test_d20_critical_applies():
    config = D20VsTargetConfig(family="D20VsTarget", critical_success_on=20)
    # Roll a 20 without critical_applies
    class MockRNG:
        def randint(self, a, b):
            return 20
    request = ResolutionRequest(modifier=-5, difficulty=25, critical_applies=False)
    result = resolve(config, request, MockRNG())
    assert result.outcome == Outcome.FAILURE # total 15 vs 25 -> fails since critical doesn't apply

    # Roll a 20 with critical_applies
    request2 = ResolutionRequest(modifier=-5, difficulty=25, critical_applies=True)
    result2 = resolve(config, request2, MockRNG())
    assert result2.outcome == Outcome.CRITICAL_SUCCESS

def test_dice_pool_botch_conditions():
    config_more_botches = DicePoolSuccessConfig(
        family="DicePoolSuccess",
        dice_faces=10,
        success_threshold=8,
        botch_rule=True,
        botch_threshold=1,
        botch_condition="more_botches_than_successes"
    )
    config_any_botch = DicePoolSuccessConfig(
        family="DicePoolSuccess",
        dice_faces=10,
        success_threshold=8,
        botch_rule=True,
        botch_threshold=1,
        botch_condition="any_botch"
    )

    class MockRNG:
        def __init__(self, rolls):
            self.rolls = rolls.copy()
        def randint(self, a, b):
            return self.rolls.pop(0)

    # Roll: 1 success, 1 botch.
    # more_botches: botches(1) > successes(1) is False -> SUCCESS (since 1 success)
    request = ResolutionRequest(pool_size=2, difficulty=1)
    res1 = resolve(config_more_botches, request, MockRNG([8, 1]))
    assert res1.outcome == Outcome.SUCCESS

    # any_botch: botches(1) > 0 is True -> CRITICAL_FAILURE
    res2 = resolve(config_any_botch, request, MockRNG([8, 1]))
    assert res2.outcome == Outcome.CRITICAL_FAILURE


def test_trigger_rule_key_template_invalid():
    from mechanics.models import TriggerRule
    with pytest.raises(ValueError, match="key_template must contain '{key}'"):
        TriggerRule(id="test", kind="consume", target="foo", key_template="level_", keywords={"fr": ["test"]})

def test_trigger_rule_key_template_valid():
    from mechanics.models import TriggerRule
    rule = TriggerRule(id="test", kind="consume", target="foo", key_template="level_{key}", keywords={"fr": ["test"]})
    assert rule.key_template == "level_{key}"
