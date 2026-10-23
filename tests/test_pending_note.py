from mechanics.task_flow import format_pending_allocation_note
from mechanics.models import ResourcesConfig, PoolDef
from pydantic import BaseModel

class DummyPack(BaseModel):
    resources: ResourcesConfig | None = None

def test_format_pending_allocation_note():
    pack = DummyPack(
        resources=ResourcesConfig(
            pools=[
                PoolDef(id="hp", name="Points de Vie", current_path="hp.current", max_value=10, kind="health", recovery=[])
            ],
            recovery_triggers=[],
            pool_groups=[],
            pooled_recoveries=[],
            recovery_sequences=[]
        )
    )

    sheet = {"hp": {"current": 4}}
    pending = {"points": 3, "among": ["hp", "mp"]}

    note = format_pending_allocation_note(pending, sheet, pack)
    expected = "\n⏳ RÉPARTITION EN ATTENTE : 3 point(s) de récupération à répartir entre : hp (Points de Vie) 4/10, mp. Demande au joueur comment les répartir ; ne décide pas à sa place."

    assert note == expected

def test_format_pending_allocation_note_defensive():
    assert format_pending_allocation_note(None, {}, None) == ""
    assert format_pending_allocation_note({}, {}, None) == ""
    assert format_pending_allocation_note({"points": "trois", "among": ["hp"]}, {}, None) == ""
