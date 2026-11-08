import json
import pytest
from mechanics.character import CharacterSchemaConfig, schema_as_legacy_dict
from validation import validate_character_sheet
from systems.validate import validate_pack
from game_state_engine import GameStateEngine
from systems.loader import load_pack

def setup_pack(tmp_path, pack_id="test_pack", schema_data=None, manual_data=None, pools=None):
    pack_dir = tmp_path / pack_id
    pack_dir.mkdir(parents=True, exist_ok=True)

    # Write manifest
    (pack_dir / "manifest.yaml").write_text("""id: test_pack
name: Test Pack
version: 1.0.0
language: fr
family: StepTargetD20
source_pdfs: []
""")

    # Write resolution
    (pack_dir / "resolution.json").write_text(json.dumps({
        "family": "StepTargetD20",
        "multiplier": 3,
        "min_difficulty": 0,
        "max_difficulty": 10,
        "auto_success_at": 0
    }))

    if pools is None:
        pools = []

    # Write resources
    (pack_dir / "resources.json").write_text(json.dumps({
        "recovery_triggers": [],
        "pools": pools,
        "pool_groups": []
    }))

    # Write optional files
    if schema_data is not None:
        (pack_dir / "character_schema.json").write_text(json.dumps(schema_data))
    if manual_data is not None:
        (pack_dir / "creation_manual.json").write_text(json.dumps(manual_data))

    return str(pack_dir)


def test_schema_as_legacy_dict():
    schema_json = {
        "required_fields": [
            {"path": "name", "type": "string"},
            {"path": "resources.hit_points", "type": "object", "sub_fields": ["current", "max"]}
        ]
    }
    cfg = CharacterSchemaConfig.model_validate(schema_json)
    legacy = schema_as_legacy_dict(cfg)

    # Valid character
    char_valid = {
        "name": "Hero",
        "resources": {
            "hit_points": {"current": 10, "max": 10}
        }
    }
    is_complete, missing = validate_character_sheet(char_valid, legacy)
    assert is_complete
    assert not missing

    # Invalid character
    char_invalid = {
        "name": "Hero"
    }
    is_complete, missing = validate_character_sheet(char_invalid, legacy)
    assert not is_complete
    assert missing == ["resources.hit_points"]


def test_validate_pack_character_files_valid(tmp_path):
    schema_data = {
        "required_fields": [{"path": "name", "type": "string"}]
    }
    manual_data = {
        "steps": [{"step": 1, "name": "Step 1", "description": "Do stuff"}]
    }
    pack_dir = setup_pack(tmp_path, schema_data=schema_data, manual_data=manual_data)
    issues = validate_pack(pack_dir)
    assert not any(i.severity == "error" for i in issues)


def test_validate_pack_schema_invalid_path(tmp_path):
    schema_data = {
        "required_fields": [{"path": "invalid..path", "type": "string"}]
    }
    pack_dir = setup_pack(tmp_path, schema_data=schema_data)
    issues = validate_pack(pack_dir)
    assert any(i.severity == "error" and i.file == "character_schema.json" and "Invalid path format" in i.message for i in issues)


def test_validate_pack_schema_duplicate_path(tmp_path):
    schema_data = {
        "required_fields": [
            {"path": "name", "type": "string"},
            {"path": "name", "type": "string"}
        ]
    }
    pack_dir = setup_pack(tmp_path, schema_data=schema_data)
    issues = validate_pack(pack_dir)
    assert any(i.severity == "error" and i.file == "character_schema.json" and "Duplicate path" in i.message for i in issues)


def test_validate_pack_schema_pool_coverage(tmp_path):
    pools = [{
        "id": "might",
        "name": "Might",
        "kind": "health",
        "current_path": "resources.might.current",
        "max_path": "resources.might.max",
        "recovery": []
    }]
    schema_data = {
        "required_fields": [{"path": "name", "type": "string"}]
    }
    pack_dir = setup_pack(tmp_path, schema_data=schema_data, pools=pools)
    issues = validate_pack(pack_dir)
    warnings = [i for i in issues if i.severity == "warning" and i.file == "character_schema.json" and "not covered" in i.message]
    assert len(warnings) == 1
    assert "pool 'might' (resources.might.current) is not covered" in warnings[0].message

    # Now cover it
    schema_data_covered = {
        "required_fields": [
            {"path": "name", "type": "string"},
            {"path": "resources.might", "type": "object", "sub_fields": ["current", "max"]}
        ]
    }
    pack_dir_covered = setup_pack(tmp_path, schema_data=schema_data_covered, pools=pools)
    issues_covered = validate_pack(pack_dir_covered)
    assert not any(i.severity == "warning" and "not covered" in i.message for i in issues_covered)


def test_validate_pack_manual_steps(tmp_path):
    # Empty steps should warn
    manual_empty = {"steps": []}
    pack_dir = setup_pack(tmp_path, manual_data=manual_empty)
    issues = validate_pack(pack_dir)
    assert any(i.severity == "warning" and i.file == "creation_manual.json" and "no steps" in i.message for i in issues)

    # Non-consecutive steps should error
    manual_non_consecutive = {
        "steps": [
            {"step": 1, "name": "Step 1", "description": "a"},
            {"step": 3, "name": "Step 3", "description": "b"}
        ]
    }
    pack_dir2 = setup_pack(tmp_path, pack_id="test2", manual_data=manual_non_consecutive)
    issues2 = validate_pack(pack_dir2)
    assert any(i.severity == "error" and i.file == "creation_manual.json" and "Expected step 2, got 3" in i.message for i in issues2)


def test_synchronize_and_recalculate_generic_pools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import os; os.makedirs("Memory", exist_ok=True)
    # 1. Generic pool clamping (min_value and max_path)
    pools1 = [{
        "id": "sanity",
        "name": "Santé mentale",
        "kind": "counter",
        "current_path": "resources.sanity.current",
        "max_path": "resources.sanity.max",
        "min_value": 0,
        "recovery": []
    }]
    setup_pack(tmp_path, pack_id="pack_sanity", pools=pools1)
    pack1 = load_pack("pack_sanity", base_dir=str(tmp_path))

    engine1 = GameStateEngine(pack=pack1)
    engine1.state = {
        "resources": {"sanity": {"current": 85, "max": 60}},
    }
    engine1.synchronize_and_recalculate()
    assert engine1.state["resources"]["sanity"]["current"] == 60
    assert "hit_points" not in engine1.state["resources"]

    # 2. Missing current initialized from max, negative clamped to min_value
    engine2 = GameStateEngine(pack=pack1)
    engine2.state = {"resources": {"sanity": {"max": 50}}}
    engine2.synchronize_and_recalculate()
    assert engine2.state["resources"]["sanity"]["current"] == 50

    engine2.state["resources"]["sanity"]["current"] = -5
    engine2.synchronize_and_recalculate()
    assert engine2.state["resources"]["sanity"]["current"] == 0


def test_synchronize_pools_across_dnd_and_cypher(tmp_path, monkeypatch):
    """
    Vérifie que synchronize_and_recalculate borne génériquement toutes les pools
    et pool_groups du pack actif (D&D 5e et Cypher) sans injecter de clés hit_points/pv hors-pack.
    """
    monkeypatch.chdir(tmp_path)
    import os; os.makedirs("Memory", exist_ok=True)
    from systems.loader import load_pack
    from pathlib import Path

    base_sys_dir = Path(__file__).parent.parent / "systems"

    # 1. Pack dnd5e_srd : current > max ramené à max, spell_slots bornés
    pack_dnd = load_pack("dnd5e_srd", base_dir=str(base_sys_dir))
    engine_dnd = GameStateEngine(pack=pack_dnd)
    engine_dnd.state = {
        "level": "3",
        "xp": "900",
        "resources": {
            "hit_points": {"current": 30, "max": 20},
            "spells_per_day": {"level_1": {"current": 5, "max": 2}},
        },
    }
    engine_dnd.synchronize_and_recalculate()
    assert engine_dnd.state["resources"]["hit_points"]["current"] == 20
    assert engine_dnd.state["resources"]["spells_per_day"]["level_1"]["current"] == 2
    assert engine_dnd.state["level"] == 3
    assert engine_dnd.state["xp"] == 900

    # 2. Pack cypher_system : might/speed/intellect bornés + damage_track recalculé, aucun hit_points ajouté
    pack_cypher = load_pack("cypher_system", base_dir=str(base_sys_dir))
    engine_cypher = GameStateEngine(pack=pack_cypher)
    engine_cypher.state = {
        "tier": "2",
        "resources": {
            "might": {"current": 0, "max": 12},
            "speed": {"current": 15, "max": 10},
            "intellect": {"current": 8, "max": 10},
        },
    }
    engine_cypher.synchronize_and_recalculate()
    assert "hit_points" not in engine_cypher.state["resources"]
    assert engine_cypher.state["resources"]["speed"]["current"] == 10
    assert engine_cypher.state["damage_track"] == "impaired"
    assert engine_cypher.state["tier"] == 2


def test_build_character_view_and_list_available_packs():
    from mechanics.character import build_character_view
    from systems.loader import list_available_packs, load_pack

    packs = list_available_packs()
    pack_ids = [p["id"] for p in packs]
    assert "cypher_system" in pack_ids
    assert "dnd5e_srd" in pack_ids

    # 1. Cypher view
    cypher_pack = load_pack("cypher_system")
    cypher_sheet = {
        "name": "Hector",
        "descriptor": "Clever",
        "type": "Adept",
        "focus": "Crafts Illusions",
        "tier": 1,
        "effort": 1,
        "armor": 1,
        "xp": 3,
        "damage_track": "hale",
        "cypher_limit": 2,
        "edge": {"might": 0, "speed": 0, "intellect": 1},
        "resources": {
            "might": {"current": 10, "max": 10},
            "speed": {"current": 10, "max": 10},
            "intellect": {"current": 15, "max": 15},
        },
        "skills": ["Psychologie", "Tromperie visuelle"],
        "abilities": ["Onslaught", "Ward", "Minor Illusion", "Far Step", "Scan"],
        "cyphers": ["Détonateur thermique (Niv 4)"],
        "equipment": ["Bâton runique", "Vêtements sombres"],
    }
    c_view = build_character_view(cypher_sheet, pack=cypher_pack)
    assert c_view["name"] == "Hector"
    scalar_labels = {s["path"]: s["label"] for s in c_view["scalars"]}
    assert scalar_labels["descriptor"] == "Descripteur"
    assert scalar_labels["tier"] == "Tier"
    assert scalar_labels["damage_track"] == "Piste de dégâts"
    assert len(c_view["pools"]) == 3
    assert c_view["stat_groups"][0]["label"] == "Avantage (Edge)"
    cypher_list_entry = next(l for l in c_view["lists"] if l["path"] == "cyphers")
    assert cypher_list_entry["max_items"] == 2
    assert c_view["other"] == {}

    # 2. D&D 5e view
    dnd_pack = load_pack("dnd5e_srd")
    dnd_sheet = {
        "name": "Lyra",
        "race": "Elfe",
        "class": "Magicien",
        "background": "Sage",
        "level": 2,
        "xp": 300,
        "proficiency_bonus": 2,
        "ac": 12,
        "statistics": {
            "strength": 8,
            "dexterity": 14,
            "constitution": 12,
            "intelligence": 16,
            "wisdom": 13,
            "charisma": 10,
        },
        "resources": {
            "hit_points": {"current": 11, "max": 12},
            "spells_per_day": {"level_1": {"current": 2, "max": 3}},
        },
        "skills": ["Arcanes", "Histoire"],
        "spells": ["Projectile magique", "Bouclier"],
        "equipment": ["Grimoire", "Dague"],
    }
    d_view = build_character_view(dnd_sheet, pack=dnd_pack)
    assert d_view["name"] == "Lyra"
    assert d_view["stat_groups"][0]["label"] == "Caractéristiques"
    assert d_view["stat_groups"][0]["items"][3] == {"key": "intelligence", "label": "Intelligence", "value": 16}
    assert len(d_view["pools"]) == 1
    assert len(d_view["pool_groups"]) == 1
    assert d_view["other"] == {}


def test_cypher_pack_character_schema_and_manual_loaded():
    """Vérifie que le pack cypher_system charge son character_schema.json et creation_manual.json avec options."""
    pack = load_pack("cypher_system")
    assert pack.character_schema is not None
    assert pack.creation_manual is not None

    # Options (types, descriptors, foci)
    options = pack.creation_manual.options
    assert "types" in options and len(options["types"]) == 4
    assert "descriptors" in options and len(options["descriptors"]) >= 10
    assert "foci" in options and len(options["foci"]) >= 10

    # Validation stricte du pack
    from pathlib import Path
    pack_dir = Path(__file__).parent.parent / "systems" / "cypher_system"
    issues = validate_pack(str(pack_dir))
    assert not issues, f"Issues found in cypher_system: {issues}"


def test_character_creator_and_sheet_manager_use_pack(tmp_path, monkeypatch):
    """Vérifie que CharacterCreator et SheetManagerAgent utilisent uniquement le pack."""
    monkeypatch.chdir(tmp_path)
    from unittest.mock import MagicMock, patch
    from agent import CharacterCreator, SheetManagerAgent

    pack = load_pack("cypher_system")

    with patch("agent.get_llm", return_value=MagicMock()):
        creator = CharacterCreator(pack=pack)
        manual_str = creator._load_manual()
        manual_data = json.loads(manual_str)
        assert manual_data is not None
        assert len(manual_data["steps"]) == 7
        assert "types" in manual_data["options"]
        assert "Warrior (Guerrier)" in manual_str

        ctx = creator.get_context("Je veux créer un Warrior Tough")
        assert "EXPECTED SHEET FIELDS (GLOSSARY):" in ctx
        assert "resources.might.max" in ctx

        sheet_mgr = SheetManagerAgent(pack=pack)
        glossary_str = sheet_mgr._format_glossary()
        assert "resources.might.current" in glossary_str
        assert "edge.might" in glossary_str
        assert "damage_track" in glossary_str


def test_finalize_pack_character_deterministic_and_no_llm(tmp_path, monkeypatch):
    """Vérifie que _extract_and_add_resources utilise la finalisation déterministe du pack (sans appel LLM)."""
    monkeypatch.chdir(tmp_path)
    from unittest.mock import MagicMock, patch
    from agent import RPGAgent

    pack = load_pack("cypher_system")

    with patch("agent.get_llm", return_value=MagicMock()), \
         patch("agent.get_embeddings", return_value=MagicMock()), \
         patch("agent.Chroma"), \
         patch("agent.CharacterCreator"), \
         patch("agent.Narrator"), \
         patch("agent.ChronicleAgent"), \
         patch("agent.SheetManagerAgent"), \
         patch("agent.ScenarioExtractorAgent"):
        agent = RPGAgent(verbose=False)
        agent.gse = GameStateEngine(pack=pack)
        agent.llm = MagicMock()

        raw_sheet = {
            "name": "Kaelen",
            "descriptor": "Tough",
            "type": "Warrior",
            "focus": "Masters Weaponry",
            "resources": {
                "might": {"total": 12},
                "speed": {"max": 10, "current": 0},
                "intellect": {"restants": 8, "total": 8}
            },
            "skills": ["Might defense"],
            "abilities": ["Weapon Master", "Bash", "Swipe", "Control the Field", "Practiced in Armor"],
            "equipment": ["Épée longue", "Armure moyenne"]
        }

        agent.character_data = raw_sheet
        agent._extract_and_add_resources()
        final_sheet = agent.character_data

        # Aucun appel LLM ne doit avoir eu lieu
        agent.llm.invoke.assert_not_called()

        # Normalisation total -> max et initialisation current = max
        assert final_sheet["resources"]["might"] == {"current": 12, "max": 12}
        assert final_sheet["resources"]["speed"] == {"current": 10, "max": 10}
        assert final_sheet["resources"]["intellect"] == {"current": 8, "max": 8}

        # Application des defaults du character_schema Cypher
        assert final_sheet["tier"] == 1
        assert final_sheet["effort"] == 1
        assert final_sheet["armor"] == 0
        assert final_sheet["damage_track"] == "hale"
        assert final_sheet["recovery_step"] == 0
        assert final_sheet["edge"] == {"might": 0, "speed": 0, "intellect": 0}
        assert final_sheet["skills"] == ["Might defense"]
        assert final_sheet["equipment"] == ["Épée longue", "Armure moyenne"]

        # Une fiche à l'étape 4 (1 seule capacité de Focus et équipement vide) ne doit PAS être complète
        schema = agent._load_character_schema()
        step4_sheet = dict(final_sheet, abilities=["Minor Illusion"], equipment=[])
        is_step4_complete, step4_missing = validate_character_sheet(step4_sheet, schema)
        assert not is_step4_complete
        assert "abilities (1/5)" in step4_missing
        assert "equipment" in step4_missing

        # La fiche finalisée complète doit valider le schéma du pack Cypher
        is_complete, missing = validate_character_sheet(final_sheet, schema)
        assert is_complete, f"Sheet should be complete, missing: {missing}"

        # Et elle doit fonctionner directement avec execute_task du moteur Cypher
        agent.gse.state = final_sheet
        res = agent.gse.execute_task(
            action_id="task",
            pool_id="might",
            difficulty=3,
            step_adjustments={"skill": -1},
            effort_levels=1
        )
        assert res.success
        # Coût d'1 niveau d'effort sans edge = 3 points de might
        assert res.state_changes["effort"]["cost"] == 3
        assert agent.gse.state["resources"]["might"]["current"] == 9


def test_dnd5e_pack_character_schema_and_manual(tmp_path, monkeypatch):
    """Vérifie que le pack dnd5e_srd possède un schéma et un manuel complets et finalise correctement une fiche."""
    monkeypatch.chdir(tmp_path)
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from agent import RPGAgent

    pack = load_pack("dnd5e_srd")
    assert pack.character_schema is not None
    assert pack.creation_manual is not None
    assert len(pack.creation_manual.steps) == 6
    assert len(pack.creation_manual.options["races"]) == 9
    assert len(pack.creation_manual.options["classes"]) == 12

    pack_dir = Path(__file__).parent.parent / "systems" / "dnd5e_srd"
    issues = validate_pack(str(pack_dir))
    assert not issues, f"Issues found in dnd5e_srd: {issues}"

    with patch("agent.get_llm", return_value=MagicMock()), \
         patch("agent.get_embeddings", return_value=MagicMock()), \
         patch("agent.Chroma"), \
         patch("agent.CharacterCreator"), \
         patch("agent.Narrator"), \
         patch("agent.ChronicleAgent"), \
         patch("agent.SheetManagerAgent"), \
         patch("agent.ScenarioExtractorAgent"):
        agent = RPGAgent(verbose=False)
        agent.gse = GameStateEngine(pack=pack)
        agent.llm = MagicMock()

        # Une fiche avec seulement le nom ne doit PAS être complète
        schema = agent._load_character_schema()
        is_complete_partial, missing_partial = validate_character_sheet({"name": "Elminster"}, schema)
        assert not is_complete_partial
        assert "race" in missing_partial
        assert "class" in missing_partial

        # Fiche complète de magicien avec total/restants sur PV et emplacements de sorts
        raw_dnd_sheet = {
            "name": "Elminster",
            "race": "Humain",
            "class": "Magicien",
            "ac": 12,
            "statistics": {
                "strength": 9,
                "dexterity": 14,
                "constitution": 13,
                "intelligence": 16,
                "wisdom": 13,
                "charisma": 11
            },
            "resources": {
                "hit_points": {"total": 7},
                "spells_per_day": {
                    "level_1": {"total": 2}
                }
            },
            "skills": ["Arcanes", "Histoire"],
            "equipment": ["Bâton", "Grimoire"]
        }

        agent.character_data = raw_dnd_sheet
        agent._extract_and_add_resources()
        final_dnd = agent.character_data

        assert final_dnd["level"] == 1
        assert final_dnd["proficiency_bonus"] == 2
        assert final_dnd["resources"]["hit_points"] == {"current": 7, "max": 7}
        assert final_dnd["resources"]["spells_per_day"]["level_1"] == {"current": 2, "max": 2}

        is_complete, missing = validate_character_sheet(final_dnd, schema)
        assert is_complete, f"D&D sheet should be complete, missing: {missing}"


def test_terminer_verb_does_not_trigger_premature_summary(tmp_path, monkeypatch):
    """
    Régression : le verbe 'terminer' dans 'Pour terminer le profil d'Hector...'
    ne doit PAS déclencher ending_announced ni audit_and_complete ni basculer en SUMMARY.
    """
    monkeypatch.chdir(tmp_path)
    from unittest.mock import MagicMock, patch
    from agent import RPGAgent

    pack = load_pack("cypher_system")

    with patch("agent.get_llm", return_value=MagicMock()), \
         patch("agent.get_embeddings", return_value=MagicMock()), \
         patch("agent.Chroma"), \
         patch("agent.CharacterCreator"), \
         patch("agent.Narrator"), \
         patch("agent.ChronicleAgent"), \
         patch("agent.SheetManagerAgent"), \
         patch("agent.ScenarioExtractorAgent"):
        agent = RPGAgent(verbose=False)
        agent.gse = GameStateEngine(pack=pack)

        agent.character_creator.generate_response = MagicMock(
            return_value="Pour terminer le profil d'Hector, quel équipement principal souhaitez-vous porter ?"
        )
        agent.sheet_manager.update_sheet = MagicMock(
            return_value={
                "name": "Hector",
                "descriptor": "Stealthy",
                "type": "Explorer",
                "focus": "Fights Dirty",
                "resources": {
                    "might": {"current": 10, "max": 10},
                    "speed": {"current": 11, "max": 11},
                    "intellect": {"current": 9, "max": 9}
                },
                "skills": ["Furtivité"],
                "abilities": ["Tracker & Stalker"],
                "equipment": ["Arme légère dissimulée"]
            }
        )
        agent.sheet_manager.audit_and_complete = MagicMock()

        agent.chat("Je choisis Fights Dirty")

        # audit_and_complete ne doit PAS avoir été appelé et on reste en CREATION
        agent.sheet_manager.audit_and_complete.assert_not_called()
        assert agent.game_state == "CREATION"



