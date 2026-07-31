import json
import os
import shutil
import pytest
from unittest import mock
from validation import validate_character_sheet
from agent import RPGAgent, SheetManagerAgent
from scenario_agents import ManualGeneratorAgent

def setup_module():
    if os.path.exists("Memory"):
        shutil.rmtree("Memory")
    os.makedirs("Memory", exist_ok=True)


def test_pure_validation_add_like():
    schema = {
        "required_fields": [
            {"path": "name", "type": "string"},
            {"path": "statistics", "type": "object", "sub_fields": ["Force", "Dexterite", "Constitution", "Intelligence", "Sagesse", "Charisme"]},
            {"path": "resources.hit_points", "type": "object", "sub_fields": ["current", "max"]},
            {"path": "equipment", "type": "list", "non_empty": True}
        ]
    }

    # Esteban's sheet incomplete -> equipment and resources.hit_points missing
    incomplete_char = {
        "name": "Esteban",
        "statistics": {
            "Force": 15, "Dexterite": 12, "Constitution": 14,
            "Intelligence": 10, "Sagesse": 13, "Charisme": 14
        }
    }
    is_complete, missing = validate_character_sheet(incomplete_char, schema)
    assert not is_complete
    assert "resources.hit_points" in missing
    assert "equipment" in missing

    # Same schema + completed sheet
    complete_char = {
        "name": "Esteban",
        "statistics": {
            "Force": 15, "Dexterite": 12, "Constitution": 14,
            "Intelligence": 10, "Sagesse": 13, "Charisme": 14
        },
        "resources": {
            "hit_points": {"current": 12, "max": 12}
        },
        "equipment": ["Epée de bois", "Bouclier en cuir"]
    }
    is_complete, missing = validate_character_sheet(complete_char, schema)
    assert is_complete
    assert len(missing) == 0


def test_pure_validation_investigation_horror():
    schema = {
        "required_fields": [
            {"path": "name", "type": "string"},
            {"path": "statistics", "type": "object", "sub_fields": ["FOR", "DEX", "POU", "CON", "APP", "EDU", "INT", "TAI"]},
            {"path": "resources.sante_physique", "type": "object", "sub_fields": ["current", "max"]},
            {"path": "resources.sante_mentale", "type": "object", "sub_fields": ["current", "max"]},
            {"path": "competences", "type": "object", "sub_fields": ["Trouver Objet", "Psychologie"]}
        ]
    }

    # Incomplete sheet -> resources and skills missing
    incomplete_char = {
        "name": "Harvey",
        "statistics": {
            "FOR": 50, "DEX": 60, "POU": 75, "CON": 50, "APP": 40, "EDU": 80, "INT": 85, "TAI": 65
        }
    }
    is_complete, missing = validate_character_sheet(incomplete_char, schema)
    assert not is_complete
    assert "resources.sante_physique" in missing
    assert "resources.sante_mentale" in missing
    assert "competences" in missing

    # Same horror schema + completed sheet
    complete_char = {
        "name": "Harvey",
        "statistics": {
            "FOR": 50, "DEX": 60, "POU": 75, "CON": 50, "APP": 40, "EDU": 80, "INT": 85, "TAI": 65
        },
        "resources": {
            "sante_physique": {"current": 10, "max": 10},
            "sante_mentale": {"current": 75, "max": 75}
        },
        "competences": {
            "Trouver Objet": 45,
            "Psychologie": 60
        }
    }
    is_complete, missing = validate_character_sheet(complete_char, schema)
    assert is_complete
    assert len(missing) == 0


def test_pure_validation_fallback_schema():
    # Schema absent / empty -> must return False and clear message
    is_complete, missing = validate_character_sheet({"name": "Test"}, None)
    assert not is_complete
    assert "validation schema missing or empty" in missing[0]

    # Empty schema
    is_complete_empty_schema, missing_empty_schema = validate_character_sheet({"name": "Test"}, {"required_fields": []})
    assert not is_complete_empty_schema
    assert "validation schema missing or empty" in missing_empty_schema[0]

    is_complete_empty, missing_empty = validate_character_sheet(None, {"required_fields": [{"path": "name", "type": "string"}]})
    assert not is_complete_empty
    assert "character_data missing" in missing_empty


def test_orchestration_audit_triggered():
    agent = RPGAgent()

    # 1. Write schema in Memory
    schema = {
        "required_fields": [
            {"path": "name", "type": "string"},
            {"path": "race", "type": "string"}
        ]
    }
    os.makedirs("Memory", exist_ok=True)
    with open("Memory/character_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f)

    agent.sheet_manager.update_sheet = mock.Mock(
        return_value={"name": "Aragorn"}  # Incomplete, "race" missing
    )
    agent.character_creator.generate_response = mock.Mock(
        return_value="Félicitations, la création est terminée, tu es prêt pour l'aventure ! CREATION_COMPLETED"
    )
    agent._extract_and_add_resources = mock.Mock()

    # Mock audit_and_complete to return incomplete sheet first
    agent.sheet_manager.audit_and_complete = mock.Mock(
        return_value={"name": "Aragorn"}  # Remains incomplete
    )

    agent.chat("Finir creation")
    assert agent.sheet_manager.audit_and_complete.called
    assert agent.game_state == "CREATION"  # No transition because still incomplete after audit

    # Second case: audit_and_complete returns complete sheet
    agent.sheet_manager.audit_and_complete = mock.Mock(
        return_value={"name": "Aragorn", "race": "Humain"}  # Becomes complete
    )
    agent.chat("Finir creation")
    assert agent.game_state == "SUMMARY"  # Transition because complete after audit


def test_orchestration_transition_no_audit_when_already_complete():
    agent = RPGAgent()

    schema = {
        "required_fields": [
            {"path": "name", "type": "string"},
            {"path": "race", "type": "string"}
        ]
    }
    os.makedirs("Memory", exist_ok=True)
    with open("Memory/character_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f)

    agent.sheet_manager.update_sheet = mock.Mock(
        return_value={"name": "Aragorn", "race": "Humain"}  # Already complete
    )
    agent.character_creator.generate_response = mock.Mock(
        return_value="Tu as choisi l'Humain."  # No ending keyword
    )
    agent.sheet_manager.audit_and_complete = mock.Mock()
    agent._extract_and_add_resources = mock.Mock()

    agent.chat("Je choisis l'Humain")
    assert not agent.sheet_manager.audit_and_complete.called
    assert agent.game_state == "SUMMARY"


def test_manual_generator_agent_schema_fallback():
    # Test where ManualGeneratorAgent's LLM call mocked for schema returns invalid/empty JSON
    # -> check fallback on {"required_fields": []} and check that character_schema.json is written.

    schema_path = "Memory/character_schema.json"
    if os.path.exists(schema_path):
        os.remove(schema_path)

    mock_store = mock.Mock()
    mock_store.get.return_value = {
        "documents": ["Règles de création de personnage."],
        "metadatas": [{"page": 1}]
    }
    mock_store.similarity_search.return_value = [mock.Mock(page_content="Règles de création de personnage.")]

    agent = ManualGeneratorAgent(mock_store)

    with mock.patch.object(agent.llm.__class__, "invoke", side_effect=[
        mock.Mock(content='{"steps": [{"step": 1, "name": "Step 1", "description": "Desc"}, {"step": 2, "name": "Step 2", "description": "Desc"}, {"step": 3, "name": "Step 3", "description": "Desc"}, {"step": 4, "name": "Step 4", "description": "Desc"}]}'),
        Exception("LLM crash")
    ]):
        res = agent.generate()

    assert len(res.get("steps", [])) == 4
    assert os.path.exists(schema_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        written_schema = json.load(f)
    assert written_schema == {"required_fields": []}
