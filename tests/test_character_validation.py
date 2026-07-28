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
        "champs_requis": [
            {"chemin": "nom", "type": "string"},
            {"chemin": "caracteristiques", "type": "object", "sous_champs": ["Force", "Dexterite", "Constitution", "Intelligence", "Sagesse", "Charisme"]},
            {"chemin": "ressources.points_de_vie", "type": "object", "sous_champs": ["actuels", "max"]},
            {"chemin": "equipement", "type": "list", "non_vide": True}
        ]
    }

    # Fiche d'Esteban incomplète -> equipement et ressources.points_de_vie manquants
    incomplete_char = {
        "nom": "Esteban",
        "caracteristiques": {
            "Force": 15, "Dexterite": 12, "Constitution": 14,
            "Intelligence": 10, "Sagesse": 13, "Charisme": 14
        }
    }
    is_complete, missing = validate_character_sheet(incomplete_char, schema)
    assert not is_complete
    assert "ressources.points_de_vie" in missing
    assert "equipement" in missing

    # Même schéma + fiche complétée
    complete_char = {
        "nom": "Esteban",
        "caracteristiques": {
            "Force": 15, "Dexterite": 12, "Constitution": 14,
            "Intelligence": 10, "Sagesse": 13, "Charisme": 14
        },
        "ressources": {
            "points_de_vie": {"actuels": 12, "max": 12}
        },
        "equipement": ["Epée de bois", "Bouclier en cuir"]
    }
    is_complete, missing = validate_character_sheet(complete_char, schema)
    assert is_complete
    assert len(missing) == 0


def test_pure_validation_investigation_horror():
    schema = {
        "champs_requis": [
            {"chemin": "nom", "type": "string"},
            {"chemin": "caracteristiques", "type": "object", "sous_champs": ["FOR", "DEX", "POU", "CON", "APP", "EDU", "INT", "TAI"]},
            {"chemin": "ressources.sante_physique", "type": "object", "sous_champs": ["actuels", "max"]},
            {"chemin": "ressources.sante_mentale", "type": "object", "sous_champs": ["actuels", "max"]},
            {"chemin": "competences", "type": "object", "sous_champs": ["Trouver Objet", "Psychologie"]}
        ]
    }

    # Fiche incomplète -> ressources et compétences manquantes
    incomplete_char = {
        "nom": "Harvey",
        "caracteristiques": {
            "FOR": 50, "DEX": 60, "POU": 75, "CON": 50, "APP": 40, "EDU": 80, "INT": 85, "TAI": 65
        }
    }
    is_complete, missing = validate_character_sheet(incomplete_char, schema)
    assert not is_complete
    assert "ressources.sante_physique" in missing
    assert "ressources.sante_mentale" in missing
    assert "competences" in missing

    # Même schéma horreur + fiche complète
    complete_char = {
        "nom": "Harvey",
        "caracteristiques": {
            "FOR": 50, "DEX": 60, "POU": 75, "CON": 50, "APP": 40, "EDU": 80, "INT": 85, "TAI": 65
        },
        "ressources": {
            "sante_physique": {"actuels": 10, "max": 10},
            "sante_mentale": {"actuels": 75, "max": 75}
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
    # Schéma absent / vide -> doit retourner False et un message clair
    is_complete, missing = validate_character_sheet({"nom": "Test"}, None)
    assert not is_complete
    assert "schema de validation absent ou vide" in missing[0]

    # Schéma vide
    is_complete_empty_schema, missing_empty_schema = validate_character_sheet({"nom": "Test"}, {"champs_requis": []})
    assert not is_complete_empty_schema
    assert "schema de validation absent ou vide" in missing_empty_schema[0]

    is_complete_empty, missing_empty = validate_character_sheet(None, {"champs_requis": [{"chemin": "nom", "type": "string"}]})
    assert not is_complete_empty
    assert "character_data absent" in missing_empty


def test_orchestration_audit_triggered():
    # Un test où update_sheet renvoie une fiche incomplète et generate_response renvoie "CREATION_TERMINEE" (sans accents)
    # -> vérifier que audit_and_complete est appelé.
    agent = RPGAgent()

    # 1. Écriture du schéma dans Memory
    schema = {
        "champs_requis": [
            {"chemin": "nom", "type": "string"},
            {"chemin": "race", "type": "string"}
        ]
    }
    os.makedirs("Memory", exist_ok=True)
    with open("Memory/character_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f)

    agent.sheet_manager.update_sheet = mock.Mock(
        return_value={"nom": "Aragorn"}  # Incomplet, "race" manque
    )
    agent.character_creator.generate_response = mock.Mock(
        return_value="Félicitations, la création est terminée, tu es prêt pour l'aventure ! CREATION_TERMINEE"
    )
    agent._extract_and_add_resources = mock.Mock()

    # On mocke audit_and_complete pour renvoyer une fiche incomplète d'abord
    agent.sheet_manager.audit_and_complete = mock.Mock(
        return_value={"nom": "Aragorn"}  # Reste incomplet
    )

    agent.chat("Finir creation")
    assert agent.sheet_manager.audit_and_complete.called
    assert agent.game_state == "CREATION"  # Pas de transition car resté incomplet après audit

    # Deuxième cas : audit_and_complete renvoie une fiche complète
    agent.sheet_manager.audit_and_complete = mock.Mock(
        return_value={"nom": "Aragorn", "race": "Humain"}  # Devient complet
    )
    agent.chat("Finir creation")
    assert agent.game_state == "SUMMARY"  # Transition car complet après audit


def test_orchestration_transition_no_audit_when_already_complete():
    # Un test où la fiche est complète dès la mise à jour incrémentale (sans mot-clé de fin)
    # -> vérifier que la transition vers SUMMARY a lieu sans appeler l'audit.
    agent = RPGAgent()

    schema = {
        "champs_requis": [
            {"chemin": "nom", "type": "string"},
            {"chemin": "race", "type": "string"}
        ]
    }
    os.makedirs("Memory", exist_ok=True)
    with open("Memory/character_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f)

    agent.sheet_manager.update_sheet = mock.Mock(
        return_value={"nom": "Aragorn", "race": "Humain"}  # Déjà complet
    )
    agent.character_creator.generate_response = mock.Mock(
        return_value="Tu as choisi l'Humain."  # Sans mot-clé de fin
    )
    agent.sheet_manager.audit_and_complete = mock.Mock()
    agent._extract_and_add_resources = mock.Mock()

    agent.chat("Je choisis l'Humain")
    assert not agent.sheet_manager.audit_and_complete.called
    assert agent.game_state == "SUMMARY"


def test_manual_generator_agent_schema_fallback():
    # Un test où ManualGeneratorAgent's appel LLM mocké pour le schéma renvoie un JSON invalide/vide
    # -> vérifier le repli sur {"champs_requis": []} et que character_schema.json est écrit.

    # On vide la Memory au début
    schema_path = "Memory/character_schema.json"
    if os.path.exists(schema_path):
        os.remove(schema_path)

    # Mock du core_store
    mock_store = mock.Mock()
    mock_store.similarity_search.return_value = [mock.Mock(page_content="Règles de création de personnage.")]

    agent = ManualGeneratorAgent(mock_store)

    # On mocke l'appel invoke de self.chain (pour creation_manual)
    agent.chain = mock.Mock()
    agent.chain.invoke.return_value = mock.Mock(content='{"etapes": [{"etape": 1, "nom": "Etape 1", "description": "Desc"}, {"etape": 2, "nom": "Etape 2", "description": "Desc"}, {"etape": 3, "nom": "Etape 3", "description": "Desc"}, {"etape": 4, "nom": "Etape 4", "description": "Desc"}]}')

    # On mocke l'appel LLM pour le schéma afin qu'il lève une exception ou renvoie du texte vide
    agent.llm = mock.Mock()
    agent.llm.invoke.side_effect = Exception("LLM crash")

    res = agent.generate()

    assert len(res.get("etapes", [])) == 4
    assert os.path.exists(schema_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        written_schema = json.load(f)
    assert written_schema == {"champs_requis": []}
