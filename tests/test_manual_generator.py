import json
import os
import shutil
import pytest
from unittest import mock
from scenario_agents import ManualGeneratorAgent, DISCOVERY_PROMPT, SCHEMA_PROMPT
import config

def setup_module():
    if os.path.exists("Memory"):
        shutil.rmtree("Memory")
    os.makedirs("Memory", exist_ok=True)


def test_manual_generator_below_threshold():
    """
    Core store with total text under CORE_FULLTEXT_THRESHOLD_CHARS
    -> no call to DISCOVERY_PROMPT, no call to similarity_search;
    the full text is passed directly to manual generation prompt.
    """
    mock_store = mock.Mock()
    mock_store.get.return_value = {
        "documents": ["Règles de jeu courtes."],
        "metadatas": [{"page": 1}]
    }

    agent = ManualGeneratorAgent(mock_store)

    # Patch config.CORE_FULLTEXT_THRESHOLD_CHARS to 1000 so "Règles de jeu courtes." (22 chars) < 1000
    with mock.patch.object(agent.llm.__class__, "invoke", side_effect=[
        mock.Mock(content='{"steps": [{"step": 1, "name": "Etape 1", "description": "Procedure"}]}'),
        mock.Mock(content='{"required_fields": [{"path": "name", "type": "string"}]}')
    ]):
        with mock.patch.object(config, "CORE_FULLTEXT_THRESHOLD_CHARS", 1000):
            res = agent.generate()

    assert res == {"steps": [{"step": 1, "name": "Etape 1", "description": "Procedure"}]}

    # Verify similarity_search was NOT called
    assert not mock_store.similarity_search.called


def test_manual_generator_above_threshold_success():
    """
    Core store exceeding threshold, DISCOVERY_PROMPT mocked returning {"components": ["Type", "Descripteur", "Focus"]}
    -> verify targeted RAG queries are correctly built from these terms.
    """
    mock_store = mock.Mock()
    # Ensure total size exceeds threshold
    mock_store.get.return_value = {
        "documents": ["Règles longues. " * 5000], # ~75000 chars > 40000
        "metadatas": [{"page": 1}]
    }

    # Simulate similarity_search results
    mock_store.similarity_search.return_value = [
        mock.Mock(page_content="Extrait RAG.")
    ]

    agent = ManualGeneratorAgent(mock_store)

    with mock.patch.object(agent.llm.__class__, "invoke", side_effect=[
        mock.Mock(content='{"components": ["Type", "Descripteur", "Focus"]}'),
        mock.Mock(content='{"steps": []}'),
        mock.Mock(content='{"required_fields": [{"path": "name", "type": "string"}]}')
    ]):
        with mock.patch.object(config, "CORE_FULLTEXT_THRESHOLD_CHARS", 40000):
            agent.generate()

    # Verify similarity_search was called with discovered terms
    called_queries = [call[0][0] for call in mock_store.similarity_search.call_args_list]

    assert "création de personnage, character creation, comment créer un personnage" in called_queries
    assert "construire un personnage, personnage joueur, feuille de personnage" in called_queries

    assert "Type, création de personnage" in called_queries
    assert "Descripteur, création de personnage" in called_queries
    assert "Focus, création de personnage" in called_queries


def test_manual_generator_above_threshold_invalid_json_fallback():
    """
    DISCOVERY_PROMPT mocked returning invalid/empty JSON -> fall back on default generic queries, no crash.
    """
    mock_store = mock.Mock()
    mock_store.get.return_value = {
        "documents": ["Règles longues. " * 5000],
        "metadatas": [{"page": 1}]
    }
    mock_store.similarity_search.return_value = [
        mock.Mock(page_content="Extrait RAG.")
    ]

    agent = ManualGeneratorAgent(mock_store)

    with mock.patch.object(agent.llm.__class__, "invoke", side_effect=[
        mock.Mock(content='JSON invalide { ...'),
        mock.Mock(content='{"steps": []}'),
        mock.Mock(content='{"required_fields": []}')
    ]):
        with mock.patch.object(config, "CORE_FULLTEXT_THRESHOLD_CHARS", 40000):
            res = agent.generate()

    assert res == {"steps": []}

    called_queries = [call[0][0] for call in mock_store.similarity_search.call_args_list]

    assert "création de personnage, character creation, comment créer un personnage" in called_queries
    assert "création de personnage, caractéristiques, capacités spéciales" in called_queries
    assert "équipement, ressources de départ, progression du personnage" in called_queries


def test_no_dnd_bias_in_prompts():
    """
    Verify that the final system prompt sent for manual generation does not contain
    pre-conceived D&D terms in hardcoded form.
    """
    mock_store = mock.Mock()
    agent = ManualGeneratorAgent(mock_store)

    # Retrieve system prompt
    system_prompt = agent.prompt.messages[0].prompt.template

    forbidden_terms = ["elfe", "nain", "guerrier"]

    for term in forbidden_terms:
        assert term not in system_prompt.lower(), f"Forbidden term '{term}' is still present in ManualGeneratorAgent prompt."


def test_manual_generator_discovery_fallback_queries():
    """
    Verifies that with fewer than MIN_COMPOSANTES_DECOUVERTES components,
    the fallback queries are correctly added.
    """
    mock_store = mock.Mock()
    mock_store.get.return_value = {
        "documents": ["Règles longues. " * 5000],
        "metadatas": [{"page": 1}]
    }
    mock_store.similarity_search.return_value = [
        mock.Mock(page_content="Extrait RAG.")
    ]

    agent = ManualGeneratorAgent(mock_store)

    with mock.patch.object(agent.llm.__class__, "invoke", side_effect=[
        mock.Mock(content='{"components": ["Concept1", "Concept2"]}'),
        mock.Mock(content='{"steps": []}'),
        mock.Mock(content='{"required_fields": []}')
    ]):
        with mock.patch.object(config, "CORE_FULLTEXT_THRESHOLD_CHARS", 40000):
            agent.generate()

    called_queries = [call[0][0] for call in mock_store.similarity_search.call_args_list]

    assert "Concept1, création de personnage" in called_queries
    assert "Concept2, création de personnage" in called_queries

    assert "étapes numérotées de création de personnage, procédure complète" in called_queries
    assert "caractéristiques, points de vie, équipement, capacités de classe" in called_queries


def test_prompts_contain_french_dialogue_instruction():
    """
    Verify that the system prompts translated to English explicitly instruct
    the LLM to respond to the player in French.
    """
    mock_store = mock.Mock()
    agent = ManualGeneratorAgent(mock_store)
    system_prompt = agent.prompt.messages[0].prompt.template
    assert "in french" in system_prompt.lower()
