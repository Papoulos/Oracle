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
    Core store dont le texte complet tient sous CORE_FULLTEXT_THRESHOLD_CHARS
    -> aucun appel à DISCOVERY_PROMPT, aucun appel à similarity_search;
    le texte complet est passé tel quel au prompt de génération du manuel.
    """
    mock_store = mock.Mock()
    mock_store.get.return_value = {
        "documents": ["Règles de jeu courtes."],
        "metadatas": [{"page": 1}]
    }

    agent = ManualGeneratorAgent(mock_store)

    # On mocke l'appel LLM (chain et llm.invoke)
    agent.chain = mock.Mock()
    agent.chain.invoke.return_value = mock.Mock(content='{"etapes": [{"etape": 1, "nom": "Etape 1", "description": "Procedure"}]}')

    # On patche config.CORE_FULLTEXT_THRESHOLD_CHARS à 1000 pour être sûr que "Règles de jeu courtes." (22 chars) < 1000
    with mock.patch.object(agent.llm.__class__, "invoke", return_value=mock.Mock(content='{"champs_requis": [{"chemin": "nom", "type": "string"}]}')):
        with mock.patch.object(config, "CORE_FULLTEXT_THRESHOLD_CHARS", 1000):
            res = agent.generate()

    assert res == {"etapes": [{"etape": 1, "nom": "Etape 1", "description": "Procedure"}]}

    # Vérifier que similarity_search n'a PAS été appelé
    assert not mock_store.similarity_search.called

    # Vérifier que le texte complet a été passé à la chaine
    agent.chain.invoke.assert_called_once_with({"context": "Règles de jeu courtes."})


def test_manual_generator_above_threshold_success():
    """
    Core store dépassant le seuil, DISCOVERY_PROMPT mocké renvoyant {"composantes": ["Type", "Descripteur", "Focus"]}
    -> vérifier que les requêtes RAG de la phase d'extraction ciblée sont bien construites à partir de ces termes,
    pas des 7 requêtes fixes de l'ancienne version.
    """
    mock_store = mock.Mock()
    # On s'assure que la taille totale dépasse le seuil
    mock_store.get.return_value = {
        "documents": ["Règles longues. " * 5000], # ~75000 chars > 40000
        "metadatas": [{"page": 1}]
    }

    # Simulation des résultats de similarity_search
    # Premier appel pour discovery_context
    # Deuxième appel pour les requêtes ciblées
    mock_store.similarity_search.return_value = [
        mock.Mock(page_content="Extrait RAG.")
    ]

    agent = ManualGeneratorAgent(mock_store)

    # Mock de agent.chain (qui est DISCOVERY_PROMPT / self.prompt | self.llm)
    agent.chain = mock.Mock()
    agent.chain.invoke.return_value = mock.Mock(content='{"etapes": []}')

    # Mock de self.llm.invoke via patch.object sur la classe de l'llm
    # Le premier appel de llm est pour discovery_chain (DISCOVERY_PROMPT | self.llm) -> doit renvoyer {"composantes": ["Type", "Descripteur", "Focus"]}
    # Le deuxième appel est pour schema_chain (SCHEMA_PROMPT | self.llm) -> on renvoie un schéma minimal
    with mock.patch.object(agent.llm.__class__, "invoke", side_effect=[
        mock.Mock(content='{"composantes": ["Type", "Descripteur", "Focus"]}'),
        mock.Mock(content='{"champs_requis": [{"chemin": "nom", "type": "string"}]}')
    ]):
        with mock.patch.object(config, "CORE_FULLTEXT_THRESHOLD_CHARS", 40000):
            agent.generate()

    # Vérifier que similarity_search a été appelé avec les termes de découverte
    # Les requêtes construites doivent être :
    # "Type, création de personnage"
    # "Descripteur, création de personnage"
    # "Focus, création de personnage"
    called_queries = [call[0][0] for call in mock_store.similarity_search.call_args_list]

    # Les deux premières requêtes sont celles de discovery (création de personnage...)
    assert "création de personnage, character creation, comment créer un personnage" in called_queries
    assert "construire un personnage, personnage joueur, feuille de personnage" in called_queries

    # Les requêtes suivantes doivent contenir les composantes découvertes
    assert "Type, création de personnage" in called_queries
    assert "Descripteur, création de personnage" in called_queries
    assert "Focus, création de personnage" in called_queries


def test_manual_generator_above_threshold_invalid_json_fallback():
    """
    DISCOVERY_PROMPT mocké renvoyant un JSON invalide/vide -> repli sur les requêtes génériques par défaut, pas de crash.
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

    agent.chain = mock.Mock()
    agent.chain.invoke.return_value = mock.Mock(content='{"etapes": []}')

    # Le premier appel renvoie un JSON invalide
    with mock.patch.object(agent.llm.__class__, "invoke", side_effect=[
        mock.Mock(content='JSON invalide { ...'),
        mock.Mock(content='{"champs_requis": []}')
    ]):
        with mock.patch.object(config, "CORE_FULLTEXT_THRESHOLD_CHARS", 40000):
            res = agent.generate()

    assert res == {"etapes": []}

    called_queries = [call[0][0] for call in mock_store.similarity_search.call_args_list]

    # Requêtes de découverte
    assert "création de personnage, character creation, comment créer un personnage" in called_queries

    # Requêtes génériques de repli
    assert "création de personnage, caractéristiques, capacités spéciales" in called_queries
    assert "équipement, ressources de départ, progression du personnage" in called_queries


def test_no_dnd_bias_in_prompts():
    """
    Vérifier que le prompt système final envoyé à la génération du manuel ne contient plus
    les chaînes "Race", "Classe", "PV/CA", "Elfe", "Nain", "Guerrier" (insensible à la casse) en dur.
    """
    mock_store = mock.Mock()
    agent = ManualGeneratorAgent(mock_store)

    # Récupérer le prompt système
    system_prompt = agent.prompt.messages[0].prompt.template

    # Liste des termes D&D interdits
    forbidden_terms = ["race", "classe", "pv/ca", "elfe", "nain", "guerrier"]

    for term in forbidden_terms:
        assert term not in system_prompt.lower(), f"Le terme interdit '{term}' est toujours présent (même en minuscules) dans le prompt de ManualGeneratorAgent."
