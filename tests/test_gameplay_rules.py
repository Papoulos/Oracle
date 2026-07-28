import os
import json
import pytest
from unittest import mock
from game_state_engine import GameStateEngine
from agent import RPGAgent
from scenario_agents import GameplayRulesAgent

@pytest.fixture
def temp_character_file(tmp_path):
    char_file = tmp_path / "character.json"
    data = {
        "nom": "Test Hero",
        "niveau": 1,
        "xp": 0,
        "xp_prochain_niveau": 1000,
        "pv": 10,
        "ressources": {
            "points_de_vie": {"actuels": 4, "max": 10},
            "sorts_par_jour": {
                "niveau_1": {"restants": 0, "max": 2}
            },
            "points_de_rage": {"restants": 1, "max": 2}
        }
    }
    char_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(char_file)

@pytest.fixture
def mock_recovery_rules_file():
    path = "Memory/recovery_rules.json"
    backup = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                backup = f.read()
        except Exception:
            pass

    rules = {
        "paliers_repos": [
            {
                "id": "repos_long",
                "nom": "Repos Long",
                "declencheurs_texte": ["dormir dans un lit", "bivouac sécurisé"],
                "effets": [
                    {"ressource": "ressources.points_de_vie", "action": "restaurer_complet", "valeur": None},
                    {"ressource": "ressources.sorts_par_jour", "action": "restaurer_complet", "valeur": None}
                ]
            },
            {
                "id": "repos_court",
                "nom": "Repos Court",
                "declencheurs_texte": ["récupération de 10 minutes", "petite pause"],
                "effets": [
                    {"ressource": "ressources.points_de_vie", "action": "restaurer_pourcentage", "valeur": 25}
                ]
            }
        ]
    }
    os.makedirs("Memory", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=4, ensure_ascii=False)

    yield path

    if backup is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(backup)
    elif os.path.exists(path):
        os.remove(path)


@pytest.fixture
def mock_action_catalog_file():
    path = "Memory/action_catalog.json"
    backup = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                backup = f.read()
        except Exception:
            pass

    catalog = {
        "actions_courantes": [
            {
                "nom": "Attaquer",
                "declencheurs": ["attaque", "frapper"],
                "resolution": "Jet de combat",
                "en_cas_de_succes": "Ennemi blessé",
                "en_cas_d_echec": "Rien"
            }
        ]
    }
    os.makedirs("Memory", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=4, ensure_ascii=False)

    yield path

    if backup is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(backup)
    elif os.path.exists(path):
        os.remove(path)


def test_gse_rest_repos_long_dynamic(temp_character_file, mock_recovery_rules_file):
    gse = GameStateEngine(temp_character_file)
    # verify initial conditions
    hp, hp_max = gse.get_hp()
    assert hp == 4
    assert gse.get_resource("sorts_par_jour", level=1)[0] == 0

    # Call rest with mock long rest
    res = gse.rest("repos_long")
    assert res.success is True
    assert gse.get_hp()[0] == 10
    assert gse.get_resource("sorts_par_jour", level=1)[0] == 2
    assert gse.state["pv"] == 10


def test_gse_rest_repos_court_dynamic(temp_character_file, mock_recovery_rules_file):
    gse = GameStateEngine(temp_character_file)
    # verify initial conditions: 4 hp, max is 10. 25% of 10 is 2. So 4 + 2 = 6 hp.
    res = gse.rest("repos_court")
    assert res.success is True
    assert gse.get_hp()[0] == 6
    assert gse.state["pv"] == 6


def test_gse_rest_palier_inexistant(temp_character_file, mock_recovery_rules_file):
    gse = GameStateEngine(temp_character_file)
    res = gse.rest("palier_inexistant")
    assert res.success is False
    assert "inconnu" in res.message


def test_gse_detect_action_type_dynamic(temp_character_file, mock_recovery_rules_file):
    gse = GameStateEngine(temp_character_file)
    # Terms non-D&D listed in mock_recovery_rules_file for "repos_court"
    assert gse.detect_action_type("Je prends une petite pause") == "rest:repos_court"
    assert gse.detect_action_type("Je décide de dormir dans un lit") == "rest:repos_long"


def test_agent_chat_covered_by_catalog(mock_action_catalog_file):
    with mock.patch("agent.RPGAgent.get_core_context") as mock_get_core:
        agent = RPGAgent()
        agent.game_state = "ADVENTURE"
        agent.chronicle_data = {"summary": "L'aventure commence."}
        agent.npcs_data = []
        agent.current_scene_id = "SCENE_01"
        agent.scenario_structure = {
            "metadata": {"titre": "La Quête", "pitch_global": "...", "scene_initiale": "SCENE_01"},
            "entites": {"pnj": [], "lieux": []},
            "macro_structure": [
                {"id_acte": "ACTE_1", "titre": "Acte 1", "scenes_incluses": ["SCENE_01"]}
            ],
            "noeuds_sceniques": [
                {
                    "id_scene": "SCENE_01",
                    "acte_rattache_id": "ACTE_1",
                    "titre": "Scène 1",
                    "objectif_mj": "...",
                    "condition_resolution": "Le PJ avance",
                    "sorties_logiques": []
                }
            ],
            "horloges_globales": []
        }
        agent._build_lookups()
        agent.progression = {
            "acte_courant": "ACTE_1",
            "scene_courante": "SCENE_01",
            "scenes_resolues": [],
            "scenes_contournees": [],
            "horloges": {},
            "ecarts_notables": []
        }
        agent.character_data = {
            "nom": "Hero",
            "niveau": 1,
            "xp": 0,
            "ressources": {"points_de_vie": {"actuels": 10, "max": 10}}
        }

        # Mock prompt-aware LLM invoke
        def invoke_mock(prompt, **kwargs):
            p_str = str(prompt)
            if "Tu es un arbitre de JDR. Analyse l'action du joueur" in p_str:
                return mock.Mock(content=json.dumps({"consomme": False}))
            elif "Analyse l'action du joueur par rapport à la scène courante" in p_str:
                return mock.Mock(content=json.dumps({"categorie": "improvisation", "scene_suivante": None}))
            elif "CATALOGUE D'ACTIONS COURANTES DE CE SYSTÈME" in p_str:
                return mock.Mock(content=json.dumps({
                    "couvert_par_catalogue": True,
                    "need_roll": True,
                    "stat": "combat",
                    "bonus": 2,
                    "calculation_breakdown": "+2 dague",
                    "dc": 12,
                    "reason": "Attaque à la dague",
                    "mechanical_decision": {"action": None, "amount": None}
                }))
            else:
                return mock.Mock(content="{}")

        agent.llm = mock.Mock()
        agent.llm.invoke = mock.Mock(side_effect=invoke_mock)
        agent.narrator.generate_response = mock.Mock(return_value="Vous attaquez avec succès !")
        agent.sheet_manager.update_sheet = mock.Mock(return_value={})
        agent.update_chronicle = mock.Mock()

        response = agent.chat("J'attaque le gobelin")

        # Verify we didn't search the core rules store via similarity search
        mock_get_core.assert_not_called()
        assert "Jet de combat" in response
        assert "12" in response


def test_agent_chat_not_covered_by_catalog(mock_action_catalog_file):
    with mock.patch("agent.RPGAgent.get_core_context", return_value="Règles de crochetage...") as mock_get_core:
        agent = RPGAgent()
        agent.game_state = "ADVENTURE"
        agent.chronicle_data = {"summary": "L'aventure commence."}
        agent.npcs_data = []
        agent.current_scene_id = "SCENE_01"
        agent.scenario_structure = {
            "metadata": {"titre": "La Quête", "pitch_global": "...", "scene_initiale": "SCENE_01"},
            "entites": {"pnj": [], "lieux": []},
            "macro_structure": [
                {"id_acte": "ACTE_1", "titre": "Acte 1", "scenes_incluses": ["SCENE_01"]}
            ],
            "noeuds_sceniques": [
                {
                    "id_scene": "SCENE_01",
                    "acte_rattache_id": "ACTE_1",
                    "titre": "Scène 1",
                    "objectif_mj": "...",
                    "condition_resolution": "Le PJ avance",
                    "sorties_logiques": []
                }
            ],
            "horloges_globales": []
        }
        agent._build_lookups()
        agent.progression = {
            "acte_courant": "ACTE_1",
            "scene_courante": "SCENE_01",
            "scenes_resolues": [],
            "scenes_contournees": [],
            "horloges": {},
            "ecarts_notables": []
        }
        agent.character_data = {
            "nom": "Hero",
            "niveau": 1,
            "xp": 0,
            "ressources": {"points_de_vie": {"actuels": 10, "max": 10}}
        }

        # Mock prompt-aware LLM invoke
        def invoke_mock(prompt, **kwargs):
            p_str = str(prompt)
            if "Tu es un arbitre de JDR. Analyse l'action du joueur" in p_str:
                return mock.Mock(content=json.dumps({"consomme": False}))
            elif "Analyse l'action du joueur par rapport à la scène courante" in p_str:
                return mock.Mock(content=json.dumps({"categorie": "improvisation", "scene_suivante": None}))
            elif "CATALOGUE D'ACTIONS COURANTES DE CE SYSTÈME" in p_str:
                return mock.Mock(content=json.dumps({"couvert_par_catalogue": False}))
            elif "Basé sur les RÈGLES du CODEX suivantes" in p_str:
                return mock.Mock(content=json.dumps({
                    "need_roll": True,
                    "stat": "crochetage",
                    "bonus": 3,
                    "calculation_breakdown": "+3 Dextérité",
                    "dc": 15,
                    "reason": "Crochetage de serrure complexe",
                    "mechanical_decision": {"action": None, "amount": None}
                }))
            else:
                return mock.Mock(content="{}")

        agent.llm = mock.Mock()
        agent.llm.invoke = mock.Mock(side_effect=invoke_mock)
        agent.narrator.generate_response = mock.Mock(return_value="Vous crochetez la serrure !")
        agent.sheet_manager.update_sheet = mock.Mock(return_value={})
        agent.update_chronicle = mock.Mock()

        response = agent.chat("Je crochette la porte")

        # Core RAG context search SHOULD be called
        mock_get_core.assert_called_once_with("Je crochette la porte", k=mock.ANY)
        assert "Jet de crochetage" in response
        assert "15" in response


def test_agent_chat_empty_catalog():
    # Empty or missing catalog
    if os.path.exists("Memory/action_catalog.json"):
        os.remove("Memory/action_catalog.json")

    with mock.patch("agent.RPGAgent.get_core_context", return_value="Règles d'escalade...") as mock_get_core:
        agent = RPGAgent()
        agent.game_state = "ADVENTURE"
        agent.chronicle_data = {"summary": "L'aventure commence."}
        agent.npcs_data = []
        agent.current_scene_id = "SCENE_01"
        agent.scenario_structure = {
            "metadata": {"titre": "La Quête", "pitch_global": "...", "scene_initiale": "SCENE_01"},
            "entites": {"pnj": [], "lieux": []},
            "macro_structure": [
                {"id_acte": "ACTE_1", "titre": "Acte 1", "scenes_incluses": ["SCENE_01"]}
            ],
            "noeuds_sceniques": [
                {
                    "id_scene": "SCENE_01",
                    "acte_rattache_id": "ACTE_1",
                    "titre": "Scène 1",
                    "objectif_mj": "...",
                    "condition_resolution": "Le PJ avance",
                    "sorties_logiques": []
                }
            ],
            "horloges_globales": []
        }
        agent._build_lookups()
        agent.progression = {
            "acte_courant": "ACTE_1",
            "scene_courante": "SCENE_01",
            "scenes_resolues": [],
            "scenes_contournees": [],
            "horloges": {},
            "ecarts_notables": []
        }
        agent.character_data = {
            "nom": "Hero",
            "niveau": 1,
            "xp": 0,
            "ressources": {"points_de_vie": {"actuels": 10, "max": 10}}
        }

        # Mock prompt-aware LLM invoke
        def invoke_mock(prompt, **kwargs):
            p_str = str(prompt)
            if "Tu es un arbitre de JDR. Analyse l'action du joueur" in p_str:
                return mock.Mock(content=json.dumps({"consomme": False}))
            elif "Analyse l'action du joueur par rapport à la scène courante" in p_str:
                return mock.Mock(content=json.dumps({"categorie": "improvisation", "scene_suivante": None}))
            elif "Basé sur les RÈGLES du CODEX suivantes" in p_str:
                return mock.Mock(content=json.dumps({
                    "need_roll": True,
                    "stat": "athlétisme",
                    "bonus": 1,
                    "calculation_breakdown": "+1 Force",
                    "dc": 10,
                    "reason": "Escalade de la falaise",
                    "mechanical_decision": {"action": None, "amount": None}
                }))
            else:
                return mock.Mock(content="{}")

        agent.llm = mock.Mock()
        agent.llm.invoke = mock.Mock(side_effect=invoke_mock)
        agent.narrator.generate_response = mock.Mock(return_value="Vous escaladez la falaise !")
        agent.sheet_manager.update_sheet = mock.Mock(return_value={})
        agent.update_chronicle = mock.Mock()

        response = agent.chat("Je grimpe sur la falaise")

        # Direct fallback to RAG
        mock_get_core.assert_called_once_with("Je grimpe sur la falaise", k=mock.ANY)
        # LLM called for: check resources, scene classification, and core rules prompt
        assert agent.llm.invoke.call_count == 3
        assert "Jet de athlétisme" in response
