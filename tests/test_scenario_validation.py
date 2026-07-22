import os
import json
import pytest
from unittest import mock
from validation import validate_scenario_structure
from agent import RPGAgent
import config

@pytest.fixture
def clean_scenario_data():
    return {
        "metadata": {
            "titre": "La Larme de l'Oubli",
            "pitch_global": "Un scénario d'exploration fantastique.",
            "scene_initiale": "SCENE_01_AUBERGE"
        },
        "macro_structure": [
            {
                "id_acte": "ACTE_1",
                "titre": "Le Départ",
                "condition_entree": "Création terminée",
                "condition_validation": "Le joueur quitte l'auberge",
                "scenes_incluses": ["SCENE_01_AUBERGE", "SCENE_02_ROUTE"]
            }
        ],
        "horloges_globales": [
            {
                "nom": "L'Orage",
                "declencheur": "Le temps passe",
                "consequence": "La foudre tombe",
                "seuil": 6
            }
        ],
        "entites": {
            "pnj": [
                {
                    "id": "MAITRE_ELROND",
                    "nom_complet": "Maître Elrond",
                    "localisation_habituelle": "LIEU_FONDCOMBE",
                    "agenda_et_motivation": "Aider",
                    "peurs_et_faiblesses": "Aucune",
                    "attitude_initiale": "Neutre",
                    "stats_et_capacites": "Fort"
                }
            ],
            "lieux": [
                {
                    "id": "LIEU_FONDCOMBE",
                    "nom_complet": "Fondcombe",
                    "ambiance_sensorielle": "Calme",
                    "elements_interactifs": "Livre"
                }
            ]
        },
        "noeuds_sceniques": [
            {
                "id_scene": "SCENE_01_AUBERGE",
                "acte_rattache_id": "ACTE_1",
                "lieu_rattache_id": "LIEU_FONDCOMBE",
                "titre": "Rencontre à l'auberge",
                "pnj_presents": ["MAITRE_ELROND"],
                "objectif_mj": "Présenter Elrond",
                "condition_resolution": "Le PJ parle à Elrond",
                "limites_et_regles_locales": "Pas de combat",
                "defis_et_rencontres": [],
                "sorties_logiques": [
                    {
                        "action_ou_direction": "Prendre la route",
                        "destination_scene_id": "SCENE_02_ROUTE"
                    }
                ]
            },
            {
                "id_scene": "SCENE_02_ROUTE",
                "acte_rattache_id": "ACTE_1",
                "lieu_rattache_id": "LIEU_FONDCOMBE",
                "titre": "Sur la route",
                "pnj_presents": [],
                "objectif_mj": "Avancer",
                "condition_resolution": "La route se termine",
                "limites_et_regles_locales": "",
                "defis_et_rencontres": [],
                "sorties_logiques": []
            }
        ]
    }

def test_validation_propre(clean_scenario_data):
    # Case 1: Extractions propre -> 0 warnings, 0 errors
    data, warnings, errors = validate_scenario_structure(clean_scenario_data)
    assert len(warnings) == 0
    assert len(errors) == 0

def test_validation_idempotente(clean_scenario_data):
    # Verify idempotency
    data, warnings, errors = validate_scenario_structure(clean_scenario_data)
    data2, warnings2, errors2 = validate_scenario_structure(data)
    assert len(warnings2) == 0
    assert len(errors2) == 0

def test_validation_casse_et_reparable(clean_scenario_data):
    # Case 2: Version sabotée mais réparable
    saboted = clean_scenario_data
    # 1. Lieu inconnu sur scène
    saboted["noeuds_sceniques"][0]["lieu_rattache_id"] = "LIEU_INCONNU"
    # 2. PNJ orphelin
    saboted["noeuds_sceniques"][0]["pnj_presents"].append("PNJ_ORPHELIN")
    # 3. Sortie logique vers scène inconnue
    saboted["noeuds_sceniques"][0]["sorties_logiques"].append({
        "action_ou_direction": "Inconnu",
        "destination_scene_id": "SCENE_INCONNUE"
    })
    # 4. Incohérence bidirectionnelle : acte ne liste pas la scène
    saboted["macro_structure"][0]["scenes_incluses"].remove("SCENE_02_ROUTE")
    # 5. Scene_incluse orpheline dans l'acte
    saboted["macro_structure"][0]["scenes_incluses"].append("SCENE_ORPHELINE")
    # 6. Seuil de l'horloge manquant
    del saboted["horloges_globales"][0]["seuil"]

    data, warnings, errors = validate_scenario_structure(saboted)

    # All of these should be fixed and yield warnings, not errors
    assert len(errors) == 0
    assert len(warnings) > 0

    # Assertions on corrections
    assert data["noeuds_sceniques"][0]["lieu_rattache_id"] is None
    assert "PNJ_ORPHELIN" not in data["noeuds_sceniques"][0]["pnj_presents"]
    assert len(data["noeuds_sceniques"][0]["sorties_logiques"]) == 1 # unknown output removed
    assert "SCENE_02_ROUTE" in data["macro_structure"][0]["scenes_incluses"] # bidirectionally repaired
    assert "SCENE_ORPHELINE" not in data["macro_structure"][0]["scenes_incluses"] # orphan removed
    assert data["horloges_globales"][0]["seuil"] == 6 # fallback to default

def test_validation_erreurs_bloquantes(clean_scenario_data):
    # Case 3: Blocking errors that cannot be fixed
    # 1. Missing condition_resolution
    del clean_scenario_data["noeuds_sceniques"][0]["condition_resolution"]
    # 2. Invalide acte_rattache_id
    clean_scenario_data["noeuds_sceniques"][0]["acte_rattache_id"] = "ACTE_INEXISTANT"

    data, warnings, errors = validate_scenario_structure(clean_scenario_data)
    assert len(errors) == 2
    assert any("condition_resolution manquant" in e for e in errors)
    assert any("acte_rattache_id" in e for e in errors)

def test_validation_doublons_et_localisation_pnj(clean_scenario_data):
    # Test new rules added: duplicates detection and pnj localisation_habituelle validation
    data = clean_scenario_data
    # Duplicate PNJ ID
    data["entites"]["pnj"].append({
        "id": "MAITRE_ELROND",
        "nom_complet": "Autre Elrond",
        "localisation_habituelle": "LIEU_INCONNU",
        "agenda_et_motivation": "Rien",
        "peurs_et_faiblesses": "Aucune",
        "attitude_initiale": "Amical",
        "stats_et_capacites": "Inconnu"
    })

    corrected, warnings, errors = validate_scenario_structure(data)
    assert len(errors) == 0
    assert any("id dupliqué 'MAITRE_ELROND'" in w for w in warnings)
    assert any("localisation_habituelle 'LIEU_INCONNU' inconnue -> mise à null." in w for w in warnings)

    assert corrected["entites"]["pnj"][0]["localisation_habituelle"] == "LIEU_FONDCOMBE"
    assert corrected["entites"]["pnj"][1]["localisation_habituelle"] is None

def test_setup_world_json_scenarios(tmp_path, monkeypatch):
    # Setup test configuration environment paths
    monkeypatch.setattr(config, "SCENARIO_DATA_PATH", str(tmp_path))

    agent = RPGAgent()
    agent.scenario_extractor_agent = mock.Mock()
    agent._check_collections = mock.Mock(return_value=True)

    # 1. Scenario with zero JSON files -> calls ScenarioExtractorAgent
    # Prepare some mock generation return value
    mock_sc_structure = {
        "metadata": {"titre": "Aventure", "pitch_global": "Un pitch", "scene_initiale": "SCENE_01"},
        "macro_structure": [{"id_acte": "ACTE_1", "titre": "Acte", "scenes_incluses": ["SCENE_01"]}],
        "horloges_globales": [],
        "entites": {"pnj": [], "lieux": []},
        "noeuds_sceniques": [{"id_scene": "SCENE_01", "acte_rattache_id": "ACTE_1", "condition_resolution": "PJ sort"}]
    }
    agent.scenario_extractor_agent.generate = mock.Mock(return_value=mock_sc_structure)

    # Clean Memory/ directory if present
    for f in ["character.json", "scenario_structure.json", "progression.json", "scenario.json"]:
        p = os.path.join("Memory", f)
        if os.path.exists(p):
            os.remove(p)

    agent.setup_world()
    assert agent.scenario_extractor_agent.generate.called
    assert agent.scenario_structure["metadata"]["titre"] == "Aventure"

    # 2. Scenario with exactly one JSON file -> loads directly, never calls ScenarioExtractorAgent
    agent.scenario_extractor_agent.generate.reset_mock()
    one_json = tmp_path / "scenario_structure.json"
    with open(one_json, "w", encoding="utf-8") as f:
        json.dump(mock_sc_structure, f)

    agent.setup_world()
    assert not agent.scenario_extractor_agent.generate.called
    assert agent.scenario_structure["metadata"]["titre"] == "Aventure"

    # 3. Scenario with multiple JSON files -> raises ValueError
    another_json = tmp_path / "another_scenario.json"
    with open(another_json, "w", encoding="utf-8") as f:
        json.dump(mock_sc_structure, f)

    with pytest.raises(ValueError, match="Plusieurs fichiers JSON de scénario trouvés"):
        agent.setup_world()

def test_full_text_below_threshold_avoid_similarity_search():
    # Setup scenario store mock
    store = mock.Mock()
    # Mocking get returns few documents
    store.get = mock.Mock(return_value={
        "documents": ["Mon petit scénario court sur une page."],
        "metadatas": [{"page": 0}]
    })
    store.similarity_search = mock.Mock(return_value=[])

    from scenario_agents import ScenarioExtractorAgent
    extractor = ScenarioExtractorAgent(store)

    # Set threshold to a value that is higher than length of the documents (length = 37)
    import config
    original_threshold = config.SCENARIO_FULLTEXT_THRESHOLD_CHARS
    config.SCENARIO_FULLTEXT_THRESHOLD_CHARS = 100

    # Mock LLM and extract entities
    mock_res = mock.Mock()
    mock_res.content = "{}"
    extractor.llm = mock.Mock()
    extractor.llm.invoke = mock.Mock(return_value=mock_res)

    extractor._extract_entites(log=lambda x: None)

    # similarity_search should NOT be called because length of text is 37 <= 100
    assert not store.similarity_search.called

    # Set threshold to a very low value so similarity_search is called (length = 37 > 10)
    config.SCENARIO_FULLTEXT_THRESHOLD_CHARS = 10
    extractor._extract_entites(log=lambda x: None)
    assert store.similarity_search.called

    # Restore threshold
    config.SCENARIO_FULLTEXT_THRESHOLD_CHARS = original_threshold
