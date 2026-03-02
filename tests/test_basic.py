import pytest
from agent import RPGAgent
import os

def test_agent_initialization():
    # On mocke l'initialisation si nécessaire ou on vérifie juste que la classe existe
    agent = RPGAgent()
    assert agent is not None
    assert hasattr(agent, 'chat')
    assert hasattr(agent, 'history')

def test_config():
    import config
    assert hasattr(config, 'OLLAMA_MODEL')
    assert hasattr(config, 'CORE_COLLECTION_NAME')
    assert hasattr(config, 'SCENARIO_COLLECTION_NAME')
