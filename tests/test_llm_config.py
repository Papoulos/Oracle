import pytest
from unittest.mock import patch, MagicMock
import config
from base_utils import get_llm, BaseAgent
from pack_extractor import was_truncated, response_text, PackExtractorAgent

def test_parse_tristate():
    assert config.parse_tristate("true") is True
    assert config.parse_tristate("TRUE") is True
    assert config.parse_tristate("false") is False
    assert config.parse_tristate("FALSE") is False
    assert config.parse_tristate("default") is None
    assert config.parse_tristate("") is None
    assert config.parse_tristate(None) is None
    assert config.parse_tristate("n'importe quoi") is None

@patch('base_utils.ChatOllama')
@patch('base_utils.config.LLM_PROVIDER', 'ollama')
def test_get_llm_with_options(mock_chat_ollama):
    get_llm("test_model", 0.7, num_ctx=4000, num_predict=1000, reasoning=True, keep_alive="1h")

    mock_chat_ollama.assert_called_once()
    kwargs = mock_chat_ollama.call_args.kwargs
    assert kwargs["model"] == "test_model"
    assert kwargs["temperature"] == 0.7
    assert kwargs["num_ctx"] == 4000
    assert kwargs["num_predict"] == 1000
    assert kwargs["reasoning"] is True
    assert kwargs["keep_alive"] == "1h"

@patch('base_utils.ChatOllama')
@patch('base_utils.config.LLM_PROVIDER', 'ollama')
def test_get_llm_defaults(mock_chat_ollama):
    get_llm("test_model", 0.7)

    mock_chat_ollama.assert_called_once()
    kwargs = mock_chat_ollama.call_args.kwargs
    assert kwargs["num_ctx"] == 16384
    assert kwargs["num_predict"] == 2048
    assert "reasoning" not in kwargs
    assert "keep_alive" not in kwargs

def test_was_truncated():
    resp1 = MagicMock()
    resp1.response_metadata = {"done_reason": "length"}
    assert was_truncated(resp1) is True

    resp2 = MagicMock()
    resp2.response_metadata = {"finish_reason": "MAX_TOKENS"}
    assert was_truncated(resp2) is True

    resp3 = MagicMock()
    resp3.response_metadata = {"stop_reason": "max_tokens"}
    assert was_truncated(resp3) is True

    resp4 = MagicMock()
    resp4.response_metadata = {"done_reason": "stop"}
    assert was_truncated(resp4) is False

    resp5 = MagicMock()
    del resp5.response_metadata
    assert was_truncated(resp5) is False

def test_response_text():
    resp_str = MagicMock()
    resp_str.content = "hello world"
    assert response_text(resp_str) == "hello world"

    resp_list = MagicMock()
    resp_list.content = [
        {"type": "text", "text": "hello "},
        {"type": "text", "text": "world"}
    ]
    assert response_text(resp_list) == "hello world"

@patch('base_utils.config.LLM_PROVIDER', 'ollama')
@patch('base_utils.config.PACK_REASONING', True)
@patch('base_utils.config.PACK_REASONING_RAW', 'true')
@patch('base_utils.config.LLM_REASONING', False)
@patch('base_utils.get_llm')
def test_pack_extractor_agent_options_priority(mock_get_llm):
    agent = PackExtractorAgent()
    mock_get_llm.assert_called_once()
    kwargs = mock_get_llm.call_args.kwargs
    assert kwargs["reasoning"] is True
    assert kwargs["num_ctx"] == config.PACK_NUM_CTX
    assert kwargs["num_predict"] == config.PACK_NUM_PREDICT
