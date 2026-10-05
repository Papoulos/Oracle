import os
import json
import pytest
from unittest.mock import patch, MagicMock

import config
from fpdf import FPDF
from langchain_core.documents import Document

from systems.pdf_pages import PageText, read_pdf_pages, format_pages
from systems.provenance import ProvenanceData, REVIEW_CONFIDENCE_THRESHOLD
from pack_extractor import select_pages, PackExtractorAgent
from systems.promote import init_review, promote_pack

# Helper to generate short PDF for tests
def create_pdf(path, text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Replaces accents for test simplicity
    pdf.cell(200, 10, txt=text, ln=True, align="L")
    pdf.output(path)

@pytest.fixture
def test_pdfs(tmp_path):
    pdf1 = tmp_path / "rules_part1.pdf"
    pdf2 = tmp_path / "rules_part2.pdf"

    create_pdf(str(pdf1), "Action resolution requires rolling a d20 against a target. Critical success on 20.")
    create_pdf(str(pdf2), "You recover hit points with a short rest.")

    return [str(pdf1), str(pdf2)]

def test_pdf_pages(test_pdfs):
    pages = read_pdf_pages(test_pdfs)
    assert len(pages) == 2
    assert pages[0].source == "rules_part1.pdf"
    assert pages[0].page == 1
    assert "Action resolution" in pages[0].text

    fmt = format_pages(pages)
    assert "[[rules_part1.pdf p.1]]" in fmt
    assert "[[rules_part2.pdf p.1]]" in fmt

def test_select_pages():
    mock_store = MagicMock()
    # Return document with 0-indexed page pointing to our 1-indexed PageText
    mock_store.similarity_search.return_value = [
        Document(page_content="...", metadata={"source": "rules.pdf", "page": 0})
    ]
    pages = [PageText("rules.pdf", 1, "test " * 10)]

    selected = select_pages(mock_store, ["query"], pages, max_chars=1000, k=1)
    assert len(selected) == 1
    assert selected[0].source == "rules.pdf"

def test_fuzzy_match():
    agent = PackExtractorAgent()

    # Exact match
    assert agent._fuzzy_match("Action resolution", "This is an Action resolution process.") is True

    # One changed word
    assert agent._fuzzy_match("Action resoltuion", "This is an Action resolution process.") is True

    # Typos
    assert agent._fuzzy_match("Acton resolution", "This is an Action resolution process.") is True

    # Invented excerpt
    assert agent._fuzzy_match("Completely made up text", "This is an Action resolution process.") is False

    # Ligature and hyphenation
    page_text = "This is an ac-\ntion resol\uFB01ution process."
    excerpt = "action resolution"
    assert agent._fuzzy_match(excerpt, page_text) is True

@patch.object(PackExtractorAgent, '_invoke_logged')
def test_pack_extractor_empty_selected_pages(mock_invoke, test_pdfs, tmp_path):
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with patch('systems.promote.SYSTEMS_DIR', str(tmp_path / "systems")):
            mock_store = MagicMock()
            mock_store.similarity_search.return_value = []
            agent = PackExtractorAgent(store=mock_store)

            with patch('pack_extractor.select_pages', return_value=[]):
                with patch('pack_extractor.config.PACK_CONTEXT_MAX_CHARS', 1):
                    with pytest.raises(ValueError, match="Core index is empty or does not contain these PDFs. Run `python indexer.py --core` first."):
                        agent.extract("test_empty", test_pdfs)
    finally:
        os.chdir(old_cwd)

@patch.object(PackExtractorAgent, '_invoke_logged')
def test_pack_extractor_unknown_family(mock_invoke, test_pdfs, tmp_path):
    # Set cwd to tmp_path for isolation
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        import systems.promote
        systems.promote.SYSTEMS_DIR = str(tmp_path / 'systems')
        import systems.promote
        systems.promote.SYSTEMS_DIR = str(tmp_path / 'systems')
        agent = PackExtractorAgent()

        # Mock step A to return unknown
        mock_msg = MagicMock()
        mock_msg.content = '```json\n{"family": "unknown", "reason": "Too ambiguous", "candidates": [{"family": "D20VsTarget", "confidence": 50, "why": "Has d20"}]}\n```'
        mock_invoke.return_value = mock_msg

        agent.extract("test_unknown", test_pdfs)

        prov_path = os.path.join(tmp_path, "systems", "draft", "test_unknown", "provenance.json")
        assert os.path.exists(prov_path)
        with open(prov_path, "r") as f:
            prov = json.load(f)

        assert prov["status"] == "unknown_family"
        assert len(prov["candidates"]) == 1
        assert not os.path.exists(os.path.join(tmp_path, "systems", "draft", "test_unknown", "manifest.yaml"))
    finally:
        os.chdir(old_cwd)


@patch.object(PackExtractorAgent, '_invoke_logged')
def test_pack_extractor_full_flow(mock_invoke, test_pdfs, tmp_path):
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        import systems.promote
        systems.promote.SYSTEMS_DIR = str(tmp_path / 'systems')
        import systems.promote
        systems.promote.SYSTEMS_DIR = str(tmp_path / 'systems')
        agent = PackExtractorAgent()

        # We need mock responses for Step A and Step B
        # Step A response: D20VsTarget
        msg_a = MagicMock()
        msg_a.content = '{"family": "D20VsTarget", "confidence": 95, "why": "Uses d20", "name": "English Test System", "language": "en"}'

        # Step B response: Valid configurations with provenance
        msg_b = MagicMock()
        msg_b.content = """
        {
          "resolution.json": {
            "config": {
              "family": "D20VsTarget",
              "advantage_enabled": true,
              "advantage_dice": 2,
              "critical_success_on": 20,
              "critical_failure_on": 1
            },
            "provenance": [
              {"path": "/critical_success_on", "source": "rules_part1.pdf", "page": 1, "excerpt": "Critical success on 20", "confidence": 90}
            ]
          },
          "resources.json": {
             "config": {
                "recovery_triggers": [{"id": "short_rest", "name": "Short Rest"}],
                "pools": [],
                "pool_groups": []
             },
             "provenance": [
                 {"path": "/recovery_triggers/0/id", "source": "rules_part2.pdf", "page": 1, "excerpt": "short rest", "confidence": 85},
                 {"path": "/recovery_triggers/0/name", "source": "rules_part2.pdf", "page": 1, "excerpt": "short rest", "confidence": 85}
             ]
          },
          "triggers.json": {
             "config": {
                 "version": 1,
                 "rules": []
             },
             "provenance": [{"path": "/version", "source": "rules_part1.pdf", "page": 1, "excerpt": "Action resolution", "confidence": 80}]
          }
        }
        """

        # We mock side_effect to return msg_a then msg_b
        mock_invoke.side_effect = [msg_a, msg_b]

        agent.extract("test_full", test_pdfs)

        draft_dir = os.path.join(tmp_path, "systems", "draft", "test_full")
        assert os.path.exists(os.path.join(draft_dir, "manifest.yaml"))
        assert os.path.exists(os.path.join(draft_dir, "resolution.json"))

        with open(os.path.join(draft_dir, "manifest.yaml"), "r") as f:
            import yaml
            manifest_data = yaml.safe_load(f)
            assert manifest_data["name"] == "English Test System"
            assert manifest_data["language"] == "en"

        # Check provenance
        with open(os.path.join(draft_dir, "provenance.json"), "r") as f:
             prov = json.load(f)

        assert prov["status"] == "complete"
        entries = prov["entries"]

        # One of them is advantage_enabled which has NO provenance in our mock
        missing_prov_entry = next((e for e in entries if e["path"] == "/advantage_enabled"), None)
        assert missing_prov_entry is not None
        assert missing_prov_entry["needs_review"] is True
        assert "defaulted_by_schema" in missing_prov_entry["review_reasons"]

        # For testing, we mock SYSTEMS_DIR to our tmp_path
        with patch('systems.promote.SYSTEMS_DIR', str(tmp_path / "systems")):
            # Check promotion refusal without review.json
            with pytest.raises(SystemExit) as e:
                 promote_pack("test_full", force=False)
            assert e.value.code == 1

            # Init review
            init_review("test_full")
            review_path = os.path.join(draft_dir, "review.json")
            assert os.path.exists(review_path)

            with open(review_path, "r") as f:
                 review = json.load(f)

            # Validate them manually
            for k in review["validated"]:
                 review["validated"][k] = True

            with open(review_path, "w") as f:
                 json.dump(review, f)

            # Now promote should work
            promote_pack("test_full", force=False)

            # Test paths based on promote.py directory resolution
            from systems.promote import get_target_dir
            target_dir = get_target_dir("test_full")

            assert os.path.exists(os.path.join(target_dir, "manifest.yaml"))
            assert not os.path.exists(draft_dir)

            # Now force promotion and test backup
            os.makedirs(draft_dir, exist_ok=True)
            # Create dummy complete provenance
            with open(os.path.join(draft_dir, "provenance.json"), "w") as f:
                f.write(json.dumps({
                    "status": "complete", "pack_id": "test_full", "model": "test",
                    "generated_at": "test", "context_pages": [], "entries": []
                }))

            # Create manifest that matches pack_id
            with open(os.path.join(draft_dir, "manifest.yaml"), "w") as f:
                 f.write("id: test_full\nversion: '1'\nname: test\nlanguage: en\nfamily: D20VsTarget\nsource_pdfs: []")

            with open(os.path.join(draft_dir, "resolution.json"), "w") as f:
                 f.write('{"family": "D20VsTarget", "advantage_enabled": false, "critical_success_on": 20, "critical_failure_on": 1}')

            with open(os.path.join(draft_dir, "resources.json"), "w") as f:
                 f.write('{"recovery_triggers": [], "pools": [], "pool_groups": []}')

            # Run promotion, backup should take place and delete old version
            promote_pack("test_full", force=True)

            assert os.path.exists(os.path.join(target_dir, "manifest.yaml"))
            assert not os.path.exists(target_dir + ".bak") # Backup deleted if success

            # Force promote a failing test
            os.makedirs(draft_dir, exist_ok=True)
            with open(os.path.join(draft_dir, "manifest.yaml"), "w") as f:
                 f.write("id: test_full\nversion: '1'\nname: test\nlanguage: en\nfamily: D20VsTarget\nsource_pdfs: []")
            # Missing provenance, so promotion will fail
            with pytest.raises(SystemExit) as e:
                promote_pack("test_full", force=True)
            assert e.value.code == 1
            # Backup should have been restored
            assert os.path.exists(os.path.join(target_dir, "manifest.yaml"))
            assert not os.path.exists(target_dir + ".bak") # Backup restored

    finally:
        os.chdir(old_cwd)


@patch.object(PackExtractorAgent, '_invoke_logged')
def test_pack_extractor_retry_logic(mock_invoke, test_pdfs, tmp_path):
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        import systems.promote
        systems.promote.SYSTEMS_DIR = str(tmp_path / 'systems')
        import systems.promote
        systems.promote.SYSTEMS_DIR = str(tmp_path / 'systems')
        agent = PackExtractorAgent()

        # Step A
        msg_a = MagicMock()
        msg_a.content = '{"family": "D20VsTarget", "confidence": 95, "why": "Uses d20"}'

        # Step B1: Invalid JSON (e.g. missing required field critical_failure_on)
        msg_b1 = MagicMock()
        msg_b1.content = """
        {
          "resolution.json": {
            "config": {
              "family": "D20VsTarget",
              "advantage_enabled": true,
              "advantage_dice": 2,
              "critical_success_on": 20
            },
            "provenance": []
          }
        }
        """

        # Step B2: Valid
        msg_b2 = MagicMock()
        msg_b2.content = """
        {
          "resolution.json": {
            "config": {
              "family": "D20VsTarget",
              "advantage_enabled": true,
              "advantage_dice": 2,
              "critical_success_on": 20,
              "critical_failure_on": 1
            },
            "provenance": []
          },
          "resources.json": {"config": {"recovery_triggers": [], "pools": [], "pool_groups": []}, "provenance": []},
          "triggers.json": {"config": {"version": 1, "rules": []}, "provenance": []}
        }
        """

        mock_invoke.side_effect = [msg_a, msg_b1, msg_b2]

        agent.extract("test_retry", test_pdfs)

        draft_dir = os.path.join(tmp_path, "systems", "draft", "test_retry")
        with open(os.path.join(draft_dir, "provenance.json"), "r") as f:
             prov = json.load(f)

        assert prov["status"] == "complete"
    finally:
         os.chdir(old_cwd)



@patch.object(PackExtractorAgent, '_invoke_logged')
def test_language_normalization(mock_invoke, test_pdfs, tmp_path):
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        import systems.promote
        systems.promote.SYSTEMS_DIR = str(tmp_path / 'systems')
        agent = PackExtractorAgent()

        def run_lang_test(lang_val, expected_lang, expected_defaulted):
            msg_a = MagicMock()
            msg_a.content = json.dumps({
                "family": "D20VsTarget",
                "confidence": 95,
                "why": "Uses d20",
                "name": "Test System",
                "language": lang_val
            })

            msg_b = MagicMock()
            msg_b.content = """
            {
              "resolution.json": {"config": {"family": "D20VsTarget", "advantage_enabled": true, "critical_success_on": 20, "critical_failure_on": 1}, "provenance": []},
              "resources.json": {"config": {"recovery_triggers": [], "pools": [], "pool_groups": []}, "provenance": []},
              "triggers.json": {"config": {"version": 1, "rules": []}, "provenance": []}
            }
            """

            mock_invoke.side_effect = [msg_a, msg_b]

            pack_id = f"test_lang_{lang_val if isinstance(lang_val, str) else 'none'}"
            pack_id = pack_id.replace('-', '_').lower()
            agent.extract(pack_id, test_pdfs)

            draft_dir = os.path.join(tmp_path, "systems", "draft", pack_id)
            with open(os.path.join(draft_dir, "manifest.yaml"), "r") as f:
                import yaml
                manifest_data = yaml.safe_load(f)
                assert manifest_data["language"] == expected_lang

            with open(os.path.join(draft_dir, "provenance.json"), "r") as f:
                prov = json.load(f)

            lang_entry = next((e for e in prov["entries"] if e["path"] == "/language"), None)
            if expected_defaulted:
                assert lang_entry is not None
                assert lang_entry["needs_review"] is True
                assert "language_defaulted" in lang_entry["review_reasons"]
                assert lang_entry["file"] == "manifest.yaml"
            else:
                assert lang_entry is None

        run_lang_test("fr-FR", "fr", False)
        run_lang_test("French", "en", True)
        run_lang_test(None, "en", True)

    finally:
        os.chdir(old_cwd)

def test_parse_confidence():
    from pack_extractor import parse_confidence
    assert parse_confidence(95) == 95
    assert parse_confidence(95.4) == 95
    assert parse_confidence("95") == 95
    assert parse_confidence(" 95 ") == 95
    assert parse_confidence("95%") == 95
    assert parse_confidence("95.5") == 96
    assert parse_confidence("150") == 100
    assert parse_confidence(float("nan")) is None
    assert parse_confidence("high") is None
    assert parse_confidence("about 95 or 80") is None
    assert parse_confidence(None) is None
    assert parse_confidence(True) is None

def test_normalize_language():
    from pack_extractor import normalize_language
    assert normalize_language("fr-FR") == "fr"
    assert normalize_language("fr_FR") == "fr"
    assert normalize_language(" fr ") == "fr"
    assert normalize_language("zh-Hans-CN") == "zh"
    assert normalize_language("French") is None
    assert normalize_language("fra") is None
    assert normalize_language("") is None
    assert normalize_language(123) is None
    assert normalize_language(None) is None

@patch.object(PackExtractorAgent, '_invoke_logged')
def test_confidence_conversion(mock_invoke, test_pdfs, tmp_path):
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        import systems.promote
        systems.promote.SYSTEMS_DIR = str(tmp_path / 'systems')
        agent = PackExtractorAgent()

        def run_conf_test(conf_val, expected_status):
            msg_a = MagicMock()
            msg_a.content = json.dumps({
                "family": "D20VsTarget",
                "confidence": conf_val,
                "why": "Uses d20",
                "name": "Test System",
                "language": "en"
            })

            msg_b = MagicMock()
            msg_b.content = """
            {
              "resolution.json": {"config": {"family": "D20VsTarget", "advantage_enabled": true, "critical_success_on": 20, "critical_failure_on": 1}, "provenance": []},
              "resources.json": {"config": {"recovery_triggers": [], "pools": [], "pool_groups": []}, "provenance": []},
              "triggers.json": {"config": {"version": 1, "rules": []}, "provenance": []}
            }
            """

            mock_invoke.side_effect = [msg_a, msg_b]

            pack_id = f"test_conf_{str(conf_val).replace('%', '_pct')}"
            agent.extract(pack_id, test_pdfs)

            draft_dir = os.path.join(tmp_path, "systems", "draft", pack_id)
            with open(os.path.join(draft_dir, "provenance.json"), "r") as f:
                prov = json.load(f)

            assert prov["status"] == expected_status

        run_conf_test(95, "complete")
        run_conf_test("95", "complete")
        run_conf_test("95%", "complete")
        run_conf_test("high", "unknown_family")
        run_conf_test(None, "unknown_family")

    finally:
        os.chdir(old_cwd)
