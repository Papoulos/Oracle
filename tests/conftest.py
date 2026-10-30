import os
import pytest
import systems.promote

# Capture the original path before any patching occurs
_ORIGINAL_SYSTEMS_DIR = systems.promote.SYSTEMS_DIR

@pytest.fixture(autouse=True, scope="function")
def prevent_draft_pollution():
    """
    Ensures that no test writes to the real systems/draft directory.
    Takes a snapshot before the test and compares after.
    """
    real_draft_dir = os.path.join(_ORIGINAL_SYSTEMS_DIR, "draft")

    # Take a snapshot of existing files/folders
    snapshot = None
    if os.path.exists(real_draft_dir):
        snapshot = set(os.listdir(real_draft_dir))

    yield

    # Check after the test
    if not os.path.exists(real_draft_dir) and snapshot is None:
        return

    if os.path.exists(real_draft_dir):
        current = set(os.listdir(real_draft_dir))
        if snapshot is None:
            pytest.fail(f"Test created {real_draft_dir} which did not exist before. Contents: {current}")
        else:
            new_entries = current - snapshot
            if new_entries:
                pytest.fail(f"Test polluted {real_draft_dir} with new entries: {new_entries}")

def _check_memory_files(initial_states: dict, scope_label: str):
    for f, before in initial_states.items():
        content_after = f.read_bytes() if f.exists() else None
        if before != content_after:
            pytest.fail(
                f"Test ({scope_label}) modified {f.name}. Tests should use tmp_path for GameStateEngine and Memory/."
            )


def _snapshot_memory_files() -> dict:
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent / "Memory"
    files_to_monitor = [
        base_dir / "character.json",
        base_dir / "creation_manual.json",
        base_dir / "scenario.json",
    ]
    return {f: (f.read_bytes() if f.exists() else None) for f in files_to_monitor}


@pytest.fixture(autouse=True, scope="module")
def prevent_module_memory_pollution():
    """
    Module-level guard that catches setup_module()/teardown_module() modifying versioned Memory files.
    """
    initial_states = _snapshot_memory_files()
    yield
    _check_memory_files(initial_states, "module")


@pytest.fixture(autouse=True, scope="function")
def prevent_character_json_pollution(monkeypatch):
    """
    Ensures that no test modifies the versioned Memory files, and isolates tests
    from the developer's local SYSTEM_PACK environment variable in .env.
    """
    monkeypatch.delenv("SYSTEM_PACK", raising=False)
    initial_states = _snapshot_memory_files()
    yield
    _check_memory_files(initial_states, "function")
