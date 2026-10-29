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

@pytest.fixture(autouse=True, scope="function")
def prevent_character_json_pollution():
    """
    Ensures that no test modifies the versioned Memory files.
    Compares their byte content before and after each test.
    """
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent / "Memory"
    files_to_monitor = [
        base_dir / "character.json",
        base_dir / "creation_manual.json",
        base_dir / "scenario.json"
    ]

    # Store initial state
    initial_states = {}
    for f in files_to_monitor:
        initial_states[f] = f.read_bytes() if f.exists() else None

    yield

    # Check final state
    for f in files_to_monitor:
        content_after = f.read_bytes() if f.exists() else None
        if initial_states[f] != content_after:
            pytest.fail(f"Test modified {f.name}. Tests should use tmp_path for GameStateEngine.")
