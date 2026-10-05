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
