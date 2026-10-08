import re

with open('tests/test_pooled_recoveries.py', 'r') as f:
    code = f.read()

# Fix mock for systems dir - it seems GameStateEngine(character_file) is not loading the pack
# The problem is GameStateEngine.__init__ only loads the pack if `os.environ.get("SYSTEM_PACK")` is set, which we did.
# Let's check how GameStateEngine initializes self.pack.
# It calls _load_system_pack() which looks at `SYSTEM_PACK`. It then probably constructs the path as `systems/<pack_id>` or similar, but the tests need it to look at `tests/fixtures/<pack_id>`.
# We need to mock `systems.promote.get_draft_dir` or something to make sure it loads from fixtures. But `game_state_engine` might import `SYSTEMS_DIR` from `systems.promote`.
# Wait, let's look at game_state_engine.py how it loads the pack.
