import unittest
import os
import config

class TestConfig(unittest.TestCase):
    def test_default_values(self):
        self.assertEqual(config.OLLAMA_MODEL, "gemma3")
        self.assertEqual(config.CHROMA_PATH, "./chroma_db")
        self.assertEqual(config.COLLECTION_CODEX, "codex")
        self.assertEqual(config.COLLECTION_INTRIGUE, "intrigue")

if __name__ == "__main__":
    unittest.main()
