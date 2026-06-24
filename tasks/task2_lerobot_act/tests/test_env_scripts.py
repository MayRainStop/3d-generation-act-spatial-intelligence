from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class EnvScriptTests(unittest.TestCase):
    def test_train_wrapper_records_failures_after_training_command(self):
        text = (ROOT / "scripts" / "05_train_run.sh").read_text(encoding="utf-8")
        self.assertIn("set +e", text)
        self.assertIn("PIPESTATUS", text)
        self.assertIn("exit_code", text)

    def test_train_wrapper_uses_hf_mirror_by_default(self):
        text = (ROOT / "scripts" / "05_train_run.sh").read_text(encoding="utf-8")
        self.assertIn("HF_ENDPOINT", text)
        self.assertIn("https://hf-mirror.com", text)


if __name__ == "__main__":
    unittest.main()
