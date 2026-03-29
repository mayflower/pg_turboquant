import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / "benchmarks" / "rag"
README = RAG_DIR / "README.md"
BOOTSTRAP = RAG_DIR / "bootstrap_bergen.sh"


class RagBenchmarkScaffoldContractTest(unittest.TestCase):
    def test_rag_benchmark_skeleton_exists(self):
        self.assertTrue(RAG_DIR.is_dir(), "expected benchmarks/rag directory")
        self.assertTrue(README.exists(), "expected benchmarks/rag/README.md")
        self.assertTrue(BOOTSTRAP.exists(), "expected BERGEN bootstrap script")

        readme = README.read_text(encoding="utf-8")
        self.assertIn("VSBT", readme)
        self.assertIn("BERGEN", readme)
        self.assertIn("BEIR", readme)
        self.assertIn("LoTTE", readme)
        self.assertIn("layered benchmark strategy", readme.lower())

    def test_bootstrap_dry_run_emits_expected_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_dir = Path(tmpdir) / "bergen-env"
            result = subprocess.run(
                [str(BOOTSTRAP), "--dry-run", "--env-dir", str(env_dir)],
                check=True,
                capture_output=True,
                text=True,
            )

        stdout = result.stdout
        self.assertIn("DRY RUN", stdout)
        self.assertIn(f"create uv environment at {env_dir}", stdout)
        self.assertIn("uv venv", stdout)
        self.assertIn("uv pip install --python", stdout)
        self.assertIn("requirements-bergen.txt", stdout)
        self.assertIn("BERGEN", stdout)


if __name__ == "__main__":
    unittest.main()
