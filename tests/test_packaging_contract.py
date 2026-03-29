import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "pg_turboquant.control"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
SQL_DIR = ROOT / "sql"


class PackagingContractTest(unittest.TestCase):
    def test_control_declares_upgrade_target(self):
        control = CONTROL.read_text(encoding="utf-8")
        default_match = re.search(r"^default_version\s*=\s*'([^']+)'", control, re.MULTILINE)
        self.assertIsNotNone(default_match)
        default_version = default_match.group(1)

        self.assertNotEqual(
            default_version,
            "0.1.0",
            "default_version should move forward once upgrade scripts exist",
        )

        install_script = SQL_DIR / f"pg_turboquant--{default_version}.sql"
        self.assertTrue(
            install_script.exists(),
            f"expected install script for default version {default_version}",
        )

        self.assertTrue(
            (SQL_DIR / "pg_turboquant--0.1.0--0.1.1.sql").exists(),
            "expected an explicit upgrade script from 0.1.0 to 0.1.1",
        )
        self.assertTrue(
            (SQL_DIR / "pg_turboquant--0.1.1--0.1.2.sql").exists(),
            "expected an explicit upgrade script from 0.1.1 to 0.1.2",
        )
        self.assertTrue(
            (SQL_DIR / "pg_turboquant--0.1.2--0.1.3.sql").exists(),
            "expected an explicit upgrade script from 0.1.2 to 0.1.3",
        )
        self.assertTrue(
            (SQL_DIR / "pg_turboquant--0.1.3--0.1.4.sql").exists(),
            "expected an explicit upgrade script from 0.1.3 to 0.1.4",
        )

    def test_ci_matrix_covers_pg16_and_pg17(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("matrix:", workflow)
        self.assertRegex(
            workflow,
            r"postgres(?:ql)?_version:\s*\[.*16.*17.*\]",
            "CI matrix should include PostgreSQL 16 and 17",
        )
        self.assertIn("postgresql-${{ matrix.postgresql_version }}", workflow)
        self.assertIn("postgresql-server-dev-${{ matrix.postgresql_version }}", workflow)


if __name__ == "__main__":
    unittest.main()
