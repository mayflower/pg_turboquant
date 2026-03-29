import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "pg_turboquant.control"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
SQL_DIR = ROOT / "sql"


class PackagingContractTest(unittest.TestCase):
    def test_control_declares_single_public_install_target(self):
        control = CONTROL.read_text(encoding="utf-8")
        default_match = re.search(r"^default_version\s*=\s*'([^']+)'", control, re.MULTILINE)
        self.assertIsNotNone(default_match)
        default_version = default_match.group(1)

        self.assertEqual(
            default_version,
            "0.1.0",
            "first public release should expose a single install version",
        )

        install_script = SQL_DIR / f"pg_turboquant--{default_version}.sql"
        self.assertTrue(
            install_script.exists(),
            f"expected install script for default version {default_version}",
        )

        install_scripts = sorted(SQL_DIR.glob("pg_turboquant--*.sql"))
        upgrade_script_pattern = re.compile(r"^pg_turboquant--[^-]+--[^-]+\.sql$")
        upgrade_scripts = [path for path in install_scripts if upgrade_script_pattern.match(path.name)]

        self.assertEqual(
            [path.name for path in install_scripts],
            ["pg_turboquant--0.1.0.sql"],
            "public release should ship one current install script",
        )
        self.assertEqual(upgrade_scripts, [], "public release should not ship internal upgrade-history scripts")

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
