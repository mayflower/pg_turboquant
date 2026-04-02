import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "pg_turboquant.control"
MAKEFILE = ROOT / "Makefile"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
SQL_DIR = ROOT / "sql"
INSTALL_SCRIPT = SQL_DIR / "pg_turboquant--0.1.0.sql"
COMPATIBILITY_DOC = ROOT / "docs" / "reference" / "compatibility.md"
SUPPORT_CONTROL = ROOT / "pg_turboquant_test_support.control"
SUPPORT_INSTALL_SCRIPT = SQL_DIR / "pg_turboquant_test_support--0.1.0.sql"


class PackagingContractTest(unittest.TestCase):
    def test_control_declares_single_public_install_target(self):
        control = CONTROL.read_text(encoding="utf-8")
        makefile = MAKEFILE.read_text(encoding="utf-8")
        default_match = re.search(r"^default_version\s*=\s*'([^']+)'", control, re.MULTILINE)
        self.assertIsNotNone(default_match)
        default_version = default_match.group(1)
        extversion_match = re.search(r"^EXTVERSION\s*=\s*(\S+)", makefile, re.MULTILINE)
        self.assertIsNotNone(extversion_match)
        extversion = extversion_match.group(1)

        self.assertEqual(
            default_version,
            "0.1.0",
            "first public release should expose a single install version",
        )
        self.assertEqual(
            extversion,
            default_version,
            "Makefile EXTVERSION should match the control file public version",
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

    def test_ci_declares_compatibility_and_sanitizer_lanes(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("compatibility-matrix:", workflow)
        self.assertIn("pgvector_track: [latest_supported]", workflow)
        self.assertIn("PGVECTOR_REF: v0.8.1", workflow)
        self.assertIn("sanitizer:", workflow)
        self.assertIn("ASAN_OPTIONS:", workflow)
        self.assertIn("UBSAN_OPTIONS:", workflow)
        self.assertIn("python3 -m unittest tests.test_packaging_contract", workflow)
        self.assertIn("public_api_surface corruption_guardrails", workflow)
        self.assertIn("t/012_maintenance_reuse_restart.pl t/015_concurrent_build_and_write.pl", workflow)

    def test_make_install_does_not_install_pgvector_as_side_effect(self):
        makefile = MAKEFILE.read_text(encoding="utf-8")

        self.assertNotRegex(
            makefile,
            r"^install:\s+install-pgvector\b",
            "make install should not provision pgvector as a side effect",
        )

    def test_public_install_script_excludes_regression_only_debug_helpers(self):
        install_script = INSTALL_SCRIPT.read_text(encoding="utf-8")

        for helper_name in (
            "tq_debug_validate_reloptions",
            "tq_debug_router_metadata",
            "tq_debug_transform_metadata",
            "tq_test_corrupt_meta_magic",
        ):
            self.assertNotIn(
                helper_name,
                install_script,
                f"public install script should not expose regression helper {helper_name}",
            )

    def test_support_extension_assets_exist_and_require_pg_turboquant(self):
        support_control = SUPPORT_CONTROL.read_text(encoding="utf-8")
        support_install_script = SUPPORT_INSTALL_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("requires = 'pg_turboquant'", support_control)
        self.assertIn("default_version = '0.1.0'", support_control)
        for helper_name in (
            "tq_debug_validate_reloptions",
            "tq_debug_router_metadata",
            "tq_debug_transform_metadata",
            "tq_test_corrupt_meta_magic",
            "tq_test_corrupt_first_batch_occupied_count",
        ):
            self.assertIn(helper_name, support_install_script)

    def test_public_sql_surface_uses_turboquant_wrapper_support_functions(self):
        install_script = INSTALL_SCRIPT.read_text(encoding="utf-8")
        opclass_section = install_script.split("CREATE OPERATOR CLASS tq_cosine_ops", 1)[1]

        for wrapper_name in (
            "tq_vector_negative_inner_product",
            "tq_vector_l2_squared_distance",
            "tq_vector_norm",
            "tq_halfvec_negative_inner_product",
            "tq_halfvec_l2_squared_distance",
            "tq_halfvec_norm",
        ):
            self.assertIn(wrapper_name, install_script)

        for upstream_name in (
            "vector_negative_inner_product",
            "vector_l2_squared_distance",
            "halfvec_negative_inner_product",
            "halfvec_l2_squared_distance",
            "vector_norm",
            "l2_norm",
        ):
            self.assertNotRegex(
                opclass_section,
                rf"\bFUNCTION\s+\d+\s+{re.escape(upstream_name)}\s*\(",
                f"public opclass SQL should not bind directly to upstream function {upstream_name}",
            )

    def test_pgvector_headers_are_confined_to_the_compat_layer_in_src(self):
        src_dir = ROOT / "src"
        offenders = []

        for path in src_dir.glob("*.c"):
            text = path.read_text(encoding="utf-8")
            if "third_party/pgvector" in text and path.name != "tq_pgvector_compat.c":
                offenders.append(path.name)

        self.assertEqual(
            offenders,
            [],
            "pgvector source-header inclusion should stay inside tq_pgvector_compat.c",
        )

    def test_compatibility_doc_declares_supported_contract(self):
        compatibility_doc = COMPATIBILITY_DOC.read_text(encoding="utf-8")

        self.assertIn("PostgreSQL 16", compatibility_doc)
        self.assertIn("PostgreSQL 17", compatibility_doc)
        self.assertIn("pgvector v0.8.1", compatibility_doc)
        self.assertIn("REINDEX", compatibility_doc)


if __name__ == "__main__":
    unittest.main()
