"""Host-only regression tests for the fail-closed M33 runtime pin gate.

Run: python3 -m unittest discover -s scripts/tests -p 'test_flat_m33_pin.py'
No GPU, network, or Cargo invocation is required.
"""
import importlib.util
import pathlib
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "check-flat-m33-pin.py"
spec = importlib.util.spec_from_file_location("flat_m33_pin", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

RUNTIME = "c74a3806f8cfe621671694692fc14adfe08116f0"
PLANNER = "4529a2079434965e13e90ddd2e98ecc88ee0cb3a"
REPO = "git+https://github.com/Memorithm/FLAT-ATTENTION"


class PinGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        (self.root / "scirust-gpu").mkdir()

    def write(self, declared=RUNTIME, resolved=RUNTIME, planner=True):
        (self.root / "scirust-gpu/Cargo.toml").write_text(
            '[dependencies]\nflat-attention = { git = "https://github.com/Memorithm/FLAT-ATTENTION", '
            f'rev = "{declared}", optional = true }}\n'
        )
        sources = [f'{REPO}?rev={declared}#{resolved}']
        if planner:
            sources.append(f'{REPO}?rev={PLANNER}#{PLANNER}')
        packages = '\n'.join(
            '[[package]]\nname = "flat-attention"\nversion = "0.1.0"\n'
            f'source = "{source}"\n' for source in sources
        )
        (self.root / "Cargo.lock").write_text('version = 4\n' + packages)

    def test_exact_runtime_pin_passes_with_distinct_planner(self):
        self.write()
        self.assertEqual(gate.verify(self.root, RUNTIME), [])

    def test_candidate_mismatch_fails(self):
        self.write()
        self.assertTrue(gate.verify(self.root, PLANNER))

    def test_resolved_sha_mismatch_fails(self):
        self.write(resolved=PLANNER)
        self.assertTrue(gate.verify(self.root, RUNTIME))

    def test_conflicting_runtime_resolution_fails_even_with_valid_entry(self):
        self.write()
        lock = self.root / "Cargo.lock"
        lock.write_text(
            lock.read_text()
            + '[[package]]\nname = "flat-attention"\nversion = "0.1.0"\n'
            + f'source = "{REPO}?rev={RUNTIME}#{PLANNER}"\n'
        )
        failures = gate.verify(self.root, RUNTIME)
        self.assertIn(
            "Cargo.lock has conflicting resolved SHA for runtime FLAT rev",
            failures,
        )

    def test_missing_runtime_lock_entry_fails(self):
        self.write()
        lock = self.root / "Cargo.lock"
        lock.write_text('version = 4\n')
        self.assertTrue(gate.verify(self.root, RUNTIME))

    def test_short_candidate_fails(self):
        self.write()
        self.assertTrue(gate.verify(self.root, RUNTIME[:12]))

    def test_invalid_toml_fails_closed(self):
        self.write()
        (self.root / "Cargo.lock").write_text("not = [")
        self.assertTrue(gate.verify(self.root, RUNTIME))

    def test_manifest_declared_revision_must_be_full_sha(self):
        self.write(declared=RUNTIME[:12])
        self.assertTrue(gate.verify(self.root, RUNTIME))

    def test_missing_optional_flag_fails_closed(self):
        self.write()
        manifest = self.root / "scirust-gpu/Cargo.toml"
        manifest.write_text(manifest.read_text().replace(", optional = true", ""))
        self.assertTrue(gate.verify(self.root, RUNTIME))

    def test_planner_only_lock_entry_cannot_impersonate_runtime(self):
        self.write()
        lock = self.root / "Cargo.lock"
        lock.write_text(
            'version = 4\n[[package]]\nname = "flat-attention"\n'
            'version = "0.1.0"\n'
            f'source = "{REPO}?rev={PLANNER}#{PLANNER}"\n'
        )
        failures = gate.verify(self.root, RUNTIME)
        self.assertIn(
            "Cargo.lock has no runtime FLAT package with matching rev AND resolved SHA",
            failures,
        )
        self.assertFalse(any("unreadable" in failure for failure in failures))

    def test_non_table_lock_package_fails_closed(self):
        self.write()
        (self.root / "Cargo.lock").write_text('version = 4\npackage = ["not-a-table"]\n')
        failures = gate.verify(self.root, RUNTIME)
        self.assertIn("Cargo.lock package entries must be TOML tables", failures)

    def test_non_list_lock_package_fails_closed(self):
        self.write()
        (self.root / "Cargo.lock").write_text('version = 4\n[package]\nname = "flat-attention"\n')
        failures = gate.verify(self.root, RUNTIME)
        self.assertIn("Cargo.lock package entries must be TOML tables", failures)

    def test_non_table_manifest_dependencies_fails_closed(self):
        self.write()
        manifest = self.root / "scirust-gpu/Cargo.toml"
        manifest.write_text('dependencies = ["not-a-table"]\n')
        failures = gate.verify(self.root, RUNTIME)
        self.assertIn("runtime manifest dependencies must be a TOML table", failures)

    def test_missing_manifest_fails_closed_with_unreadable_diagnostic(self):
        self.write()
        (self.root / "scirust-gpu/Cargo.toml").unlink()
        failures = gate.verify(self.root, RUNTIME)
        self.assertTrue(any("unreadable" in failure for failure in failures))

    def test_missing_lockfile_fails_closed_with_unreadable_diagnostic(self):
        self.write()
        (self.root / "Cargo.lock").unlink()
        failures = gate.verify(self.root, RUNTIME)
        self.assertTrue(any("unreadable" in failure for failure in failures))

    def test_uppercase_candidate_fails_closed(self):
        self.write()
        self.assertTrue(gate.verify(self.root, RUNTIME.upper()))


if __name__ == "__main__":
    unittest.main()
