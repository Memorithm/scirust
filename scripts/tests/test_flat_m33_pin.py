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


if __name__ == "__main__":
    unittest.main()
