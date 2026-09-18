#!/usr/bin/env python3
"""CPU-only regression tests for the API example inventory and policy gate."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "api_example_audit", Path(__file__).with_name("api-example-audit.py")
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class FenceTests(unittest.TestCase):
    def count(self, text):
        return audit.count_examples(text.splitlines())

    def test_plain_rust_and_tilde_fences(self):
        counts = self.count("```\nassert!(true);\n```\n~~~rust\nassert!(true);\n~~~")
        self.assertEqual(counts["runnable_candidates"], 2)

    def test_nonexecuted_fences_never_count_as_runnable(self):
        for tag, expected in [
            ("no_run", "compile_only"), ("rust,no_run", "compile_only"),
            ("compile_fail", "compile_fail"), ("ignore", "ignored"),
            ("rust,ignore-x86_64", "ignored"), ("should_panic", "should_panic"),
            ("python", "other_fences"), ("text", "other_fences"),
            ("rust,unknown", "other_fences"),
        ]:
            with self.subTest(tag=tag):
                result = self.count(f"```{tag}\nexample();\n```")
                self.assertEqual(result[expected], 1)
                self.assertEqual(result["runnable_candidates"], 0)

    def test_ignore_overrides_rust_and_no_run(self):
        self.assertEqual(self.count("```rust,no_run,ignore\nx();\n```")["ignored"], 1)

    def test_edition_and_hidden_setup_are_observed(self):
        self.assertEqual(self.count("```rust,edition2024\n# let x = 1;\nassert_eq!(x, 1);\n```")["runnable_candidates"], 1)

    def test_empty_and_unclosed_fences_do_not_count(self):
        result = self.count("```rust\n \n```\n```rust\nx();")
        self.assertEqual(result["empty_fences"], 1)
        self.assertEqual(result["unclosed_fences"], 1)
        self.assertEqual(result["runnable_candidates"], 0)

    def test_long_fence_is_not_closed_by_short_or_other_marker(self):
        result = self.count("````rust\n```\n~~~\nx();\n````")
        self.assertEqual(result["runnable_candidates"], 1)
        self.assertEqual(sum(result.values()), 1)

    def test_closing_fence_cannot_have_information(self):
        result = self.count("```rust\nx();\n```rust")
        self.assertEqual(result["unclosed_fences"], 1)
        self.assertEqual(result["runnable_candidates"], 0)

    def test_four_space_indented_and_inline_backticks_are_not_fences(self):
        result = self.count("Some `inline` text.\n    ```rust\n    x();\n    ```")
        self.assertEqual(sum(result.values()), 0)

    def test_backtick_in_backtick_info_is_not_an_opener(self):
        self.assertEqual(sum(self.count("```rust`invalid").values()), 0)

    def test_adjacent_attributes_and_blank_lines(self):
        lines = ["/// Summary.", "/// ```", "/// x();", "/// ```", "", "#[inline]", "pub fn f() {}"]
        result = audit.count_examples(audit.adjacent_docs(lines, 7))
        self.assertEqual(result["runnable_candidates"], 1)

    def test_four_slashes_and_preceding_items_break_adjacency(self):
        for barrier in ["//// not rustdoc", "fn private() {}", "#[cfg(", ")]", "// ordinary"]:
            lines = ["/// ```", "/// x();", "/// ```", barrier, "pub fn f() {}"]
            self.assertEqual(audit.adjacent_docs(lines, len(lines)), [])


class IndexPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "lib.rs"
        self.source.write_text("/// Example.\n/// ```\n/// assert!(true);\n/// ```\npub fn f() {}\n", encoding="utf-8")
        self.record = {"package": "example", "symbol": "f", "source": "lib.rs", "line": 5}
        self.payload = {"schema_version": 1, "filters": "none", "entries": [self.record]}
        self.policy = {"schema_version": 1, "sources": [{
            "source": "lib.rs", "min_runnable_examples": 1, "required_symbols": ["f"],
        }]}

    def entries(self):
        return audit.audit_index(self.root, self.payload)

    def test_valid_index_has_content_binding_and_passing_policy(self):
        entries = self.entries()
        self.assertEqual(len(entries[0]["source_sha256"]), 64)
        self.assertEqual(audit.check_policy(self.root, self.policy, entries), [])

    def test_missing_second_example_fails_policy(self):
        self.policy["sources"][0]["min_runnable_examples"] = 2
        errors = audit.check_policy(self.root, self.policy, self.entries())
        self.assertEqual(len(errors), 1)
        self.assertIn("require 2", errors[0])

    def test_no_run_cannot_satisfy_policy(self):
        self.source.write_text(self.source.read_text().replace("/// ```\n/// assert", "/// ```no_run\n/// assert"))
        self.assertTrue(audit.check_policy(self.root, self.policy, self.entries()))

    def test_stale_line_symbol_and_out_of_range_are_errors(self):
        for key, value in [("line", 4), ("line", 999), ("symbol", "other")]:
            saved = self.record[key]
            self.record[key] = value
            with self.assertRaisesRegex(ValueError, "stale/mismatched"):
                self.entries()
            self.record[key] = saved

    def test_duplicate_rows_are_rejected(self):
        self.payload["entries"].append(dict(self.record))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.entries()

    def test_filtered_empty_and_wrong_version_indexes_are_rejected(self):
        for update in [{"filters": "packages=example"}, {"entries": []}, {"schema_version": 2}, {"schema_version": True}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                audit.audit_index(self.root, dict(self.payload, **update))

    def test_invalid_record_fields_are_rejected(self):
        for key, value in [("line", True), ("line", 0), ("line", "5"), ("symbol", ""), ("source", None), ("package", "")]:
            saved = self.record[key]
            self.record[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.entries()
            self.record[key] = saved

    def test_paths_cannot_escape_or_be_absolute(self):
        for source in ["../lib.rs", str(self.source), "a\\lib.rs", "", "lib.txt"]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                audit.source_path(self.root, source)

    def test_symlink_outside_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "outside.rs"
            target.write_text("pub fn f() {}")
            (self.root / "link.rs").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "escapes"):
                audit.source_path(self.root, "link.rs")

    def test_policy_cannot_pass_with_missing_or_unindexed_functions(self):
        entries = self.entries()
        self.source.write_text(self.source.read_text() + "pub fn missing() {}\n")
        self.assertIn("incompletely indexed", " ".join(audit.check_policy(self.root, self.policy, entries)))
        self.policy["sources"][0]["required_symbols"].append("gone")
        self.assertIn("required symbols absent", " ".join(audit.check_policy(self.root, self.policy, entries)))

    def test_policy_cannot_pass_after_source_becomes_empty(self):
        self.source.write_text("// no API\n")
        self.assertTrue(audit.check_policy(self.root, self.policy, []))

    def test_required_symbol_count_detects_missing_same_named_callable(self):
        self.policy["sources"][0]["required_symbol_counts"] = {"f": 1}
        self.assertEqual(audit.check_policy(self.root, self.policy, self.entries()), [])
        self.policy["sources"][0]["required_symbol_counts"] = {"f": 2}
        problems = audit.check_policy(self.root, self.policy, self.entries())
        self.assertEqual(len(problems), 1)
        self.assertIn("occurs 1 times; require exactly 2", problems[0])

    def test_invalid_policy_rules_are_rejected(self):
        for key, value in [("min_runnable_examples", 0), ("min_runnable_examples", True), ("required_symbols", []), ("required_symbols", ["f", "f"]), ("required_symbol_counts", {"f": 0}), ("required_symbol_counts", {"gone": 1})]:
            rule = self.policy["sources"][0]
            marker = object()
            saved = rule.get(key, marker)
            rule[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                audit.check_policy(self.root, self.policy, self.entries())
            if saved is marker:
                del rule[key]
            else:
                rule[key] = saved
        self.policy["sources"].append(dict(self.policy["sources"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate policy"):
            audit.check_policy(self.root, self.policy, self.entries())

    def test_report_is_deterministic_and_distinguishes_candidates(self):
        report = audit.report_payload(self.entries(), [])
        self.assertEqual(report, audit.report_payload(self.entries(), []))
        self.assertEqual(report["with_runnable_candidates"], 1)
        self.assertEqual(report["with_two_runnable_candidates"], 0)
        self.assertIn("not proof of compilation", audit.render_markdown(report))

    def test_cli_emits_report_on_policy_failure_and_returns_nonzero(self):
        index = self.root / "index.json"
        policy = self.root / "policy.json"
        output = self.root / "output.json"
        markdown = self.root / "report.md"
        self.policy["sources"][0]["min_runnable_examples"] = 2
        index.write_text(json.dumps(self.payload))
        policy.write_text(json.dumps(self.policy))
        args = ["--root", str(self.root), "--index", str(index), "--policy", str(policy), "--output", str(output), "--markdown", str(markdown)]
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(audit.main(args), 1)
        self.assertTrue(json.loads(output.read_text())["policy_violations"])
        self.assertTrue(markdown.is_file())
        self.policy["sources"][0]["min_runnable_examples"] = 1
        policy.write_text(json.dumps(self.policy))
        self.assertEqual(audit.main(args), 0)

    def test_malformed_json_fails_without_claiming_success(self):
        path = self.root / "broken.json"
        path.write_text("[]")
        with self.assertRaisesRegex(ValueError, "JSON object"):
            audit.read_json(path)
        path.write_text("{")
        with contextlib.redirect_stderr(io.StringIO()):
            result = audit.main(["--index", str(path), "--output", str(self.root / "out.json")])
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
