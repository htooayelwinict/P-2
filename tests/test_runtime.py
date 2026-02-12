import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multi_agent_app.nodes import BridgeAccessError, normalize_scope_prefixes
from multi_agent_app.runtime import MultiAgentRuntime


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_runtime(base_dir: Path) -> MultiAgentRuntime:
    _write(base_dir / "admin" / "secrets.txt", "top-secret-admin-value")
    _write(base_dir / "docs" / "guide.txt", "public-doc-content")
    _write(base_dir / "docs" / "research" / "note.md", "research-note")
    return MultiAgentRuntime.create(base_dir=base_dir, model_mode="offline")


class RuntimeTests(unittest.TestCase):
    def test_admin_turn_uses_supervisor_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _make_runtime(Path(tmp))
            output = runtime.run_admin_turn("hello admin")
            self.assertIn("supervisor handled: hello admin", output)

    def test_user_turn_uses_customer_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _make_runtime(Path(tmp))
            output = runtime.run_user_turn("hello user")
            self.assertIn("customer-service handled: hello user", output)

    def test_admin_docs_request_routes_via_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _make_runtime(Path(tmp))
            output = runtime.run_admin_turn("read docs for user")
            self.assertIn("customer-service handled: read docs for user", output)

    def test_admin_non_docs_request_stays_in_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _make_runtime(Path(tmp))
            output = runtime.run_admin_turn("list files")
            self.assertIn("supervisor handled: list files", output)

    def test_bridge_rejects_non_supervisor_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _make_runtime(Path(tmp))
            with self.assertRaises(BridgeAccessError):
                runtime.bridge.invoke({"origin": "admin_cli", "admin_input": "x"})

    def test_bridge_forwards_only_admin_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _make_runtime(Path(tmp))
            payload = runtime.bridge.invoke(
                {
                    "origin": "supervisor",
                    "admin_input": "read_file /docs/guide.txt",
                    "secret_context": "never-forward-this",
                    "secret_key_ref": "key-1",
                }
            )
            self.assertEqual(
                payload,
                {
                    "origin": "bridge",
                    "bridge_input": "read_file /docs/guide.txt",
                },
            )

    def test_langgraph_and_deepagents_wiring_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _make_runtime(Path(tmp))
            self.assertEqual(runtime.admin_graph.__class__.__name__, "CompiledStateGraph")
            self.assertEqual(runtime.user_graph.__class__.__name__, "CompiledStateGraph")
            self.assertEqual(
                runtime.supervisor.worker.agent.__class__.__name__,
                "CompiledStateGraph",
            )
            self.assertEqual(
                runtime.customer.worker.agent.__class__.__name__,
                "CompiledStateGraph",
            )
            self.assertTrue(runtime.supervisor.worker.backend.virtual_mode)
            self.assertTrue(runtime.customer.worker.backend.virtual_mode)

    def test_scoped_backend_roots_are_correct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            runtime = _make_runtime(base)
            self.assertEqual(runtime.supervisor.worker.backend.cwd, (base / "admin").resolve())
            self.assertEqual(runtime.customer.worker.backend.cwd, (base / "docs").resolve())

    def test_scoped_backend_blocks_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _make_runtime(Path(tmp))
            with self.assertRaises(ValueError):
                runtime.supervisor.worker.backend.read("/../../etc/passwd")
            with self.assertRaises(ValueError):
                runtime.customer.worker.backend.read("/../../etc/passwd")

    def test_scope_prefix_normalization(self) -> None:
        self.assertEqual(
            normalize_scope_prefixes("read_file /admin/README.md", "admin"),
            "read_file /README.md",
        )
        self.assertEqual(
            normalize_scope_prefixes("list files under /admin", "admin"),
            "list files under /",
        )
        self.assertEqual(
            normalize_scope_prefixes("read_file /docs/research/a.md", "docs"),
            "read_file /research/a.md",
        )


if __name__ == "__main__":
    unittest.main()
