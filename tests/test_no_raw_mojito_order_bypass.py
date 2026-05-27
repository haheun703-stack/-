"""Regression guard: raw mojito order calls must not bypass runtime safety."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORDER_METHODS = {
    "create_limit_buy_order",
    "create_limit_sell_order",
    "create_market_buy_order",
    "create_market_sell_order",
}
ALLOWLIST = {
    Path("src/adapters/kis_order_adapter.py"),
}
SKIP_PARTS = {
    "scripts/archive",
    "scripts/one_off",
    "src/agents/code_auditor.py",
}


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root_name in ("scripts", "src"):
        root = PROJECT_ROOT / root_name
        if root.exists():
            files.extend(root.rglob("*.py"))
    return sorted(files)


def _rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _parent_functions(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST | None:
    cur = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur
    return None


def test_raw_mojito_order_calls_have_runtime_guard():
    violations: list[str] = []

    for path in _iter_python_files():
        rel = Path(_rel(path))
        rel_text = rel.as_posix()
        if rel in ALLOWLIST or any(rel_text.startswith(part) for part in SKIP_PARTS):
            continue

        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(rel))
        except SyntaxError:
            continue

        lines = source.splitlines()
        parents = _parent_functions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr not in ORDER_METHODS:
                continue

            enclosing = _enclosing_function(node, parents)
            start_line = enclosing.lineno if enclosing is not None else 1
            prior_source = "\n".join(lines[start_line - 1 : node.lineno - 1])
            if "assert_runtime_orders_allowed()" not in prior_source:
                violations.append(f"{rel_text}:{node.lineno} {func.attr}")

    assert violations == []
