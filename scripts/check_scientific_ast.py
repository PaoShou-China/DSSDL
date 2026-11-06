"""Prove that code-style changes preserve the archived Python syntax trees."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "330d311728fae0f6e1147965426baf0b29d09dfb"


class WithoutDocstrings(ast.NodeTransformer):
    """Ignore documentation whitespace, but retain all executable expressions."""

    def visit_Module(self, node):
        return self.strip(node)

    def visit_FunctionDef(self, node):
        return self.strip(node)

    def visit_AsyncFunctionDef(self, node):
        return self.strip(node)

    def visit_ClassDef(self, node):
        return self.strip(node)

    def strip(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr):
            value = node.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body = node.body[1:]
        return node


def normalized(source):
    return ast.dump(WithoutDocstrings().visit(ast.parse(source)), include_attributes=False)


def main():
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "-z", BASE], cwd=ROOT)
    records = []
    for name in names.split(b"\0"):
        if not name.endswith(b".py"):
            continue
        relative = name.decode("utf-8")
        old = subprocess.check_output(["git", "show", f"{BASE}:{relative}"], cwd=ROOT).decode(
            "utf-8-sig"
        )
        new = (ROOT / relative).read_text(encoding="utf-8-sig")
        before, after = normalized(old), normalized(new)
        if before != after:
            raise AssertionError(f"Executable syntax changed: {relative}")
        records.append(
            {"path": relative, "executable_ast_sha256": hashlib.sha256(after.encode()).hexdigest()}
        )
    print(
        json.dumps(
            {"base_commit": BASE, "python_files": len(records), "all_executable_ast_equal": True},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
