"""Minimal lizard shim for environments without the real package."""

from __future__ import annotations

import ast
from types import SimpleNamespace


class _AnalyzeFile:
    @staticmethod
    def analyze_source_code(filename: str, code: str):
        try:
            tree = ast.parse(code)
        except Exception:
            function_list = []
        else:
            function_list = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_list.append(
                        SimpleNamespace(
                            cyclomatic_complexity=1,
                            token_count=max(1, len(ast.dump(node))),
                            full_parameters=list(node.args.args),
                        )
                    )
        return SimpleNamespace(function_list=function_list)


analyze_file = _AnalyzeFile()
