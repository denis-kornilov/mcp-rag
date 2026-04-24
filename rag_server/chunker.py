from __future__ import annotations

import ast
import pathlib
import re
from typing import Dict, List, Tuple
from .error_reporter import ErrorReporter


error_reporter = ErrorReporter("rag_server.chunker")


def _py_symbols(src: str) -> List[Tuple[str, str, int, int]]:
    out: List[Tuple[str, str, int, int]] = []
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        error_reporter.warn(
            stage="chunker_py_symbols_parse",
            message="syntax error while extracting python symbols",
            exc=exc,
        )
        return out

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack: List[str] = []

        def _add(self, kind: str, node: ast.AST, name: str):
            start = getattr(node, "lineno", 1)
            end = getattr(node, "end_lineno", start)
            qual = ".".join(self.stack + [name]) if self.stack else name
            out.append((kind, qual, start, end))

        def visit_FunctionDef(self, node: ast.FunctionDef):
            self._add("function", node, node.name)
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self._add("function", node, node.name)
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_ClassDef(self, node: ast.ClassDef):
            self._add("class", node, node.name)
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

    Visitor().visit(tree)
    return out


def _find_block_end(lines: List[str], start_idx: int) -> int:
    """Return 1-based line number where the brace-block starting at start_idx ends.

    Counts { / } pairs. Returns start_idx+1 (same line, 1-based) when no
    opening brace is found (e.g. forward declaration).
    start_idx is a 0-based index into *lines*.
    """
    depth = 0
    found_open = False
    for i in range(start_idx, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                found_open = True
            elif ch == "}":
                depth -= 1
                if found_open and depth == 0:
                    return i + 1  # 1-based
    return start_idx + 1  # fallback


def _js_like_symbols(src: str) -> List[Tuple[str, str, int, int]]:
    out: List[Tuple[str, str, int, int]] = []
    lines = src.splitlines()
    for i, line in enumerate(lines, start=1):
        m1 = re.search(r"\bfunction\s+([A-Za-z0-9_]+)\s*\(", line)
        if m1:
            end = _find_block_end(lines, i - 1)
            out.append(("function", m1.group(1), i, end))
        m2 = re.search(r"\bclass\s+([A-Za-z0-9_]+)\b", line)
        if m2:
            end = _find_block_end(lines, i - 1)
            out.append(("class", m2.group(1), i, end))
    return out


def _py_module_overview(path: pathlib.Path, src: str) -> str:
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        error_reporter.warn(
            stage="chunker_py_module_overview_parse",
            symbol=path.as_posix(),
            message="syntax error while building module overview",
            exc=exc,
        )
        return f"Module {path.as_posix()}\nLanguage: python"

    module_doc = ast.get_docstring(tree) or ""
    comment_lines: List[str] = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            comment_lines.append(stripped)
    imports: List[str] = []
    globals_out: List[str] = []
    classes: List[str] = []
    functions: List[str] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            try:
                imports.append(ast.get_source_segment(src, node) or "")
            except Exception as exc:
                error_reporter.warn(
                    stage="chunker_extract_import_segment",
                    symbol=path.as_posix(),
                    message="failed to extract import source segment",
                    exc=exc,
                )
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            raw_targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in raw_targets:
                if isinstance(target, ast.Name):
                    targets.append(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for item in target.elts:
                        if isinstance(item, ast.Name):
                            targets.append(item.id)
            globals_out.extend(targets)

    lines = [
        f"Module: {path.as_posix()}",
        "Language: python",
    ]
    if module_doc:
        lines.append(f"ModuleDoc: {module_doc}")
    if comment_lines:
        lines.append("Comments:")
        lines.extend(comment_lines[:40])
    if imports:
        lines.append("Imports:")
        lines.extend(imports[:40])
    if globals_out:
        lines.append(f"Globals: {', '.join(dict.fromkeys(globals_out))}")
    if classes:
        lines.append(f"Classes: {', '.join(classes)}")
    if functions:
        lines.append(f"Functions: {', '.join(functions)}")
    return "\n".join(lines)


def _js_module_overview(path: pathlib.Path, src: str) -> str:
    lines = src.splitlines()
    imports: List[str] = []
    globals_out: List[str] = []
    classes: List[str] = []
    functions: List[str] = []
    comments: List[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            comments.append(stripped)
        if re.match(r"^(import\b|export\s+\{?|export\s+\*\s+from|const\s+\w+\s*=\s*require\()", stripped):
            imports.append(stripped)
        m_fun = re.search(r"\bfunction\s+([A-Za-z0-9_]+)\s*\(", stripped)
        if m_fun:
            functions.append(m_fun.group(1))
        m_cls = re.search(r"\bclass\s+([A-Za-z0-9_]+)\b", stripped)
        if m_cls:
            classes.append(m_cls.group(1))
        m_global = re.search(r"^(export\s+)?(const|let|var)\s+([A-Za-z0-9_]+)\s*=", stripped)
        if m_global and "=>" not in stripped:
            globals_out.append(m_global.group(3))

    lines_out = [
        f"Module: {path.as_posix()}",
        "Language: javascript/typescript",
    ]
    if imports:
        lines_out.append("Imports:")
        lines_out.extend(imports[:40])
    if comments:
        lines_out.append("Comments:")
        lines_out.extend(comments[:40])
    if globals_out:
        lines_out.append(f"Globals: {', '.join(dict.fromkeys(globals_out))}")
    if classes:
        lines_out.append(f"Classes: {', '.join(dict.fromkeys(classes))}")
    if functions:
        lines_out.append(f"Functions: {', '.join(dict.fromkeys(functions))}")
    return "\n".join(lines_out)


def _split_markdown_sections(path: pathlib.Path, content: str) -> List[Dict]:
    lines = content.splitlines()
    sections: List[Dict] = []
    current_title = "Document"
    current_start = 1
    current_lines: List[str] = []

    def flush(end_line: int) -> None:
        nonlocal current_lines, current_start, current_title
        text = "\n".join(current_lines).strip()
        if not text:
            return
        sections.append({
            "text": f"Document: {path.as_posix()}\nSection: {current_title}\n\n{text}",
            "metadata": {
                "type": "doc_section",
                "path": path.as_posix(),
                "language": "markdown",
                "symbol": current_title,
                "start_line": current_start,
                "end_line": end_line,
            },
        })

    for idx, line in enumerate(lines, start=1):
        if re.match(r"^\s{0,3}#{1,6}\s+", line):
            flush(idx - 1)
            current_title = line.lstrip("#").strip()
            current_start = idx
            current_lines = [line]
        else:
            current_lines.append(line)
    flush(len(lines))
    if not sections and content.strip():
        sections.append({
            "text": f"Document: {path.as_posix()}\nSection: Document\n\n{content.strip()}",
            "metadata": {
                "type": "doc_section",
                "path": path.as_posix(),
                "language": "markdown",
                "symbol": "Document",
                "start_line": 1,
                "end_line": len(lines) or 1,
            },
        })
    return sections


def _split_text_sections(path: pathlib.Path, content: str) -> List[Dict]:
    lines = content.splitlines()
    sections: List[Dict] = []
    start = None
    buf: List[str] = []

    def flush(end_line: int) -> None:
        nonlocal start, buf
        text = "\n".join(buf).strip()
        if not text or start is None:
            return
        preview = text.splitlines()[0][:80]
        sections.append({
            "text": f"Document: {path.as_posix()}\nBlock: {preview}\n\n{text}",
            "metadata": {
                "type": "doc_block",
                "path": path.as_posix(),
                "language": "text",
                "symbol": preview,
                "start_line": start,
                "end_line": end_line,
            },
        })
        start = None
        buf = []

    for idx, line in enumerate(lines, start=1):
        if line.strip():
            if start is None:
                start = idx
            buf.append(line)
        else:
            flush(idx - 1)
    flush(len(lines))
    if not sections and content.strip():
        sections.append({
            "text": f"Document: {path.as_posix()}\nBlock: Document\n\n{content.strip()}",
            "metadata": {
                "type": "doc_block",
                "path": path.as_posix(),
                "language": "text",
                "symbol": "Document",
                "start_line": 1,
                "end_line": len(lines) or 1,
            },
        })
    return sections


def _top_level_kv_sections(path: pathlib.Path, content: str, language: str) -> List[Dict]:
    lines = content.splitlines()
    sections: List[Dict] = []
    current_key = "root"
    current_start = 1
    current_lines: List[str] = []

    def flush(end_line: int) -> None:
        text = "\n".join(current_lines).strip()
        if not text:
            return
        sections.append({
            "text": f"Document: {path.as_posix()}\nSection: {current_key}\n\n{text}",
            "metadata": {
                "type": "config_section",
                "path": path.as_posix(),
                "language": language,
                "symbol": current_key,
                "start_line": current_start,
                "end_line": end_line,
            },
        })

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        is_top_level = bool(re.match(r"^[A-Za-z0-9_.-]+\s*:", line)) and (len(line) - len(line.lstrip(" ")) == 0)
        is_json_top_level = bool(re.match(r'^\s*"[^"]+"\s*:', line)) and (len(line) - len(line.lstrip(" ")) <= 2)
        if is_top_level or is_json_top_level:
            flush(idx - 1)
            current_key = stripped.split(":", 1)[0].strip().strip('"')
            current_start = idx
            current_lines = [line]
        else:
            current_lines.append(line)
    flush(len(lines))
    if not sections and content.strip():
        sections.append({
            "text": f"Document: {path.as_posix()}\nSection: root\n\n{content.strip()}",
            "metadata": {
                "type": "config_section",
                "path": path.as_posix(),
                "language": language,
                "symbol": "root",
                "start_line": 1,
                "end_line": len(lines) or 1,
            },
        })
    return sections


def _split_env_sections(path: pathlib.Path, content: str) -> List[Dict]:
    secret_markers = ("key", "token", "secret", "password", "passwd")
    out_lines: List[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out_lines.append(line)
            continue
        key, value = stripped.split("=", 1)
        if any(marker in key.lower() for marker in secret_markers):
            out_lines.append(f"{key}=<redacted>")
        else:
            out_lines.append(f"{key}={value}")
    safe = "\n".join(out_lines).strip()
    return [{
        "text": f"Env file: {path.as_posix()}\n\n{safe}",
        "metadata": {
            "type": "env",
            "path": path.as_posix(),
            "language": "dotenv",
        },
    }] if safe else []


def chunk_file(path: pathlib.Path, content: str) -> List[Dict]:
    """Return a list of chunks for a source file with metadata.

    Each item: {text, metadata}
    metadata keys: type, path, language, symbol, start_line, end_line
    """
    suffix = path.suffix.lower()
    language = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".tsx": "tsx", ".json": "json", ".md": "markdown",
        ".html": "html", ".css": "css",
    }.get(suffix, suffix.lstrip("."))

    lines = content.splitlines()

    def slice_segment(start: int, end: int) -> str:
        start = max(1, start)
        end = max(start, end)
        return "\n".join(lines[start - 1 : end])

    items: List[Dict] = []
    if suffix == ".py":
        symbols = _py_symbols(content)
        items.append({
            "text": _py_module_overview(path, content),
            "metadata": {
                "type": "module",
                "path": path.as_posix(),
                "language": language,
            },
        })
        if symbols:
            for kind, qual, start, end in symbols:
                text = slice_segment(start, end)
                items.append({
                    "text": text,
                    "metadata": {
                        "type": kind,
                        "path": path.as_posix(),
                        "language": language,
                        "symbol": qual,
                        "start_line": start,
                        "end_line": end,
                    },
                })
        else:
            items.append({
                "text": content,
                "metadata": {
                    "type": "file",
                    "path": path.as_posix(),
                    "language": language,
                },
            })
        return items

    # JS/TS heuristic
    if suffix in {".js", ".ts", ".tsx"}:
        symbols = _js_like_symbols(content)
        items.append({
            "text": _js_module_overview(path, content),
            "metadata": {
                "type": "module",
                "path": path.as_posix(),
                "language": language,
            },
        })
        if symbols:
            for kind, name, start, end in symbols:
                text = slice_segment(start, end)
                items.append({
                    "text": text,
                    "metadata": {
                        "type": kind,
                        "path": path.as_posix(),
                        "language": language,
                        "symbol": name,
                        "start_line": start,
                        "end_line": end,
                    },
                })
        else:
            items.append({
                "text": content,
                "metadata": {
                    "type": "file",
                    "path": path.as_posix(),
                    "language": language,
                },
            })
        return items

    if suffix == ".md":
        return _split_markdown_sections(path, content)

    if suffix == ".txt":
        return _split_text_sections(path, content)

    if suffix in {".yaml", ".yml"}:
        return _top_level_kv_sections(path, content, language="yaml")

    if suffix == ".json":
        return _top_level_kv_sections(path, content, language="json")

    if path.name.lower() in {".env", ".env.example", ".env.sample", ".env.template"} or path.name.lower().startswith(".env."):
        return _split_env_sections(path, content)

    # Default: one chunk per file
    items.append({
        "text": content,
        "metadata": {
            "type": "file",
            "path": path.as_posix(),
            "language": language,
        },
    })
    return items
