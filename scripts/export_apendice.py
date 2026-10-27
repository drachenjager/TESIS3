"""
Genera el contenido del APÉNDICE A para la tesis.

Produce DOS archivos en la carpeta raíz del proyecto:
  - apendice_a_estructura.txt   → árbol de carpetas (para pegar como texto)
  - apendice_a_codigo.txt       → todos los archivos con su contenido

Cómo usarlo:
    python scripts/export_apendice.py

Cómo pegarlo en Word:
    1. Abre apendice_a_codigo.txt
    2. Copia TODO el contenido (Ctrl+A → Ctrl+C)
    3. En Word, pega como TEXTO SIN FORMATO (Ctrl+Shift+V o "Mantener solo texto")
    4. Selecciona las líneas de código → cambia fuente a Courier New 9pt
    5. Las líneas con "══ ARCHIVO:" son los títulos de cada archivo
       → aplícales el estilo "Textoindependiente" de tu tesis
"""

import os
from pathlib import Path

# ── Configuración ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent   # raíz del proyecto (tesis-forecast/)

# Archivos/carpetas a EXCLUIR del apéndice
EXCLUDE_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "env",
    ".claude", "node_modules",
    "results", "preds", "out",   # salidas del banco de pruebas (benchmark/)
}
EXCLUDE_FILES = {
    ".gitkeep", ".DS_Store", "Thumbs.db",
    "apendice_a_codigo.txt", "apendice_a_estructura.txt",
}
EXCLUDE_EXTENSIONS = {
    ".pyc", ".pyo", ".ico", ".png", ".jpg", ".jpeg",
    ".svg", ".gif", ".pdf", ".xlsx", ".zip",
}
# Solo incluir estos tipos de archivos como código
CODE_EXTENSIONS = {
    ".py", ".txt", ".html", ".css", ".js",
    ".md", ".toml", ".yaml", ".yml", ".cfg",
    "", "Dockerfile", "Makefile",
}

SEPARATOR = "═" * 70


def should_include(path: Path) -> bool:
    if path.name in EXCLUDE_FILES:
        return False
    if path.suffix in EXCLUDE_EXTENSIONS:
        return False
    if path.suffix in CODE_EXTENSIONS or path.name in {"Dockerfile", "Makefile", ".gitignore", ".dockerignore"}:
        return True
    return False


def build_tree(directory: Path, prefix: str = "", is_last: bool = True) -> list[str]:
    """Construye el árbol de directorios estilo Unix."""
    lines = []
    entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
    entries = [e for e in entries if e.name not in EXCLUDE_DIRS and e.name not in EXCLUDE_FILES
               and (e.is_dir() or e.suffix not in EXCLUDE_EXTENSIONS)]

    for i, entry in enumerate(entries):
        is_entry_last = (i == len(entries) - 1)
        connector = "└── " if is_entry_last else "├── "
        name = entry.name + "/" if entry.is_dir() else entry.name
        lines.append(prefix + connector + name)

        if entry.is_dir() and entry.name not in EXCLUDE_DIRS:
            extension = "    " if is_entry_last else "│   "
            lines.extend(build_tree(entry, prefix + extension, is_entry_last))

    return lines


def collect_files(directory: Path) -> list[Path]:
    """Recopila archivos de código en orden lógico."""
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            # Excluir directorios ocultos/no deseados
            if any(part in EXCLUDE_DIRS for part in path.parts):
                continue
            if should_include(path):
                files.append(path)
    return files


def export():
    # ── 1. Árbol de estructura ─────────────────────────────────────────────
    tree_lines = [f"tesis-forecast/"] + build_tree(ROOT)
    tree_text = "\n".join(tree_lines)

    tree_path = ROOT / "apendice_a_estructura.txt"
    tree_path.write_text(tree_text, encoding="utf-8")
    print(f"✓ Árbol guardado en: {tree_path}")
    print()
    print(tree_text)
    print()

    # ── 2. Contenido de todos los archivos ─────────────────────────────────
    code_files = collect_files(ROOT)
    output_lines = []

    output_lines.append("APÉNDICE A — REPOSITORIO Y CÓDIGO FUENTE")
    output_lines.append(SEPARATOR)
    output_lines.append("")
    output_lines.append("Estructura del proyecto:")
    output_lines.append("")
    output_lines.append(tree_text)
    output_lines.append("")
    output_lines.append(SEPARATOR)
    output_lines.append("")

    for fpath in code_files:
        rel = fpath.relative_to(ROOT)
        output_lines.append(f"══ ARCHIVO: {rel} {'═' * max(0, 65 - len(str(rel)))}")
        output_lines.append("")
        try:
            content = fpath.read_text(encoding="utf-8")
            output_lines.extend(content.splitlines())
        except Exception as e:
            output_lines.append(f"[No se pudo leer: {e}]")
        output_lines.append("")
        output_lines.append(SEPARATOR)
        output_lines.append("")

    code_text = "\n".join(output_lines)
    code_path = ROOT / "apendice_a_codigo.txt"
    code_path.write_text(code_text, encoding="utf-8")
    print(f"✓ Código guardado en:  {code_path}")
    print(f"✓ Total de archivos:   {len(code_files)}")
    print()
    print("Próximos pasos para pegarlo en Word:")
    print("  1. Abre apendice_a_codigo.txt")
    print("  2. Ctrl+A → Ctrl+C")
    print("  3. En Word: Pegar → 'Mantener solo texto'")
    print("  4. Selecciona bloques de código → Courier New 9pt")
    print("  5. Líneas '══ ARCHIVO:' → estilo Textoindependiente (negrita)")


if __name__ == "__main__":
    export()
