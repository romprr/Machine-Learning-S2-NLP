from __future__ import annotations

from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report.md"
OUTPUT = ROOT / "Stack_Overflow_Tag_Classification_Report.pdf"

PAGE_WIDTH = 8.27
PAGE_HEIGHT = 11.69
LEFT = 0.08
RIGHT = 0.92
TOP = 0.95
BOTTOM = 0.06
LINE_STEP = 0.021
TABLE_STEP = 0.019


def clean_inline(text: str) -> str:
    text = text.replace("`", "")
    text = text.replace("### ", "")
    text = text.replace("## ", "")
    text = text.replace("# ", "")
    return text.rstrip()


def is_table_line(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def format_table_block(lines: list[str]) -> list[str]:
    rows = [split_table_row(line) for line in lines if not re.fullmatch(r"\|\s*[-: ]+\|.*", line.strip())]
    if not rows:
        return []
    widths = [0] * max(len(row) for row in rows)
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(clean_inline(cell)))
    formatted: list[str] = []
    for row_idx, row in enumerate(rows):
        padded = []
        for idx, cell in enumerate(row):
            padded.append(clean_inline(cell).ljust(widths[idx]))
        formatted.append("  ".join(padded).rstrip())
        if row_idx == 0:
            formatted.append("  ".join("-" * w for w in widths).rstrip())
    return formatted


def markdown_to_blocks(text: str) -> list[tuple[str, list[str]]]:
    lines = text.splitlines()
    blocks: list[tuple[str, list[str]]] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].rstrip()
        if not line:
            blocks.append(("blank", [""]))
            idx += 1
            continue
        if is_table_line(line):
            table_lines = []
            while idx < len(lines) and is_table_line(lines[idx].rstrip()):
                table_lines.append(lines[idx].rstrip())
                idx += 1
            blocks.append(("table", format_table_block(table_lines)))
            continue
        if line.startswith("# "):
            blocks.append(("h1", [clean_inline(line)]))
        elif line.startswith("## "):
            blocks.append(("h2", [clean_inline(line)]))
        elif line.startswith("- "):
            blocks.append(("bullet", [clean_inline(line[2:])]))
        elif re.match(r"^\d+\.\s", line):
            blocks.append(("number", [clean_inline(line)]))
        elif line.strip() == "---":
            blocks.append(("blank", [""]))
        else:
            blocks.append(("p", [clean_inline(line)]))
        idx += 1
    return blocks


def wrap_block(kind: str, lines: list[str]) -> list[str]:
    if kind == "table":
        return lines
    width = 88
    if kind == "h1":
        width = 60
    elif kind == "h2":
        width = 72
    elif kind in {"bullet", "number"}:
        width = 82
    wrapped: list[str] = []
    for line in lines:
        if kind == "bullet":
            wrapped.extend(textwrap.wrap(f"- {line}", width=width, subsequent_indent="  ") or ["-"])
        elif kind == "number":
            wrapped.extend(textwrap.wrap(line, width=width, subsequent_indent="   ") or [""])
        else:
            wrapped.extend(textwrap.wrap(line, width=width) or [""])
    return wrapped


def draw_page(page_items: list[tuple[str, list[str]]], pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    y = TOP
    for kind, lines in page_items:
        if kind == "blank":
            y -= LINE_STEP * 0.6
            continue
        if kind == "h1":
            for line in lines:
                ax.text(0.5, y, line, ha="center", va="top", fontsize=20, fontweight="bold", family="DejaVu Sans")
                y -= LINE_STEP * 1.7
            continue
        if kind == "h2":
            for line in lines:
                ax.text(LEFT, y, line, ha="left", va="top", fontsize=14, fontweight="bold", family="DejaVu Sans")
                y -= LINE_STEP * 1.4
            continue
        fontsize = 10.5
        family = "DejaVu Sans"
        step = LINE_STEP
        if kind == "table":
            family = "DejaVu Sans Mono"
            fontsize = 9.3
            step = TABLE_STEP
        for line in lines:
            ax.text(LEFT, y, line, ha="left", va="top", fontsize=fontsize, family=family)
            y -= step
    pdf.savefig(fig)
    plt.close(fig)


def paginate(blocks: list[tuple[str, list[str]]]) -> list[list[tuple[str, list[str]]]]:
    pages: list[list[tuple[str, list[str]]]] = []
    current: list[tuple[str, list[str]]] = []
    y = TOP
    for kind, lines in blocks:
        wrapped = wrap_block(kind, lines)
        if kind == "blank":
            needed = LINE_STEP * 0.6
        elif kind == "h1":
            needed = len(wrapped) * LINE_STEP * 1.7
        elif kind == "h2":
            needed = len(wrapped) * LINE_STEP * 1.4
        elif kind == "table":
            needed = len(wrapped) * TABLE_STEP
        else:
            needed = len(wrapped) * LINE_STEP
        if y - needed < BOTTOM and current:
            pages.append(current)
            current = []
            y = TOP
        current.append((kind, wrapped))
        y -= needed
    if current:
        pages.append(current)
    return pages


def main() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    blocks = markdown_to_blocks(markdown)
    pages = paginate(blocks)
    with PdfPages(OUTPUT) as pdf:
        for page in pages:
            draw_page(page, pdf)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
