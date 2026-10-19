"""
Office2MarkdownProcessor
------------------
Convert CSV, Excel, Word (.docx) and PowerPoint (.pptx) files to Markdown,
preserving headings and table structure. Designed to mirror
pdf_to_markdown.PDF2MarkdownProcessor so both can feed the same ingestion step.

Library usage
    from Office2MarkdownProcessor import Office2MarkdownProcessor, OfficeConversionError

    processor = Office2MarkdownProcessor()
    text = processor.convert("report.docx")                       # -> str
    processor.convert("deck.pptx", "out/deck.md")                 # -> str, also written to disk
    units = processor.convert("book.xlsx", units=True)            # -> list[dict], one per sheet/slide/section
    report = processor.convert_directory("docs/", "md/", recursive=True)

CLI usage
    python Office2MarkdownProcessor.py report.docx -o report.md
    python Office2MarkdownProcessor.py docs/ -o md/ --recursive
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

__all__ = [
    "Office2MarkdownProcessor",
    "OfficeConversionError",
    "ConversionReport",
]

log = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xlsm", ".docx", ".pptx"}


class OfficeConversionError(Exception):
    """Raised when a file cannot be opened or converted."""


@dataclass
class ConversionReport:
    """Outcome of a batch conversion."""

    converted: list[Path] = field(default_factory=list)
    failed: dict[Path, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def _cell_text(value: Any) -> str:
    """Render a cell value as single-line Markdown-safe text."""
    if value is None:
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", "<br>").strip()


def _markdown_table(rows: list[list[Any]]) -> str:
    """Render rows as a Markdown table. The first row is treated as the header.

    Rows are padded to a common width so ragged input still produces valid Markdown.
    Empty tables return an empty string.
    """
    rows = [list(r) for r in rows if any(_cell_text(c) for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [[_cell_text(c) for c in r] + [""] * (width - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * width) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _heading(level: int, text: str) -> str:
    return f"{'#' * max(1, min(level, 6))} {text.strip()}"


# ---------------------------------------------------------------------------
# Format handlers. Each returns a list of "units" (dicts with 'text' and
# 'metadata'), so page/slide/sheet structure survives for RAG chunking.
# ---------------------------------------------------------------------------

def _convert_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(fh, dialect))
    text = _markdown_table(rows)
    return [{"text": text, "metadata": {"unit_type": "table", "unit_index": 1, "name": path.stem}}]


def _convert_excel(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    units: list[dict[str, Any]] = []
    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    for index, (name, frame) in enumerate(sheets.items(), start=1):
        frame = frame.dropna(how="all").dropna(axis=1, how="all")
        rows = frame.fillna("").values.tolist()
        body = _markdown_table(rows)
        text = _heading(2, f"Sheet: {name}") + ("\n\n" + body if body else "")
        units.append({"text": text, "metadata": {"unit_type": "sheet", "unit_index": index, "name": str(name)}})
    return units


def _convert_docx(path: Path) -> list[dict[str, Any]]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    parts: list[str] = []
    section_heading: Optional[str] = None

    # Iterate body elements in document order so tables sit between paragraphs.
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, doc)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name or "").lower() if para.style is not None else ""
            if style.startswith("heading") or style == "title":
                level = _style_level(style)
                section_heading = text
                parts.append(_heading(level, text))
            else:
                parts.append(text)
        elif tag == "tbl":
            table = Table(child, doc)
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            body = _markdown_table(rows)
            if body:
                parts.append(body)

    text = "\n\n".join(parts)
    return [{"text": text, "metadata": {"unit_type": "document", "unit_index": 1, "name": path.stem,
                                        "heading": section_heading}}]


def _style_level(style: str) -> int:
    if style == "title":
        return 1
    digits = "".join(ch for ch in style if ch.isdigit())
    return int(digits) if digits else 1


def _convert_pptx(path: Path) -> list[dict[str, Any]]:
    from pptx import Presentation
    from pptx.util import Emu

    prs = Presentation(str(path))
    units: list[dict[str, Any]] = []

    for index, slide in enumerate(prs.slides, start=1):
        title: Optional[str] = None
        if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
            title = slide.shapes.title.text_frame.text.strip() or None

        parts: list[str] = [_heading(2, title or f"Slide {index}")]
        for shape in _sorted_shapes(slide):
            if shape == slide.shapes.title:
                continue
            if getattr(shape, "has_table", False) and shape.has_table:
                rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
                body = _markdown_table(rows)
                if body:
                    parts.append(body)
            elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs).strip()
                    if text:
                        indent = "  " * para.level
                        parts.append(f"{indent}- {text}")

        notes = ""
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
        if notes:
            parts.append(f"**Speaker notes:** {notes}")

        units.append({
            "text": "\n\n".join(parts),
            "metadata": {"unit_type": "slide", "unit_index": index, "heading": title},
        })
    return units


def _sorted_shapes(slide: Any) -> list[Any]:
    """Order shapes top-to-bottom, left-to-right, so reading order is natural."""
    return sorted(slide.shapes, key=lambda s: (s.top or 0, s.left or 0))


_HANDLERS: dict[str, Callable[[Path], list[dict[str, Any]]]] = {
    ".csv": _convert_csv,
    ".xlsx": _convert_excel,
    ".xlsm": _convert_excel,
    ".docx": _convert_docx,
    ".pptx": _convert_pptx,
}


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class Office2MarkdownProcessor:
    """Converts CSV, Excel, Word and PowerPoint files to Markdown.

    Args:
        overwrite: If False, refuse to overwrite existing output files.
        encoding: Encoding used for output files.
    """

    def __init__(self, *, overwrite: bool = True, encoding: str = "utf-8") -> None:
        self.overwrite = overwrite
        self.encoding = encoding

    def convert(
        self,
        input_path: PathLike,
        output_path: Optional[PathLike] = None,
        *,
        units: bool = False,
    ) -> Union[str, list[dict[str, Any]]]:
        """Convert one file to Markdown.

        Args:
            input_path: Path to a .csv, .xlsx, .xlsm, .docx or .pptx file.
            output_path: If given, the Markdown is also written here (atomically).
            units: If True, return a list of dicts, one per sheet (Excel),
                slide (PowerPoint), or the whole document (Word/CSV), each with
                'text' and 'metadata'. Useful for RAG chunking.

        Returns:
            The Markdown text, or the list of units when ``units`` is True.

        Raises:
            FileNotFoundError: The input file does not exist.
            FileExistsError: The output exists and overwriting is disabled.
            OfficeConversionError: Unsupported format, unreadable file, or
                conversion failure.
        """
        src = Path(input_path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"No such file: {src}")

        suffix = src.suffix.lower()
        if suffix not in _HANDLERS:
            raise OfficeConversionError(
                f"Unsupported format {suffix!r}. Supported: {sorted(SUPPORTED_SUFFIXES)}"
            )

        dst = Path(output_path).expanduser().resolve() if output_path is not None else None
        if dst is not None and dst.exists() and not self.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {dst}")

        log.info("Converting %s", src)
        try:
            produced = _HANDLERS[suffix](src)
        except OfficeConversionError:
            raise
        except ImportError as exc:
            raise OfficeConversionError(
                f"Missing dependency for {suffix}: {exc}. See module docstring for install list."
            ) from exc
        except Exception as exc:  # the parsers raise many library-specific types
            raise OfficeConversionError(f"Conversion failed for {src}: {exc}") from exc

        if dst is not None:
            self._atomic_write(dst, self._join(produced))
            log.info("Wrote %s", dst)

        return produced if units else self._join(produced)

    def convert_directory(
        self,
        src_dir: PathLike,
        out_dir: PathLike,
        *,
        recursive: bool = False,
    ) -> ConversionReport:
        """Convert every supported file in a directory, mirroring the structure.

        A failure on one file is logged and recorded; the batch continues.
        """
        src_root = Path(src_dir).expanduser().resolve()
        out_root = Path(out_dir).expanduser().resolve()
        if not src_root.is_dir():
            raise NotADirectoryError(f"Not a directory: {src_root}")

        candidates = src_root.rglob("*") if recursive else src_root.glob("*")
        files = sorted(
            p for p in candidates
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
            and not p.name.startswith("~$")  # Office lock files
        )

        report = ConversionReport()
        for path in files:
            target = (out_root / path.relative_to(src_root)).with_suffix(".md")
            try:
                self.convert(path, target)
            except Exception as exc:  # one bad file must not stop the batch
                log.error("Failed to convert %s: %s", path, exc)
                report.failed[path] = str(exc)
            else:
                report.converted.append(target)

        log.info("Batch done: %d converted, %d failed", len(report.converted), len(report.failed))
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _join(produced: list[dict[str, Any]]) -> str:
        return "\n\n".join(unit["text"] for unit in produced if unit["text"])

    def _atomic_write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding=self.encoding, newline="\n") as fh:
                fh.write(text)
            os.replace(tmp_name, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_name)
            raise


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert CSV, Excel, Word and PowerPoint files to Markdown.")
    parser.add_argument("input", type=Path, help="a file or a directory")
    parser.add_argument("-o", "--output", type=Path,
                        help="output .md file (single file) or output directory (directory input)")
    parser.add_argument("--recursive", action="store_true", help="recurse into subdirectories")
    parser.add_argument("--no-overwrite", action="store_true", help="refuse to overwrite outputs")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    processor = Office2MarkdownProcessor(overwrite=not args.no_overwrite)
    try:
        if args.input.is_dir():
            if args.output is None:
                print("error: an output directory is required for directory input", file=sys.stderr)
                return 2
            report = processor.convert_directory(args.input, args.output, recursive=args.recursive)
            print(f"Converted {len(report.converted)} file(s), {len(report.failed)} failed.")
            return 1 if report.failed else 0

        output = args.output or args.input.with_suffix(".md")
        processor.convert(args.input, output)
        print(f"Wrote {output}")
        return 0

    except (OfficeConversionError, FileNotFoundError, FileExistsError,
            NotADirectoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def main() -> int:
    """Sample test: convert a CSV built in a temp dir and check table output."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "sample.csv"
        sample.write_text("name,score\nAda,91\nGrace|Hopper,88\n", encoding="utf-8")
        processor = Office2MarkdownProcessor()
        try:
            text = processor.convert(sample)
        except OfficeConversionError as exc:
            print(f"Test failed: {exc}")
            return 1

    assert "| name | score |" in text, "expected a Markdown header row"
    assert "Grace\\|Hopper" in text, "pipe characters in cells must be escaped"
    print("CSV test passed.")
    print(text)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(cli())
    sys.exit(main())