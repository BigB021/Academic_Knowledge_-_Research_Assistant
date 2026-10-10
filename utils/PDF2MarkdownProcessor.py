"""
pdf_to_markdown
---------------
Convert PDF files to Markdown with pymupdf4llm. Usable as a library
(e.g. the ingestion step of a RAG pipeline) or from the command line.

Library usage
    from pdf_to_markdown import PDF2MarkdownProcessor, PDFConversionError

    processor = PDF2MarkdownProcessor()
    text = processor.convert("paper.pdf")                          # -> str
    processor.convert("paper.pdf", "out/paper.md")                 # -> str, also written to disk
    pages = processor.convert("paper.pdf", page_chunks=True)       # -> list[dict], one per page

    report = processor.convert_directory("docs/", "md/", recursive=True)

CLI usage
    python pdf_to_markdown.py paper.pdf -o paper.md
    python pdf_to_markdown.py docs/ -o md/ --recursive --images

Running with no arguments executes the sample test in main().
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import pymupdf
import pymupdf4llm

__all__ = [
    "PDF2MarkdownProcessor",
    "PDFConversionError",
    "ConversionReport",
]

log = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]


class PDFConversionError(Exception):
    """Raised when a PDF cannot be opened or converted."""


@dataclass
class ConversionReport:
    """Outcome of a batch conversion."""

    converted: list[Path] = field(default_factory=list)
    failed: dict[Path, str] = field(default_factory=dict)


class PDF2MarkdownProcessor:
    """Converts PDFs to Markdown using pymupdf4llm.

    Settings passed to the constructor apply to every call; per-call
    arguments override them.

    Args:
        overwrite: If False, refuse to overwrite existing output files.
        encoding: Encoding used for output files.
        **default_options: Default keyword arguments forwarded to
            ``pymupdf4llm.to_markdown`` (e.g. ``dpi``, ``header``, ``footer``).
    """

    def __init__(
        self,
        *,
        overwrite: bool = True,
        encoding: str = "utf-8",
        **default_options: Any,
    ) -> None:
        self.overwrite = overwrite
        self.encoding = encoding
        self.default_options = default_options

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        pdf_path: PathLike,
        output_path: Optional[PathLike] = None,
        *,
        pages: Optional[Iterable[int]] = None,
        page_chunks: bool = False,
        write_images: bool = False,
        image_dir: Optional[PathLike] = None,
        **options: Any,
    ) -> Union[str, list[dict[str, Any]]]:
        """Convert a single PDF to Markdown.

        Args:
            pdf_path: Path to the input PDF.
            output_path: If given, the Markdown is also written here (atomically).
                Parent directories are created as needed.
            pages: 0-based page indices to convert. ``None`` converts all pages.
            page_chunks: If True, return a list with one dict per page (keys
                include ``"text"`` and ``"metadata"``) instead of one string.
            write_images: Extract embedded images to disk.
            image_dir: Directory for extracted images. Defaults to
                ``<output stem>_images`` next to ``output_path``.
            **options: Extra keyword arguments forwarded to
                ``pymupdf4llm.to_markdown``, overriding the defaults.

        Returns:
            The Markdown text, or a list of per-page dicts when ``page_chunks`` is True.

        Raises:
            FileNotFoundError: The input file does not exist.
            FileExistsError: The output exists and overwriting is disabled.
            PDFConversionError: The file is not a readable PDF, is password
                protected, or the conversion itself failed.
            ValueError: Invalid argument combination.
        """
        src = Path(pdf_path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"No such PDF: {src}")

        dst = Path(output_path).expanduser().resolve() if output_path is not None else None
        if dst is not None and dst.exists() and not self.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {dst}")

        self._check_pdf_readable(src)

        img_dir: Optional[Path] = None
        if write_images:
            if image_dir is not None:
                img_dir = Path(image_dir).expanduser().resolve()
            elif dst is not None:
                img_dir = dst.parent / f"{dst.stem}_images"
            else:
                raise ValueError("write_images=True requires either image_dir or output_path")
            img_dir.mkdir(parents=True, exist_ok=True)

        call_kwargs: dict[str, Any] = {
            **self.default_options,
            **options,
            "page_chunks": page_chunks,
            "write_images": write_images,
        }
        if pages is not None:
            call_kwargs["pages"] = list(pages)
        if img_dir is not None:
            call_kwargs["image_path"] = str(img_dir)

        log.info("Converting %s", src)
        try:
            result = pymupdf4llm.to_markdown(str(src), **call_kwargs)
        except Exception as exc:  # pymupdf raises a variety of exception types
            raise PDFConversionError(f"Conversion failed for {src}: {exc}") from exc

        if dst is not None:
            text = self._join_chunks(result) if page_chunks else result
            self._atomic_write(dst, text)
            log.info("Wrote %s", dst)

        return result

    def convert_directory(
        self,
        src_dir: PathLike,
        out_dir: PathLike,
        *,
        recursive: bool = False,
        **options: Any,
    ) -> ConversionReport:
        """Convert every PDF in a directory to Markdown.

        The directory structure is mirrored under ``out_dir``. A failure on one
        file is logged and recorded in the report; the batch continues.

        Keyword arguments are forwarded to :meth:`convert`.
        """
        src_root = Path(src_dir).expanduser().resolve()
        out_root = Path(out_dir).expanduser().resolve()
        if not src_root.is_dir():
            raise NotADirectoryError(f"Not a directory: {src_root}")

        candidates = src_root.rglob("*") if recursive else src_root.glob("*")
        pdfs = sorted(p for p in candidates if p.is_file() and p.suffix.lower() == ".pdf")

        report = ConversionReport()
        for pdf in pdfs:
            target = (out_root / pdf.relative_to(src_root)).with_suffix(".md")
            try:
                self.convert(pdf, target, **options)
            except Exception as exc:  # one bad file must not stop the batch
                log.error("Failed to convert %s: %s", pdf, exc)
                report.failed[pdf] = str(exc)
            else:
                report.converted.append(target)

        log.info("Batch done: %d converted, %d failed", len(report.converted), len(report.failed))
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def parse_pages(spec: str) -> list[int]:
        """Parse a 0-based page spec such as ``"0,2,5-8"``."""
        pages: list[int] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                pages.extend(range(int(start), int(end) + 1))
            else:
                pages.append(int(part))
        return pages

    @staticmethod
    def _check_pdf_readable(path: Path) -> None:
        """Fail early with a clear message if the file is not an openable PDF."""
        try:
            with pymupdf.open(path) as doc:
                if doc.needs_pass:
                    raise PDFConversionError(f"PDF is password-protected: {path}")
        except PDFConversionError:
            raise
        except Exception as exc:
            raise PDFConversionError(f"Not a readable PDF: {path} ({exc})") from exc

    @staticmethod
    def _join_chunks(chunks: list[dict[str, Any]]) -> str:
        return "\n\n".join(chunk["text"] for chunk in chunks)

    def _atomic_write(self, path: Path, text: str) -> None:
        """Write via a temp file and rename, so readers never see a partial file."""
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
    parser = argparse.ArgumentParser(description="Convert PDF files to Markdown.")
    parser.add_argument("input", type=Path, help="a PDF file or a directory of PDFs")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="output .md file (single PDF) or output directory (directory input)",
    )
    parser.add_argument("--pages", help="0-based page indices, e.g. '0,2,5-8'")
    parser.add_argument("--images", action="store_true", help="extract images next to the output")
    parser.add_argument("--recursive", action="store_true", help="recurse into subdirectories")
    parser.add_argument("--no-overwrite", action="store_true", help="refuse to overwrite outputs")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    processor = PDF2MarkdownProcessor(overwrite=not args.no_overwrite)
    try:
        pages = PDF2MarkdownProcessor.parse_pages(args.pages) if args.pages else None

        if args.input.is_dir():
            if args.output is None:
                print("error: an output directory is required for directory input", file=sys.stderr)
                return 2
            report = processor.convert_directory(
                args.input,
                args.output,
                recursive=args.recursive,
                pages=pages,
                write_images=args.images,
            )
            print(f"Converted {len(report.converted)} file(s), {len(report.failed)} failed.")
            return 1 if report.failed else 0

        output = args.output or args.input.with_suffix(".md")
        processor.convert(args.input, output, pages=pages, write_images=args.images)
        print(f"Wrote {output}")
        return 0

    except (PDFConversionError, FileNotFoundError, FileExistsError,
            NotADirectoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def main() -> int:
    """Sample test: convert the sample PDF into per-page chunks."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Datasets", "pdfs")
    pdf_path = os.path.join(pdf_dir, "2410.01574v4.pdf")
    output_path = os.path.join(pdf_dir, "result.md")

    processor = PDF2MarkdownProcessor()
    try:
        chunks = processor.convert(pdf_path, output_path, page_chunks=True)
    except (FileNotFoundError, PDFConversionError) as exc:
        print(f"Test failed: {exc}")
        return 1

    assert isinstance(chunks, list) and chunks, "expected a non-empty list of page chunks"
    assert all("text" in chunk for chunk in chunks), "every chunk should contain 'text'"

    print(f"Converted {len(chunks)} page(s) from {os.path.basename(pdf_path)}")
    print(f"Markdown written to {output_path}")
    for index, chunk in enumerate(chunks[:2]):
        preview = chunk["text"][:300].replace("\n", " ")
        print(f"\n--- chunk {index} preview ---\n{preview}...")

    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(cli())
    sys.exit(main())