# Author: Youssef Aitbouddroub
# Robust PDF Preprocessing Module for Academic RAG Systems
# Supports: text extraction, OCR fallback, bilingual (EN/FR) processing, structured output

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

import pymupdf
import numpy as np
from tqdm.auto import tqdm

# NLP imports
from spacy.lang.en import English
from spacy.lang.fr import French

# Language detection
try:
    from langdetect import detect, LangDetectException
except ImportError:
    warnings.warn("langdetect not installed. Language detection will be disabled.")
    detect = None

# OCR support (optional)
try:
    import pytesseract
    from PIL import Image
    import io
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    warnings.warn("Tesseract/pytesseract not installed. OCR fallback disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PDFMetadata:
    """Extracted PDF metadata"""
    title: Optional[str]
    author: Optional[str]
    creation_date: Optional[str]
    num_pages: int
    language: Optional[str]
    has_text: bool
    required_ocr: bool


@dataclass
class PageContent:
    """Structured page content"""
    page_number: int
    text: str
    language: Optional[str]
    raw_length: int
    cleaned_length: int
    sentences: List[str]
    num_sentences: int
    tokens_estimate: float
    extraction_method: str  # 'text' or 'ocr'
    metadata: Dict


class PDFProcessor:
    """Robust PDF text extraction and preprocessing for RAG systems."""
    
    def __init__(self, 
                 enable_ocr: bool = True,
                 use_layout: bool = False,
                 clean_text: bool = True):
        """
        Initialize PDF processor.
        
        Args:
            enable_ocr: Enable OCR fallback for scanned PDFs
            use_layout: Use layout-aware text extraction (slower but preserves structure)
            clean_text: Apply aggressive cleaning to extracted text
        """
        self.enable_ocr = enable_ocr and TESSERACT_AVAILABLE
        self.use_layout = use_layout
        self.clean_text = clean_text
        
        # Initialize NLP pipelines (lazy load)
        self._nlp_en = None
        self._nlp_fr = None
        
        if not self.enable_ocr and enable_ocr:
            logger.warning("OCR requested but Tesseract not installed. Install via: apt-get install tesseract-ocr")
    
    @property
    def nlp_en(self):
        """Lazy load English spacy model"""
        if self._nlp_en is None:
            self._nlp_en = English()
            self._nlp_en.add_pipe("sentencizer")
        return self._nlp_en
    
    @property
    def nlp_fr(self):
        """Lazy load French spacy model"""
        if self._nlp_fr is None:
            self._nlp_fr = French()
            self._nlp_fr.add_pipe("sentencizer")
        return self._nlp_fr
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect language of text (en or fr)"""
        if not text.strip() or detect is None:
            return None
        try:
            lang = detect(text[:200])  # Sample first 200 chars
            return lang if lang in ['en', 'fr'] else None
        except LangDetectException:
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text while preserving structure"""
        if not text:
            return ""
        
        # Preserve paragraph breaks
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove excessive whitespace
            line = ' '.join(line.split())
            if line.strip():
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Fix common PDF artifacts
        text = text.replace('\x00', '')  # Null bytes
        text = text.replace('  ', ' ')   # Double spaces
        
        # Remove control characters but preserve newlines
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
        
        return text.strip()
    
    def _extract_text_standard(self, page: pymupdf.Page) -> str:
        """Extract text using standard method"""
        if self.use_layout:
            return page.get_text("blocks")
        return page.get_text("text")
    
    def _extract_text_ocr(self, page: pymupdf.Page) -> Optional[str]:
        """Extract text using OCR (fallback for scanned documents)"""
        if not self.enable_ocr:
            return None
        
        try:
            # Render page to image
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # 2x zoom for better OCR
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # OCR
            text = pytesseract.image_to_string(img, lang='fra+eng')
            return text if text.strip() else None
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return None
    
    def _detect_scanned_pdf(self, page: pymupdf.Page) -> bool:
        """Heuristic to detect if page is scanned (low text content)"""
        text = page.get_text("text")
        # If text is <10% of what we'd expect from page area, likely scanned
        return len(text.split()) < 50
    
    def _extract_page_content(self, page: pymupdf.Page, page_num: int) -> PageContent:
        """Extract structured content from a single page"""
        # Initial text extraction
        text = self._extract_text_standard(page)
        extraction_method = "text"
        
        # Detect if scanned and use OCR if enabled
        if self._detect_scanned_pdf(page) and self.enable_ocr:
            ocr_text = self._extract_text_ocr(page)
            if ocr_text:
                text = ocr_text
                extraction_method = "ocr"
                logger.debug(f"Page {page_num + 1}: Used OCR extraction")
        
        # Clean text
        raw_length = len(text)
        if self.clean_text:
            text = self._clean_text(text)
        cleaned_length = len(text)
        
        # Detect language
        language = self._detect_language(text)
        
        # Extract sentences
        sentences = self._extract_sentences(text, language)
        
        # Token estimation (average 1.33 tokens per word)
        token_estimate = len(text.split()) * 1.33
        
        return PageContent(
            page_number=page_num + 1,
            text=text,
            language=language,
            raw_length=raw_length,
            cleaned_length=cleaned_length,
            sentences=sentences,
            num_sentences=len(sentences),
            tokens_estimate=token_estimate,
            extraction_method=extraction_method,
            metadata={}
        )
    
    def _extract_sentences(self, text: str, language: Optional[str]) -> List[str]:
        """Extract sentences using appropriate language model"""
        if not text.strip():
            return []
        
        try:
            # Use detected language or default to English
            if language == 'fr':
                doc = self.nlp_fr(text)
            else:
                doc = self.nlp_en(text)
            
            return [sent.text.strip() for sent in doc.sents]
        except Exception as e:
            logger.warning(f"Sentence extraction failed: {e}")
            # Fallback: split by periods
            return [s.strip() for s in text.split('.') if s.strip()]
    
    def _extract_metadata(self, pdf_doc: pymupdf.Document) -> PDFMetadata:
        """Extract PDF metadata"""
        metadata = pdf_doc.metadata or {}
        
        # Get title from metadata or filename
        title = metadata.get('title', '') or pdf_doc.name
        
        # Creation date
        creation_date = None
        if 'creationDate' in metadata:
            try:
                # Parse PDF date format
                date_str = metadata['creationDate']
                if isinstance(date_str, str):
                    creation_date = date_str[:19]  # Take first part
            except:
                pass
        
        # Check if PDF has text
        has_text = False
        requires_ocr = False
        
        for page in pdf_doc:
            text = page.get_text("text")
            if len(text.split()) > 50:
                has_text = True
                break
        
        if not has_text:
            requires_ocr = True
        
        return PDFMetadata(
            title=title,
            author=metadata.get('author', None),
            creation_date=creation_date,
            num_pages=len(pdf_doc),
            language=None,  # Detected during processing
            has_text=has_text,
            required_ocr=requires_ocr
        )
    
    def process_pdf(self, pdf_path: str) -> Tuple[List[PageContent], PDFMetadata]:
        """
        Process a PDF file and extract structured text content.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (pages_content, metadata)
            
        Raises:
            FileNotFoundError: If PDF doesn't exist
            RuntimeError: If PDF processing fails
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        try:
            doc = pymupdf.open(pdf_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF {pdf_path}: {e}")
        
        try:
            # Extract metadata
            metadata = self._extract_metadata(doc)
            logger.info(f"PDF has {metadata.num_pages} pages. Text: {metadata.has_text}, OCR needed: {metadata.required_ocr}")
            
            # Extract content from each page
            pages_content = []
            for page_num in tqdm(range(len(doc)), desc=f"Extracting text"):
                try:
                    page = doc[page_num]
                    page_content = self._extract_page_content(page, page_num)
                    pages_content.append(page_content)
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {e}")
                    continue
            
            # Detect dominant language
            if pages_content:
                languages = [p.language for p in pages_content if p.language]
                if languages:
                    metadata.language = max(set(languages), key=languages.count)
            
            logger.info(f"Successfully processed {len(pages_content)}/{metadata.num_pages} pages")
            
            return pages_content, metadata
            
        finally:
            doc.close()
    
    def process_pdfs_batch(self, pdf_dir: str) -> Dict[str, Tuple[List[PageContent], PDFMetadata]]:
        """
        Process all PDFs in a directory.
        
        Args:
            pdf_dir: Directory containing PDFs
            
        Returns:
            Dictionary mapping filenames to (pages_content, metadata)
        """
        pdf_dir = Path(pdf_dir)
        results = {}
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                pages, metadata = self.process_pdf(str(pdf_path))
                results[pdf_path.name] = (pages, metadata)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
        
        return results


# =============================================================================
# CONVENIENCE FUNCTIONS (for backward compatibility and ease of use)
# =============================================================================

def open_read_pdf(pdf_path: str, enable_ocr: bool = True) -> List[PageContent]:
    """
    Simple function to extract text from PDF.
    
    Args:
        pdf_path: Path to PDF file
        enable_ocr: Enable OCR for scanned documents
        
    Returns:
        List of PageContent objects
    """
    processor = PDFProcessor(enable_ocr=enable_ocr)
    pages, _ = processor.process_pdf(pdf_path)
    return pages


def extract_sentences_from_text(pages_content: List[PageContent]) -> List[PageContent]:
    """
    Extract sentences from pages (modifies in place).
    
    Args:
        pages_content: List of PageContent objects
        
    Returns:
        Same list with sentences extracted
    """
    # Sentences are already extracted in PageContent, this is for compatibility
    return pages_content


# =============================================================================
# TESTING AND EXAMPLES
# =============================================================================

if __name__ == "__main__":
    import pandas as pd
    import random
    import sys
    
    print("=" * 80)
    print("PDF Preprocessing Module - Test Suite")
    print("=" * 80)
    
    # Test configuration
    pdf_dir = os.path.join(os.path.dirname(__file__), "..", "Datasets", "pdfs")
    
    # Check if test directory exists
    if not os.path.exists(pdf_dir):
        print(f"\n[!]  Test PDF directory not found: {pdf_dir}")
        print("Creating synthetic test example instead...\n")
        
        # Create a synthetic test
        processor = PDFProcessor(enable_ocr=False)
        
        # Test the processor initialization
        print(f"[+] Processor initialized successfully")
        print(f"  - OCR enabled: {processor.enable_ocr}")
        print(f"  - Text cleaning: {processor.clean_text}")
        print(f"  - English model: {processor.nlp_en is None}")
        print(f"  - French model: {processor.nlp_fr is None}")
        
        print("\n" + "=" * 80)
        print("Example: Processing a real PDF")
        print("=" * 80)
        print("To test with a real PDF, place your PDFs in: Datasets/pdfs/")
        print("Then run this script again.")
        sys.exit(0)
    
    # Find first PDF for testing
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        sys.exit(1)
    
    test_pdf = os.path.join(pdf_dir, pdf_files[0])
    print(f"\nTesting with: {pdf_files[0]}")
    print("=" * 80)
    
    # Initialize processor
    processor = PDFProcessor(enable_ocr=True, clean_text=True)
    
    # Process PDF
    print("\n[-] Extracting content...")
    pages_content, metadata = processor.process_pdf(test_pdf)
    
    # Display metadata
    print("\n[+] METADATA")
    print("-" * 80)
    print(f"Title:           {metadata.title}")
    print(f"Author:          {metadata.author}")
    print(f"Created:         {metadata.creation_date}")
    print(f"Pages:           {metadata.num_pages}")
    print(f"Detected Language: {metadata.language}")
    print(f"Has Text:        {metadata.has_text}")
    print(f"Required OCR:    {metadata.required_ocr}")
    
    # Display page statistics
    print("\n[+] PAGE STATISTICS")
    print("-" * 80)
    random_page = random.choice(pages_content)
    print(f"Sample Page: {random_page.page_number}")
    print(f"  Raw length:      {random_page.raw_length} chars")
    print(f"  Cleaned length:  {random_page.cleaned_length} chars")
    print(f"  Sentences:       {random_page.num_sentences}")
    print(f"  Tokens (est.):   {random_page.tokens_estimate:.0f}")
    print(f"  Language:        {random_page.language}")
    print(f"  Extraction:      {random_page.extraction_method}")
    
    # Display sample text
    print("\n[+] SAMPLE TEXT")
    print("-" * 80)
    print(f"Text preview (first 300 chars):")
    print(random_page.text[:300] + "..." if len(random_page.text) > 300 else random_page.text)
    
    # Display sample sentences
    print("\n[+] SAMPLE SENTENCES")
    print("-" * 80)
    if random_page.sentences:
        print(f"First 3 sentences from page {random_page.page_number}:")
        for i, sent in enumerate(random_page.sentences[:3], 1):
            print(f"{i}. {sent}")
    
    # Overall statistics
    print("\n[+] DOCUMENT STATISTICS")
    print("-" * 80)
    total_chars = sum([p.cleaned_length for p in pages_content])
    total_sentences = sum([p.num_sentences for p in pages_content])
    total_tokens = sum([p.tokens_estimate for p in pages_content])
    avg_sentences_per_page = total_sentences / len(pages_content) if pages_content else 0
    avg_tokens_per_page = total_tokens / len(pages_content) if pages_content else 0
    
    print(f"Total Pages:          {len(pages_content)}")
    print(f"Total Characters:     {total_chars:,}")
    print(f"Total Sentences:      {total_sentences:,}")
    print(f"Total Tokens (est.):  {total_tokens:,.0f}")
    print(f"Avg Sentences/Page:   {avg_sentences_per_page:.1f}")
    print(f"Avg Tokens/Page:      {avg_tokens_per_page:.0f}")
    
    # Create DataFrame for detailed analysis
    print("\n[+] DETAILED PAGE BREAKDOWN")
    print("-" * 80)
    df = pd.DataFrame([{
        'page': p.page_number,
        'chars': p.cleaned_length,
        'sentences': p.num_sentences,
        'tokens': p.tokens_estimate,
        'language': p.language or 'unknown',
        'method': p.extraction_method
    } for p in pages_content])
    
    print("\nDescriptive Statistics:")
    print(df[['chars', 'sentences', 'tokens']].describe().round(2))
    
    print("\nLanguage Distribution:")
    print(df['language'].value_counts())
    
    print("\nExtraction Method Distribution:")
    print(df['method'].value_counts())
    
    print("\n" + "=" * 80)
    print("[+] Test completed successfully!")
    print("=" * 80)