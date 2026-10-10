# Author Youssef Aitbouddroub
# Script to convert pdf files to markdown files
import pymupdf4llm
import os


pdf_dir = os.path.join(os.path.dirname(__file__), "..", "Datasets", "pdfs")
pdf_path = os.path.join(pdf_dir, "2410.01574v4.pdf")

md = pymupdf4llm.to_markdown(pdf_path)

print(md)

# Save the file into result.md
output_path = os.path.join(pdf_dir, "result.md")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(md)
