# Author: Youssef Aitbouddroub

import os
import pymupdf  
from tqdm.auto import tqdm
import random

pdf_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", "pdfs")

def open_read_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    pages_text = []

    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text("text")
        pages_text.append({
            "page number": page_number + 1, 
            "text": text,
            "length": len(text),
            "tokens": len(text.split()) * 1.33
            })

    doc.close()
    return pages_text

# test the function
if __name__ == "__main__":
    pdf_path = os.path.join(pdf_dir, "Face_Anti_Spoofing_with_Swin_Tiny__Transfer_Learning__Class_Imbalance_Mitigation__and_Adversarial_Robustness_on_LCC_FASD__1_.pdf")
    pages_text = open_read_pdf(pdf_path)
    random_page = random.choice(pages_text)
    print(f"Random page number: {random_page['page number']}")
    print(f"Random page text: {random_page['text']}")
    print("*" * 50)
    print("stats:")
    print(f"Total pages: {len(pages_text)}")
    print(f"Total characters: {sum([page['length'] for page in pages_text])}")
    print(f"Total tokens: {sum([page['tokens'] for page in pages_text])}")