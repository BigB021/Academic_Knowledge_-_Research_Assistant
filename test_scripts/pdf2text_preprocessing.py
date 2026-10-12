# Author: Youssef Aitbouddroub

import os
import pymupdf  
from tqdm.auto import tqdm
import random
from spacy.lang.en import English
import pandas as pd

pdf_dir = os.path.join(os.path.dirname(__file__), "..", "documents", "pdfs")

def text_formatter(text: str):
    cleaned_text = text.replace("\n", " ").replace("\t", " ")
    return cleaned_text

def open_read_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    pages_text = []

    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text("text")
        # text = text_formatter(text)
        pages_text.append({
            "page number": page_number + 1, 
            "text": text,
            "length": len(text),
            "tokens_per_page": len(text.split(" ")) * 1.33
            })

    doc.close()
    return pages_text



def extract_sentences_from_text(pages_text: list):
    nlp = English()
    nlp.add_pipe("sentencizer")

    for page in pages_text:
        doc = nlp(page["text"])
        page["sentences"] = [sent.text for sent in doc.sents]
        page["num_sentences"] = len(page["sentences"])

    return pages_text




# test the function
if __name__ == "__main__":
    pdf_path = os.path.join(pdf_dir, "Face_Anti_Spoofing_with_Swin_Tiny__Transfer_Learning__Class_Imbalance_Mitigation__and_Adversarial_Robustness_on_LCC_FASD__1_.pdf")
    pages_text = open_read_pdf(pdf_path)
    pages_text = extract_sentences_from_text(pages_text)
    random_page = random.choice(pages_text)
    print(f"Random page number: {random_page['page number']}")
    print(f"Random page text: {random_page['text']}")
    print("*" * 50)
    print(f"Random page number of sentences: {random_page['num_sentences']}")
    print(f"Random page sentences: {random_page['sentences']}")
    print("*" * 50)
    print("stats:")
    print(f"Total pages: {len(pages_text)}")
    print(f"Total characters: {sum([page['length'] for page in pages_text])}")
    print(f"Total sentences: {sum([page['num_sentences'] for page in pages_text])}")
    print(f"Total tokens (per page): {sum([page['tokens_per_page'] for page in pages_text])}")
    print("*" * 50)
    df = pd.DataFrame(pages_text)
    print(df.describe().round(2))