from src.ocr import extract_text_from_pdf, is_probably_scanned


path = "data/samples/sample_claim.pdf"

text = extract_text_from_pdf(path)
print("=" * 60)
print("EXTRACTED TEXT")
print("=" * 60)
print(text[:1500])  # first 1500 characters, enough to eyeball
print("=" * 60)
print(f"Total characters: {len(text)}")
print(f"Probably scanned? {is_probably_scanned(text)}")