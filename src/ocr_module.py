"""
OCR Module — Future Work

This module is intentionally left as a stub.
OCR-based ingredient extraction (using pytesseract / Tesseract OCR) is left as future work.

In the current pipeline, users provide the ingredient text directly via:
  - CLI:  python app.py --text "ingredient list here"
  - UI:   Streamlit text area input

When implementing OCR in the future, this module should:
  1. Accept an image path as input.
  2. Preprocess the image (grayscale, denoise, threshold) for better OCR accuracy.
  3. Run pytesseract.image_to_string() on the image.
  4. Clean and normalize the extracted text.
  5. Return the cleaned text string for downstream processing.

Example (not active):
    import pytesseract
    from PIL import Image

    def extract_text_from_image(image_path: str) -> str:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img).lower()
        return text.strip()
"""


def extract_text_from_image(image_path: str) -> str:
    """
    Stub function — OCR is not active in the current pipeline.
    Raises NotImplementedError to make the intent explicit.
    """
    raise NotImplementedError(
        "OCR integration is left as future work. "
        "Please provide ingredient text directly using --text or the Streamlit UI."
    )
