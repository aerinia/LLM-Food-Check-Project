import pytesseract
from PIL import Image
import re

# IMPORTANT: If Tesseract is not in your system's PATH (especially on Windows), 
# uncomment the following line and provide the absolute path to tesseract.exe:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path: str) -> str:
    """
    Extracts ingredient text from a food product image using Tesseract OCR.
    """
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(img)
        
        # Basic cleaning: convert to lowercase and remove excessive whitespaces/newlines
        clean_text = text.lower()
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return ""
