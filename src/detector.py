import re
from .allergen_kb import ALLERGEN_DB

def detect_allergens(ingredient_text: str) -> list:
    """
    Detects allergens in the ingredient text using keyword matching based on ALLERGEN_DB.
    """
    detected_allergens = []
    
    # Use lowercase for case-insensitive matching
    text = ingredient_text.lower()
    
    for category, keywords in ALLERGEN_DB.items():
        for keyword in keywords:
            # Word boundaries (\b) ensure we match full words (e.g. 'wheat' but not 'buckwheat')
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text):
                if category not in detected_allergens:
                    detected_allergens.append(category)
                break # Move to next category once we find a match
                
    return detected_allergens
