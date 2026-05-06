import ollama
import json

def analyze_ingredients_with_llm(ingredient_text: str, detected_allergens: list) -> dict:
    """
    Uses local Ollama with llama3 to classify vegan status and explain reasoning.
    """
    allergens_str = ", ".join(detected_allergens) if detected_allergens else "None detected"
    
    # Prompt is designed to be deterministic and force a specific JSON structure
    prompt = f"""
    You are an expert food scientist. Analyze the following ingredients and detected allergens to determine if the product is vegan.
    
    Ingredients: {ingredient_text}
    Detected Allergens: {allergens_str}
    
    Respond ONLY with a valid JSON object using the exact structure below, nothing else:
    {{
        "vegan": "yes" or "no",
        "explanation": "Short 1-2 sentence explanation of why it is or isn't vegan based on the ingredients."
    }}
    """
    
    try:
        # We specify format='json' to force Ollama to return structured JSON
        response = ollama.chat(model='llama3', messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ], format='json')
        
        result_text = response['message']['content'].strip()
        
        # Parse the JSON response
        result_json = json.loads(result_text)
        
        return {
            "vegan": str(result_json.get("vegan", "unknown")).lower(),
            "explanation": result_json.get("explanation", "No explanation provided.")
        }
        
    except Exception as e:
        print(f"Error querying LLM: {e}")
        return {
            "vegan": "error",
            "explanation": f"LLM error: {str(e)}"
        }
