"""
LLM Module — Uses Ollama with Llama 3 (local, free) for vegan classification.
Employs a Chain-of-Thought (CoT) prompt with RAG-retrieved allergen context
for more accurate and explainable reasoning.

Pipeline:
    Ingredient Text → RAG Retrieval → Augmented Prompt → Llama 3 → JSON Output
"""

import ollama
import json
import re

from .rag_module import retrieve_allergen_context


def analyze_ingredients_with_llm(ingredient_text: str, detected_allergens: list) -> dict:
    """
    Uses local Ollama (llama3) to classify vegan status with:
    - Chain-of-Thought (CoT) reasoning (4 structured steps)
    - RAG-retrieved allergen context injected into the prompt

    Args:
        ingredient_text: Raw ingredient string from the label.
        detected_allergens: List of allergen categories detected by the rule-based system.

    Returns:
        dict with keys: vegan, explanation, reasoning_steps
    """
    allergens_str = ", ".join(detected_allergens) if detected_allergens else "None"

    # ── RAG: Retrieve relevant allergen knowledge ─────────────────────────────
    rag_context = retrieve_allergen_context(ingredient_text, top_k=3)
    rag_section = ""
    if rag_context:
        rag_section = f"""
RELEVANT ALLERGEN KNOWLEDGE (retrieved from knowledge base):
{rag_context}
"""

    # ── Chain-of-Thought Prompt ───────────────────────────────────────────────
    prompt = f"""You are an expert food scientist and vegan certification specialist.

Your task is to analyze the following food ingredient list and determine whether the product is VEGAN.
{rag_section}
---
INGREDIENT LIST:
{ingredient_text}

ALLERGENS ALREADY DETECTED BY RULE-BASED SYSTEM:
{allergens_str}
---

Follow these steps EXACTLY and respond ONLY with a valid JSON object:

Step 1 - List each ingredient individually.
Step 2 - For each ingredient, identify if it is: (a) clearly vegan, (b) clearly non-vegan (animal-derived), or (c) ambiguous/possibly animal-derived.
Step 3 - Pay special attention to: E-codes (e.g. E120=carmine, E441=gelatin, E471=may be animal fat, E322=lecithin), hidden dairy (casein, whey, lactalbumin), hidden gelatin, beeswax, shellac, carmine.
Step 4 - Based on your analysis, give the final vegan verdict.

Respond ONLY with this exact JSON structure, nothing else:
{{
    "reasoning_steps": [
        "Step 1: [list each ingredient and classify it]",
        "Step 2: [flag any suspicious or non-vegan ingredients]",
        "Step 3: [check E-codes and hidden non-vegan additives]",
        "Step 4: [state your final reasoning]"
    ],
    "vegan": "yes" or "no",
    "explanation": "One clear sentence explaining the final verdict."
}}
"""

    try:
        # format='json' instructs Ollama to enforce JSON output
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
            format="json"
        )

        raw_content = response["message"]["content"].strip()

        # Attempt to parse JSON; fall back to regex extraction if needed
        try:
            result_json = json.loads(raw_content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if match:
                result_json = json.loads(match.group())
            else:
                raise ValueError("No valid JSON found in LLM response.")

        return {
            "vegan": str(result_json.get("vegan", "unknown")).lower().strip(),
            "explanation": result_json.get("explanation", "No explanation provided."),
            "reasoning_steps": result_json.get("reasoning_steps", []),
            "rag_used": bool(rag_context)
        }

    except Exception as e:
        print(f"[LLM Error] {e}")
        return {
            "vegan": "error",
            "explanation": f"LLM could not be reached or returned invalid output: {str(e)}",
            "reasoning_steps": [],
            "rag_used": False
        }
