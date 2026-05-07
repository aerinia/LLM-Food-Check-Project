"""
Allergen & Vegan Detection System
Pipeline: Text Input → Allergen Detection → LLM Analysis → Structured Output
Note: OCR integration is left as future work (see src/ocr_module.py).
"""

import argparse
import os
import json
import sys

from src.detector import detect_allergens
from src.llm_module import analyze_ingredients_with_llm


def process_text(ingredient_text: str) -> dict:
    """
    Full pipeline: Ingredient Text → Allergen Detection → LLM → Output
    """
    print("Processing ingredients...")

    # 1. Allergen Detection (Rule-based)
    print("1. Detecting allergens...")
    allergens = detect_allergens(ingredient_text)

    # 2. LLM Analysis
    print("2. Analyzing vegan status with LLM (Llama 3)...")
    llm_result = analyze_ingredients_with_llm(ingredient_text, allergens)

    # 3. Compile Results
    result = {
        "ingredients": ingredient_text,
        "vegan": llm_result.get("vegan"),
        "allergens": allergens,
        "explanation": llm_result.get("explanation"),
        "reasoning_steps": llm_result.get("reasoning_steps", [])
    }

    return result


def run_streamlit():
    """
    Streamlit UI mode — user pastes or types ingredients directly.
    """
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is not installed. Run: pip install streamlit")
        return

    st.set_page_config(page_title="Allergen & Vegan Detector", page_icon="🌱", layout="centered")

    # Header
    st.title("🌱 Allergen & Vegan Detection System")
    st.markdown(
        "Paste the **ingredient list** from a food product label below. "
        "The system will detect allergens and determine if the product is vegan "
        "using a local LLM (Llama 3 via Ollama)."
    )
    st.markdown("---")

    # Text Input
    ingredient_text = st.text_area(
        label="📋 Ingredient List",
        placeholder="e.g. water, wheat flour, sugar, whole eggs, butter, salt, vanilla extract...",
        height=180,
    )

    # Language note
    st.caption("💡 Turkish ingredient names (e.g. süt, yumurta, buğday) and E-codes (e.g. E120, E441) are also supported.")

    if st.button("🔍 Analyze Ingredients", use_container_width=True):
        if not ingredient_text.strip():
            st.warning("Please enter at least one ingredient.")
        else:
            with st.spinner("Running pipeline (Rule-base → LLM)... This may take a moment."):
                result = process_text(ingredient_text)

            st.markdown("---")
            st.subheader("Results")

            # Vegan status badge
            vegan_val = result.get("vegan", "unknown")
            if vegan_val == "yes":
                st.success("✅ **Vegan: Yes**")
            elif vegan_val == "no":
                st.error("❌ **Vegan: No**")
            else:
                st.warning(f"⚠️ **Vegan: {vegan_val}**")

            # Allergens
            allergens = result.get("allergens", [])
            if allergens:
                st.error(f"⚠️ **Allergens Detected:** {', '.join(allergens)}")
            else:
                st.success("✅ **No common allergens detected.**")

            # Reasoning steps from CoT
            steps = result.get("reasoning_steps", [])
            if steps:
                with st.expander("🧠 LLM Reasoning Steps"):
                    for i, step in enumerate(steps, 1):
                        st.markdown(f"**Step {i}:** {step}")

            # Explanation
            st.info(f"💬 **Explanation:** {result.get('explanation')}")

            # Raw JSON
            with st.expander("📄 Full JSON Output"):
                st.json(result)


def main():
    # Detect if launched via `streamlit run app.py`
    # streamlit.runtime.exists() is the officially supported way to check this
    import os as _os
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            run_streamlit()
            return
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Allergen & Vegan Detection System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--text",
        type=str,
        help='Ingredient list as a string.\nExample: --text "water, wheat, egg, butter"'
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Streamlit web UI."
    )
    args = parser.parse_args()

    if args.ui:
        os.system(f"python -m streamlit run {os.path.abspath(__file__)}")

    elif args.text:
        result = process_text(args.text)

        print("\n" + "=" * 50)
        print("FINAL RESULT")
        print("=" * 50)
        print(json.dumps(result, indent=4, ensure_ascii=False))
        print("=" * 50)

        # Save output
        os.makedirs("outputs", exist_ok=True)
        out_path = "outputs/latest_result.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"\nResult saved to: {out_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
