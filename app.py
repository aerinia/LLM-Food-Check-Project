import argparse
import os
import json
import sys

from src.ocr_module import extract_text_from_image
from src.detector import detect_allergens
from src.llm_module import analyze_ingredients_with_llm

def process_image(image_path: str):
    """
    Full pipeline: Image -> OCR -> Detection -> LLM -> Output
    """
    print(f"Processing image: {image_path}...")
    
    # 1. OCR Extraction
    print("1. Extracting text via OCR...")
    ingredients = extract_text_from_image(image_path)
    if not ingredients:
        return {"error": "Could not extract text from image."}
        
    # 2. Allergen Detection
    print("2. Detecting allergens...")
    allergens = detect_allergens(ingredients)
    
    # 3. LLM Analysis
    print("3. Analyzing vegan status with LLM...")
    llm_result = analyze_ingredients_with_llm(ingredients, allergens)
    
    # 4. Compile Results
    result = {
        "ingredients": ingredients,
        "vegan": llm_result.get("vegan"),
        "allergens": allergens,
        "explanation": llm_result.get("explanation")
    }
    
    return result

def run_streamlit():
    """
    Optional Streamlit UI mode.
    """
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is not installed. Run 'pip install streamlit' to use the UI.")
        return

    from PIL import Image
    import tempfile
    
    st.set_page_config(page_title="Allergen & Vegan Detector", layout="centered")
    st.title("🌱 Allergen & Vegan Detection System")
    st.write("Upload a food label image to detect allergens and verify if it's vegan using OCR and local LLM (Llama 3).")
    
    uploaded_file = st.file_uploader("Upload Food Label Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Analyze Label"):
            with st.spinner("Running pipeline (OCR -> Rule-base -> LLM)..."):
                # Save uploaded file to a temporary file for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    image.save(temp_file.name)
                    temp_path = temp_file.name
                
                # Execute pipeline
                result = process_image(temp_path)
                
                # Display results
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.subheader("Results")
                    st.json(result)
                
                # Cleanup temp file
                os.remove(temp_path)

def main():
    # If the script is being run by Streamlit (e.g., `streamlit run app.py`), sys.argv[0] contains 'streamlit'
    if "streamlit" in sys.argv[0]:
        run_streamlit()
        return

    parser = argparse.ArgumentParser(description="Allergen & Vegan Detection System")
    parser.add_argument("--image", type=str, help="Path to the food product image")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit UI")
    args = parser.parse_args()
    
    if args.ui:
        # Launch Streamlit process
        os.system(f"streamlit run {os.path.abspath(__file__)}")
    elif args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found at {args.image}")
            return
            
        result = process_image(args.image)
        
        print("\n" + "="*40)
        print("FINAL RESULT")
        print("="*40)
        print(json.dumps(result, indent=4))
        print("="*40)
        
        # Save output
        os.makedirs("outputs", exist_ok=True)
        base_name = os.path.basename(args.image).split('.')[0]
        out_path = f"outputs/{base_name}_result.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Result saved to {out_path}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
