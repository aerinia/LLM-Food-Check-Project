# Allergen & Vegan Detection System from Food Labels using OCR + LLM

A fully functional, local, and free system to analyze food labels. It extracts ingredient text using Tesseract OCR, detects allergens via a rule-based knowledge base, and uses a local LLM (Llama 3 via Ollama) to classify if the product is vegan and explain its reasoning.

## Tech Stack
- **OCR:** Tesseract (pytesseract)
- **LLM:** Llama 3 (via Ollama)
- **Backend:** Python
- **UI:** Streamlit
- **Evaluation:** scikit-learn

## Setup & Installation

### 1. Install System Dependencies

**Tesseract OCR:**
- **Windows:** Download and install from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Ensure the Tesseract installation path is added to your system's Environment Variables `PATH` (e.g., `C:\Program Files\Tesseract-OCR`).
- **macOS:** `brew install tesseract`
- **Linux:** `sudo apt-get install tesseract-ocr`

**Ollama:**
- Install [Ollama](https://ollama.ai/)
- Download the Llama 3 model by running:
  ```bash
  ollama run llama3
  ```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## Project Structure
- `app.py`: Main application (CLI & UI)
- `src/ocr_module.py`: Handles image-to-text extraction
- `src/detector.py`: Keyword-based allergen detection
- `src/allergen_kb.py`: Pre-defined knowledge base of allergens
- `src/llm_module.py`: Prompts local Llama 3 for vegan classification
- `evaluation.py`: Computes accuracy and recall of the pipeline using scikit-learn.

## Usage

### Streamlit UI (Recommended)
Run the application with a web interface:
```bash
python app.py --ui
```
*Alternatively, you can run `streamlit run app.py`.*

### CLI Mode
Process an image directly from the command line:
```bash
python app.py --image path/to/your/image.jpg
```
Output will be saved in the `outputs/` directory.

## Sample Output
```json
{
    "ingredients": "water, wheat flour, sugar, eggs, butter, salt",
    "vegan": "no",
    "allergens": [
        "gluten",
        "eggs",
        "milk"
    ],
    "explanation": "The product contains eggs and butter (milk), which are animal-derived ingredients."
}
```

## Evaluation
To run the simple evaluation script comparing the rule-based + LLM approach on a set of sample texts:
```bash
python evaluation.py
```
