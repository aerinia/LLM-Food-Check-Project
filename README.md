# 🌱 Allergen & Vegan Detection System

A fully local, free-to-run system that analyzes food product ingredient lists to detect allergens and determine vegan status. The pipeline combines a **rule-based knowledge base** with a **local LLM (Llama 3 via Ollama)** using Chain-of-Thought reasoning.

> ⚠️ OCR support (image → text) is left as future work. Users currently provide ingredient text directly via CLI or the Streamlit UI.

---

## Pipeline

```
User Input (text)
      ↓
Rule-Based Allergen Detection   ←── Knowledge Base (EN + TR + E-codes)
      ↓
LLM Analysis (Llama 3, CoT)     ←── Chain-of-Thought Prompt
      ↓
Structured JSON Output
```

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Llama 3 via [Ollama](https://ollama.ai/) (local, free) |
| Allergen Detection | Rule-based keyword matching (Python) |
| UI | Streamlit |
| Evaluation | scikit-learn (Accuracy, Recall, F1) |
| OCR (future) | pytesseract / Tesseract |

---

## Project Structure

```
llm-final-project/
├── app.py                  # Main app (CLI + Streamlit UI)
├── evaluation.py           # Evaluation script (33 examples, sklearn metrics)
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── allergen_kb.py      # Knowledge base (EN, Turkish, E-codes)
│   ├── detector.py         # Rule-based allergen detector
│   ├── llm_module.py       # Ollama / Llama 3 integration (CoT prompt)
│   └── ocr_module.py       # Stub — OCR left as future work
├── data/
│   └── sample_inputs/      # Place food label images here (future OCR use)
├── outputs/                # Runtime JSON outputs saved here
└── sample_outputs/
    └── turkish_label_example.json   # Real test result
```

---

## Installation

### 1. Install Ollama and pull Llama 3
Download [Ollama](https://ollama.ai/) for your OS, then run:
```bash
ollama pull llama3
```
This downloads the ~4.7 GB Llama 3 model (one-time).

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Streamlit UI (Recommended)
```bash
python -m streamlit run app.py
```
Opens a browser tab at `http://localhost:8501`. Paste ingredient text into the text area and click **Analyze Ingredients**.

### CLI Mode
```bash
python app.py --text "water, wheat flour, whole eggs, butter, salt"
```
Result is printed to terminal and saved to `outputs/latest_result.json`.

---

## Sample Input / Output

**Input:**
```
buğday unu, tam buğday unu, bitkisel yağ, şeker, pastörize yumurta,
peynir altı suyu tozu, yağsız süt tozu, tuz, aroma verici
```

**Output:**
```json
{
    "ingredients": "buğday unu, tam buğday unu, ...",
    "vegan": "no",
    "allergens": ["milk", "gluten", "eggs"],
    "explanation": "Not vegan due to pastörize yumurta, peynir altı suyu tozu, and yağsız süt tozu.",
    "reasoning_steps": [
        "Step 1: buğday unu — vegan, tam buğday unu — vegan, ...",
        "Step 2: pastörize yumurta — non-vegan (egg), ...",
        "Step 3: No problematic E-codes detected.",
        "Step 4: Product contains eggs and dairy. Verdict: NOT vegan."
    ]
}
```

---

## Knowledge Base Coverage

The allergen knowledge base (`src/allergen_kb.py`) covers 9 categories:

| Category | Examples |
|---|---|
| milk | milk, casein, whey, süt, peynir, E966 |
| gluten | wheat, barley, oats, buğday, un |
| nuts | almond, hazelnut, badem, fındık |
| soy | soy, tofu, soya, E322 |
| eggs | egg, albumen, yumurta |
| fish | fish, anchovy, balık, hamsi |
| shellfish | shrimp, crab, karides |
| gelatin | gelatin, jelatin, E441, E542 |
| carmine | carmine, cochineal, karmin, E120 |

---

## Evaluation Results

Run the evaluation script on 33 labeled examples (English, Turkish, E-code edge cases):
```bash
python evaluation.py
```

| Metric | Result |
|---|---|
| Vegan Classification Accuracy | **87.9%** |
| Vegan F1-Score | **0.778** |
| Allergen Detection Recall | **89.2%** |
| Allergen F1-Score | **0.892** |

The main failure cases involve ambiguous E-codes (e.g. E322 — lecithin can be plant or animal-derived) and some plant-based milks misclassified by the LLM.

---

## LLM Engineering Highlights

- **Chain-of-Thought Prompting:** The LLM is instructed to reason step-by-step (list ingredients → flag suspicious ones → check E-codes → final verdict) before giving a JSON answer.
- **Structured Output:** `format='json'` enforces valid JSON responses from Ollama, with a regex fallback for robustness.
- **Hybrid Architecture:** Rule-based detection handles speed and E-code coverage; LLM handles reasoning and ambiguous cases.
- **Local & Free:** No paid APIs. Runs entirely on your machine using Ollama.
