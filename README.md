# 🌱 Allergen & Vegan Detection System

A fully local, free-to-run system that analyzes food product ingredient lists to detect allergens and determine vegan status. The pipeline combines a **rule-based knowledge base**, a **RAG (Retrieval-Augmented Generation)** component powered by FAISS, and a **local LLM (Llama 3 via Ollama)** with Chain-of-Thought reasoning.

> ⚠️ OCR support (image → text) is left as future work. Users provide ingredient text directly via CLI or the Streamlit UI.

---

## Pipeline

```
User Input (ingredient text)
            ↓
 Rule-Based Allergen Detection  ←── allergen_kb.py (EN + TR + E-codes)
            ↓
   RAG Retrieval (FAISS)        ←── Top-3 relevant allergen documents
            ↓                        embedded with sentence-transformers
  Augmented LLM Prompt
            ↓
  Llama 3 via Ollama            ←── Chain-of-Thought (4 steps)
            ↓
  Structured JSON Output
  {vegan, allergens, explanation, reasoning_steps, rag_used}
```

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Llama 3 via [Ollama](https://ollama.ai/) (local, free) |
| RAG / Vector DB | FAISS + sentence-transformers (`all-MiniLM-L6-v2`) |
| Allergen Detection | Rule-based keyword matching (Python regex) |
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
│   ├── rag_module.py       # FAISS RAG: embeds & retrieves allergen context
│   ├── llm_module.py       # Ollama / Llama 3 integration (RAG + CoT prompt)
│   └── ocr_module.py       # Stub — OCR left as future work
├── data/
│   └── sample_inputs/      # Place food label images here (future OCR use)
├── outputs/                # Runtime JSON outputs saved here
└── sample_outputs/
    ├── turkish_label_example.json
    └── turkish_gummy_candy.json
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
> The first run will also download the `all-MiniLM-L6-v2` sentence-transformer model (~90 MB) for RAG embedding.

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

**Input (Turkish gummy candy label):**
```
Glikoz şurubu, şeker, su, sığır jelatini, dekstroz, nem verici (sorbitol),
aroma verici (çilek), meyve ve bitki konsantreleri, alg.
```

**Output:**
```json
{
    "ingredients": "Glikoz şurubu, şeker, su, sığır jelatini, ...",
    "vegan": "no",
    "allergens": ["gelatin"],
    "explanation": "Not vegan: contains sığır jelatini (beef gelatin, animal-derived).",
    "reasoning_steps": [
        "Step 1: Glikoz şurubu — vegan, şeker — vegan, sığır jelatini — non-vegan, ...",
        "Step 2: Flagged: sığır jelatini (beef gelatin — animal bone/skin)",
        "Step 3: No problematic E-codes detected",
        "Step 4: Product contains beef gelatin. Verdict: NOT vegan"
    ],
    "rag_used": true
}
```

---

## Knowledge Base Coverage

The allergen knowledge base (`src/allergen_kb.py`) covers 9 categories:

| Category | Examples |
|---|---|
| milk | milk, casein, whey, süt, peynir, tereyağı |
| gluten | wheat, barley, buğday, un, E1410 |
| nuts | almond, hazelnut, badem, fındık |
| soy | soy, tofu, soya, E322 |
| eggs | egg, albumen, yumurta |
| fish | fish, anchovy, balık, hamsi |
| shellfish | shrimp, crab, karides |
| gelatin | gelatin, jelatin, sığır jelatini, E441, E542 |
| carmine / animal_additives | E120, E901, E904, E920, isinglass, lard |

---

## RAG Pipeline Details

`src/rag_module.py` implements a lightweight RAG system:

1. **Documents:** 9 rich-text allergen category descriptions (EN + TR + E-codes)
2. **Embedding:** `all-MiniLM-L6-v2` via sentence-transformers (~90 MB, downloaded once)
3. **Index:** FAISS `IndexFlatIP` (cosine similarity on normalized vectors)
4. **Retrieval:** Top-3 most relevant documents retrieved for each ingredient query
5. **Augmentation:** Retrieved context injected into the Llama 3 CoT prompt
6. **Singleton pattern:** Model loaded once and reused for all queries

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

**Key failure cases:**
- E322 (lecithin) — ambiguous source (soy or egg); LLM over-flags as non-vegan
- E542 (bone phosphate) — rare E-code; LLM missed the non-vegan origin
- Plant-based milks (oat milk, coconut) — LLM occasionally misclassifies as non-vegan

---

## LLM Engineering Highlights

| Technique | Implementation |
|---|---|
| **RAG** | FAISS retrieval injects allergen context into each LLM call |
| **Prompt Engineering** | 4-step Chain-of-Thought forces explicit per-ingredient reasoning |
| **Structured Output** | `format='json'` (Ollama) + regex fallback ensures valid JSON always |
| **Hybrid Architecture** | Rule-based handles speed & keywords; LLM handles ambiguity & reasoning |
| **Quantitative Evaluation** | 33-example benchmark with Accuracy, Recall, F1-Score via sklearn |
| **Local & Free** | No paid APIs — runs entirely on-device via Ollama |
