"""
Evaluation Script
Compares the Rule-Based Allergen Detector + LLM Vegan Classifier against ground truth labels.
Metrics: Accuracy, Recall, F1-Score (via scikit-learn)
Dataset: 30+ examples covering English, Turkish, E-codes, and hidden allergen edge cases.
"""

import json
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

from src.detector import detect_allergens
from src.llm_module import analyze_ingredients_with_llm

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION DATASET — 30+ examples
# Fields:
#   ingredients  : raw ingredient string (as it would appear on a label)
#   true_allergens: list of allergen category names that should be detected
#   true_vegan   : "yes" or "no"
#   note         : short description of the edge case being tested
# ─────────────────────────────────────────────────────────────────────────────
DATASET = [
    # ── CLEARLY NON-VEGAN (Standard English) ─────────────────────────────────
    {
        "ingredients": "water, wheat flour, sugar, whole eggs, butter, salt",
        "true_allergens": ["gluten", "eggs", "milk"],
        "true_vegan": "no",
        "note": "Classic baked good — eggs + butter"
    },
    {
        "ingredients": "whole milk, cocoa butter, sugar, vanilla extract",
        "true_allergens": ["milk"],
        "true_vegan": "no",
        "note": "Milk chocolate base"
    },
    {
        "ingredients": "chicken broth, whey protein, maltodextrin, salt",
        "true_allergens": ["milk"],
        "true_vegan": "no",
        "note": "Whey is a hidden dairy allergen"
    },
    {
        "ingredients": "gelatin, sugar, citric acid, artificial flavor",
        "true_allergens": ["gelatin"],
        "true_vegan": "no",
        "note": "Gelatin is animal-derived"
    },
    {
        "ingredients": "mayonnaise (eggs, oil, vinegar), mustard, salt",
        "true_allergens": ["eggs"],
        "true_vegan": "no",
        "note": "Mayonnaise contains eggs"
    },
    {
        "ingredients": "anchovies, olive oil, garlic, salt",
        "true_allergens": ["fish"],
        "true_vegan": "no",
        "note": "Fish — anchovy"
    },
    {
        "ingredients": "shrimp, water, salt, sodium tripolyphosphate",
        "true_allergens": ["shellfish"],
        "true_vegan": "no",
        "note": "Shellfish — shrimp"
    },
    {
        "ingredients": "casein, calcium carbonate, dextrose, magnesium stearate",
        "true_allergens": ["milk"],
        "true_vegan": "no",
        "note": "Casein is hidden milk protein"
    },
    {
        "ingredients": "lard, flour, salt, baking powder",
        "true_allergens": ["gluten", "animal_additives"],
        "true_vegan": "no",
        "note": "Lard is pig fat — hidden non-vegan"
    },
    {
        "ingredients": "honey, oats, almonds, sunflower seeds",
        "true_allergens": ["gluten", "nuts"],
        "true_vegan": "no",
        "note": "Honey is not vegan"
    },

    # ── E-CODE EDGE CASES ─────────────────────────────────────────────────────
    {
        "ingredients": "sugar, water, E120, citric acid",
        "true_allergens": ["carmine", "animal_additives"],
        "true_vegan": "no",
        "note": "E120 = Carmine (insect-derived red dye)"
    },
    {
        "ingredients": "water, sugar, E441, fruit flavor",
        "true_allergens": ["gelatin"],
        "true_vegan": "no",
        "note": "E441 = Gelatin"
    },
    {
        "ingredients": "vegetable oil, E471, salt, water",
        "true_allergens": ["fish"],
        "true_vegan": "no",
        "note": "E471 = Mono/diglycerides — possibly animal fat"
    },
    {
        "ingredients": "chocolate, E322, soy lecithin, sugar",
        "true_allergens": ["soy"],
        "true_vegan": "yes",
        "note": "E322 = Lecithin — soy-derived here, product may be vegan"
    },
    {
        "ingredients": "sugar, E542, dicalcium phosphate",
        "true_allergens": ["animal_additives"],
        "true_vegan": "no",
        "note": "E542 = Bone phosphate (animal origin)"
    },
    {
        "ingredients": "apples, sugar, E901, citric acid",
        "true_allergens": ["animal_additives"],
        "true_vegan": "no",
        "note": "E901 = Beeswax (used as coating)"
    },
    {
        "ingredients": "flour, water, E920, yeast, salt",
        "true_allergens": ["gluten", "animal_additives"],
        "true_vegan": "no",
        "note": "E920 = L-cysteine (often from feathers)"
    },

    # ── TURKISH LABEL EDGE CASES ──────────────────────────────────────────────
    {
        "ingredients": "su, buğday unu, şeker, yumurta, tereyağı, tuz",
        "true_allergens": ["gluten", "eggs", "milk"],
        "true_vegan": "no",
        "note": "Turkish label — classic kurabiye"
    },
    {
        "ingredients": "süt, kakao, şeker, jelatin",
        "true_allergens": ["milk", "gelatin"],
        "true_vegan": "no",
        "note": "Turkish milk chocolate with gelatin"
    },
    {
        "ingredients": "soya yağı, su, tuz, soya fasulyesi",
        "true_allergens": ["soy"],
        "true_vegan": "yes",
        "note": "Turkish soy-based product — vegan"
    },
    {
        "ingredients": "yoğurt, su, tuz, nane",
        "true_allergens": ["milk"],
        "true_vegan": "no",
        "note": "Turkish ayran — yogurt is dairy"
    },
    {
        "ingredients": "fındık, kakao yağı, şeker, vanilin",
        "true_allergens": ["nuts"],
        "true_vegan": "yes",
        "note": "Turkish hazelnut spread — vegan if no milk"
    },
    {
        "ingredients": "badem, bal, tarçın",
        "true_allergens": ["nuts"],
        "true_vegan": "no",
        "note": "Honey (bal) makes this non-vegan"
    },
    {
        "ingredients": "ton balığı, ayçiçek yağı, tuz",
        "true_allergens": ["fish"],
        "true_vegan": "no",
        "note": "Turkish tuna can — balık"
    },

    # ── CLEARLY VEGAN ─────────────────────────────────────────────────────────
    {
        "ingredients": "water, tomatoes, onions, garlic, olive oil, herbs, salt",
        "true_allergens": [],
        "true_vegan": "yes",
        "note": "Simple tomato sauce — all plant-based"
    },
    {
        "ingredients": "rolled oats, maple syrup, cocoa powder, salt",
        "true_allergens": ["gluten"],
        "true_vegan": "yes",
        "note": "Vegan energy bar — oats are gluten risk"
    },
    {
        "ingredients": "soybean oil, water, vinegar, salt, sugar",
        "true_allergens": ["soy"],
        "true_vegan": "yes",
        "note": "Vegan dressing"
    },
    {
        "ingredients": "chickpeas, tahini, lemon juice, garlic, olive oil, salt",
        "true_allergens": [],
        "true_vegan": "yes",
        "note": "Hummus — classic vegan"
    },
    {
        "ingredients": "oat milk, water, sunflower oil, salt, vitamin D",
        "true_allergens": ["gluten"],
        "true_vegan": "yes",
        "note": "Oat milk — vegan dairy alternative"
    },
    {
        "ingredients": "coconut milk, sugar, vanilla, salt",
        "true_allergens": [],
        "true_vegan": "yes",
        "note": "Coconut-based ice cream — vegan"
    },

    # ── AMBIGUOUS / TRICKY CASES ──────────────────────────────────────────────
    {
        "ingredients": "natural flavor, sugar, palm oil, salt",
        "true_allergens": [],
        "true_vegan": "yes",
        "note": "'Natural flavor' is ambiguous — rule-base treats as vegan"
    },
    {
        "ingredients": "vitamin D3, calcium carbonate, water",
        "true_allergens": [],
        "true_vegan": "no",
        "note": "Vitamin D3 is often lanolin (sheep wool)-derived"
    },
    {
        "ingredients": "isinglass, apple juice, citric acid",
        "true_allergens": ["animal_additives"],
        "true_vegan": "no",
        "note": "Isinglass is fish-derived fining agent used in wine/beer"
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def flatten_allergen_labels(true_allergens: list, pred_allergens: list, all_categories: list):
    """
    Converts allergen category lists to binary vectors for each category.
    Used to compute per-category recall.
    """
    y_true = [1 if cat in true_allergens else 0 for cat in all_categories]
    y_pred = [1 if cat in pred_allergens else 0 for cat in all_categories]
    return y_true, y_pred


def print_table(results: list):
    """Prints a neatly formatted evaluation results table."""
    print(f"\n{'='*90}")
    print(f"{'#':<4} {'Note':<42} {'True Vegan':<12} {'Pred Vegan':<12} {'Allergens OK':<12}")
    print(f"{'─'*90}")
    for i, r in enumerate(results, 1):
        match = "✓" if r["true_vegan"] == r["pred_vegan"] else "✗"
        allergen_ok = "✓" if r["allergens_correct"] else "✗"
        print(
            f"{i:<4} {r['note'][:41]:<42} {r['true_vegan']:<12} "
            f"{r['pred_vegan']:<12} {allergen_ok:<12}  {match}"
        )
    print(f"{'='*90}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pipeline():
    """
    Runs the full rule-based + LLM pipeline on DATASET and reports metrics.
    """
    all_categories = list({cat for item in DATASET for cat in item["true_allergens"]})

    y_true_vegan, y_pred_vegan = [], []
    y_true_allergen_flat, y_pred_allergen_flat = [], []
    row_results = []

    print(f"\nEvaluating pipeline on {len(DATASET)} examples...\n")

    for idx, item in enumerate(DATASET, 1):
        print(f"[{idx:02}/{len(DATASET)}] {item['note']}")

        # Step 1: Rule-based allergen detection
        pred_allergens = detect_allergens(item["ingredients"])

        # Step 2: LLM vegan classification
        llm_result = analyze_ingredients_with_llm(item["ingredients"], pred_allergens)
        pred_vegan = llm_result.get("vegan", "unknown").strip().lower()

        # Step 3: Check if predicted allergen set covers all true allergens
        allergens_correct = all(ta in pred_allergens for ta in item["true_allergens"])

        # Collect binary labels for metrics
        y_true_vegan.append(1 if item["true_vegan"] == "yes" else 0)
        y_pred_vegan.append(1 if pred_vegan == "yes" else 0)

        # Flatten allergen labels across all categories
        t_flat, p_flat = flatten_allergen_labels(
            item["true_allergens"], pred_allergens, all_categories
        )
        y_true_allergen_flat.extend(t_flat)
        y_pred_allergen_flat.extend(p_flat)

        row_results.append({
            "note": item["note"],
            "true_vegan": item["true_vegan"],
            "pred_vegan": pred_vegan,
            "true_allergens": item["true_allergens"],
            "pred_allergens": pred_allergens,
            "allergens_correct": allergens_correct,
        })

    # ── Print Table ────────────────────────────────────────────────────────────
    print_table(row_results)

    # ── Compute Metrics ────────────────────────────────────────────────────────
    vegan_accuracy = accuracy_score(y_true_vegan, y_pred_vegan)
    allergen_recall = recall_score(y_true_allergen_flat, y_pred_allergen_flat, zero_division=0)
    allergen_f1 = f1_score(y_true_allergen_flat, y_pred_allergen_flat, zero_division=0)
    vegan_f1 = f1_score(y_true_vegan, y_pred_vegan, zero_division=0)

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total Examples          : {len(DATASET)}")
    print(f"  Vegan Classification")
    print(f"    → Accuracy            : {vegan_accuracy * 100:.1f}%")
    print(f"    → F1-Score            : {vegan_f1:.3f}")
    print(f"  Allergen Detection")
    print(f"    → Recall              : {allergen_recall * 100:.1f}%")
    print(f"    → F1-Score            : {allergen_f1:.3f}")
    print("=" * 60)

    print("\nDetailed Vegan Classification Report:")
    print(classification_report(
        y_true_vegan, y_pred_vegan,
        target_names=["Non-Vegan", "Vegan"],
        zero_division=0
    ))

    # Save results
    import os
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(row_results, f, indent=4, ensure_ascii=False)
    print("Full results saved to: outputs/evaluation_results.json")


if __name__ == "__main__":
    evaluate_pipeline()
