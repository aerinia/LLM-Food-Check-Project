import json
from sklearn.metrics import accuracy_score, recall_score
from src.detector import detect_allergens
from src.llm_module import analyze_ingredients_with_llm

# Sample Dataset for Evaluation
DATASET = [
    {
        "ingredients": "water, wheat flour, sugar, eggs, butter, salt",
        "true_allergens": ["gluten", "eggs", "milk"],
        "true_vegan": "no"
    },
    {
        "ingredients": "soybean oil, water, vinegar, salt, sugar",
        "true_allergens": ["soy"],
        "true_vegan": "yes"
    },
    {
        "ingredients": "almonds, peanuts, honey",
        "true_allergens": ["nuts"], 
        "true_vegan": "no" 
    },
    {
        "ingredients": "rolled oats, maple syrup, cocoa powder, salt",
        "true_allergens": ["gluten"], # Assuming oats fall under gluten cross-contamination risk or general grain
        "true_vegan": "yes"
    }
]

def evaluate_pipeline():
    """
    Evaluates the rule-based allergen detector and the LLM vegan classifier.
    Prints out Accuracy for Vegan Classification and Recall for Allergen Detection.
    """
    y_true_vegan = []
    y_pred_vegan = []
    
    y_true_allergens_flat = []
    y_pred_allergens_flat = []
    
    print("Running Evaluation on Sample Dataset...\n")
    
    for idx, item in enumerate(DATASET, 1):
        print(f"Test {idx}: {item['ingredients']}")
        
        # 1. Rule-based Allergen Detection
        pred_allergens = detect_allergens(item['ingredients'])
        
        # 2. LLM Analysis
        llm_result = analyze_ingredients_with_llm(item['ingredients'], pred_allergens)
        pred_vegan = llm_result.get('vegan', 'unknown').lower()
        
        print(f"  True Vegan: {item['true_vegan']} | Pred Vegan: {pred_vegan}")
        print(f"  True Allergens: {item['true_allergens']} | Pred Allergens: {pred_allergens}\n")
        
        # Collect for metrics
        y_true_vegan.append(1 if item['true_vegan'] == 'yes' else 0)
        y_pred_vegan.append(1 if pred_vegan == 'yes' else 0)
        
        # Flatten allergens to calculate global recall
        for ta in item['true_allergens']:
            y_true_allergens_flat.append(1)
            y_pred_allergens_flat.append(1 if ta in pred_allergens else 0)
            
    # Calculate Sklearn Metrics
    accuracy = accuracy_score(y_true_vegan, y_pred_vegan)
    recall = recall_score(y_true_allergens_flat, y_pred_allergens_flat)
    
    print("========================================")
    print("EVALUATION RESULTS")
    print("========================================")
    print(f"Vegan Classification Accuracy: {accuracy * 100:.2f}%")
    print(f"Allergen Detection Recall:     {recall * 100:.2f}%")
    print("========================================")

if __name__ == "__main__":
    evaluate_pipeline()
