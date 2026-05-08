"""
RAG Module — Retrieval-Augmented Generation using FAISS + Sentence Transformers

How it works:
1. Each allergen category is represented as a rich text "document"
2. Documents are embedded using a lightweight sentence transformer model
3. When an ingredient list is provided, it is embedded and used to query the FAISS index
4. The top-k most relevant allergen documents are retrieved
5. Retrieved context is injected into the LLM prompt to improve accuracy

This follows the standard RAG (Retrieval-Augmented Generation) pattern:
    Query → Retrieve → Augment → Generate
"""

import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Allergen Knowledge Documents
# Each document describes one allergen category in detail.
# These are the "chunks" that will be indexed and retrieved.
# ─────────────────────────────────────────────────────────────────────────────
ALLERGEN_DOCUMENTS = [
    {
        "category": "milk",
        "text": (
            "Milk and dairy allergens. Keywords: milk, casein, whey, lactose, butter, "
            "cheese, cream, yogurt, lactalbumin, lactoglobulin, ghee. "
            "Turkish: süt, peynir, tereyağı, yoğurt, krema, kazein. "
            "E-codes: E966 (lactitol). "
            "All dairy products are non-vegan as they are derived from animals. "
            "Hidden sources include casein in processed foods and whey in protein supplements."
        )
    },
    {
        "category": "gluten",
        "text": (
            "Gluten and wheat allergens. Keywords: wheat, barley, rye, oats, spelt, "
            "malt, seitan, semolina, flour, starch, breadcrumbs. "
            "Turkish: buğday, arpa, çavdar, yulaf, un, nişasta, ekmek kırıntısı. "
            "E-codes: E1404, E1410, E1412, E1414 (modified starches, often wheat-based). "
            "Gluten is a protein found in cereal grains. Vegan status is not affected, "
            "but it is a major allergen for celiac disease patients."
        )
    },
    {
        "category": "nuts",
        "text": (
            "Tree nuts and peanut allergens. Keywords: almond, hazelnut, walnut, cashew, "
            "pecan, macadamia, pistachio, peanut, pine nut, brazil nut, chestnut. "
            "Turkish: badem, fındık, ceviz, kaju, antep fıstığı, yer fıstığı, kestane. "
            "Nuts themselves are vegan but are a major allergen category. "
            "Hidden sources include nut oils, marzipan, and praline."
        )
    },
    {
        "category": "soy",
        "text": (
            "Soy and soybean allergens. Keywords: soy, soybean, tofu, edamame, tempeh, "
            "miso, tamari, soy sauce, soya, lecithin. "
            "Turkish: soya, soya fasulyesi. "
            "E-codes: E322 (lecithin, commonly soy-derived). "
            "Soy products are typically vegan. Lecithin (E322) is usually from soy "
            "but can sometimes be from eggs — check the source if listed."
        )
    },
    {
        "category": "eggs",
        "text": (
            "Egg allergens. Keywords: egg, albumen, globulin, ovomucoid, mayonnaise, "
            "meringue, lysozyme. "
            "Turkish: yumurta. "
            "E-codes: E1105 (lysozyme, from egg white). "
            "Eggs are non-vegan animal products. Hidden in mayonnaise, pasta, "
            "baked goods, and some sauces."
        )
    },
    {
        "category": "fish",
        "text": (
            "Fish allergens. Keywords: fish, cod, salmon, tuna, tilapia, anchovy, "
            "sardine, mackerel, halibut, bass. "
            "Turkish: balık, morina, somon, ton balığı, hamsi, sardalya. "
            "E-codes: E471 (mono- and diglycerides, can be fish-derived). "
            "Fish are non-vegan. Hidden sources include Worcestershire sauce, "
            "Caesar dressing, and some Asian sauces."
        )
    },
    {
        "category": "shellfish",
        "text": (
            "Shellfish allergens. Keywords: shrimp, crab, lobster, prawn, oyster, "
            "clam, scallop, squid, mussel. "
            "Turkish: karides, yengeç, ıstakoz, istiridye, midye, kalamar. "
            "Shellfish are non-vegan animal products."
        )
    },
    {
        "category": "gelatin",
        "text": (
            "Gelatin — hidden animal-derived ingredient. "
            "Keywords: gelatin, gelatine, collagen. "
            "Turkish: jelatin. "
            "E-codes: E441 (gelatin), E542 (bone phosphate, edible bone phosphate). "
            "Gelatin is derived from animal bones and skin — strictly non-vegan. "
            "Commonly hidden in gummy candies, marshmallows, Jell-O, yogurt, "
            "some capsule medications, and clarified juices."
        )
    },
    {
        "category": "carmine_and_animal_additives",
        "text": (
            "Hidden non-vegan E-codes and animal-derived additives. "
            "E120 = Carmine / Cochineal (red dye from insects) — non-vegan. "
            "E441 = Gelatin (animal bones/skin) — non-vegan. "
            "E542 = Bone phosphate (animal origin) — non-vegan. "
            "E901 = Beeswax (from bees) — non-vegan. "
            "E904 = Shellac (insect resin) — non-vegan. "
            "E910, E920 = L-cysteine (from feathers or hair) — non-vegan. "
            "E966 = Lactitol (milk-derived) — non-vegan. "
            "Turkish: karmin, bal mumu, jelatin, domuz yağı, iç yağı. "
            "Other: beeswax, shellac, isinglass (fish bladder), rennet, lard, tallow. "
            "These are commonly overlooked non-vegan ingredients hidden in processed foods."
        )
    },
]


class AllergenRAG:
    """
    Lightweight RAG system for allergen knowledge retrieval.
    Builds a FAISS index over allergen category documents and retrieves
    the most relevant ones based on input ingredient text.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not RAG_AVAILABLE:
            raise ImportError(
                "RAG dependencies missing. Run: pip install faiss-cpu sentence-transformers"
            )
        print("[RAG] Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        self.documents = ALLERGEN_DOCUMENTS
        self.index = None
        self.embeddings = None
        self._build_index()

    def _build_index(self):
        """Embeds all allergen documents and builds a FAISS flat L2 index."""
        texts = [doc["text"] for doc in self.documents]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)

        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine on normalized vecs
        self.index.add(self.embeddings)
        print(f"[RAG] FAISS index built — {len(self.documents)} allergen documents indexed.")

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        Retrieves the top-k most relevant allergen documents for the given query.

        Args:
            query: Ingredient list text from the user.
            top_k: Number of documents to retrieve.

        Returns:
            A formatted string of retrieved allergen context to inject into the LLM prompt.
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)

        retrieved = []
        for rank, idx in enumerate(indices[0], 1):
            doc = self.documents[idx]
            retrieved.append(f"[Context {rank} — {doc['category']}]: {doc['text']}")

        return "\n".join(retrieved)


# Singleton instance — initialized once and reused across calls
_rag_instance = None


def get_rag() -> "AllergenRAG | None":
    """Returns the singleton RAG instance, initializing it if needed."""
    global _rag_instance
    if not RAG_AVAILABLE:
        return None
    if _rag_instance is None:
        try:
            _rag_instance = AllergenRAG()
        except Exception as e:
            print(f"[RAG] Could not initialize: {e}")
            return None
    return _rag_instance


def retrieve_allergen_context(ingredient_text: str, top_k: int = 3) -> str:
    """
    Public interface: retrieve relevant allergen context for the given ingredient text.
    Returns empty string if RAG is unavailable.
    """
    rag = get_rag()
    if rag is None:
        return ""
    return rag.retrieve(ingredient_text, top_k=top_k)
