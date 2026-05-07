"""
Allergen Knowledge Base
Covers: standard English names, Turkish ingredient names, E-codes, and hidden allergen mappings.
"""

ALLERGEN_DB = {
    # ─── MILK / DAIRY ─────────────────────────────────────────────────────────
    "milk": [
        # English
        "milk", "casein", "whey", "lactose", "butter", "cheese", "cream",
        "yogurt", "dairy", "curd", "lactalbumin", "lactoglobulin", "ghee",
        # Turkish
        "süt", "peynir", "tereyağı", "yoğurt", "krema", "kazein",
        # E-codes
        "e101",   # Riboflavin (sometimes from dairy)
    ],

    # ─── GLUTEN / WHEAT ───────────────────────────────────────────────────────
    "gluten": [
        # English
        "wheat", "barley", "rye", "oats", "spelt", "malt", "seitan",
        "semolina", "flour", "breadcrumbs", "starch",
        # Turkish
        "buğday", "arpa", "çavdar", "yulaf", "un", "nişasta", "gluten",
        "ekmek kırıntısı",
        # E-codes
        "e1404", "e1410", "e1412", "e1414", "e1420", "e1422",  # Modified starches (often wheat-based)
    ],

    # ─── NUTS ─────────────────────────────────────────────────────────────────
    "nuts": [
        # English
        "almond", "hazelnut", "walnut", "cashew", "pecan", "macadamia",
        "pistachio", "peanut", "pine nut", "brazil nut", "chestnut",
        # Turkish
        "badem", "fındık", "ceviz", "kaju", "antep fıstığı", "yer fıstığı",
        "çam fıstığı", "kestane",
    ],

    # ─── SOY ──────────────────────────────────────────────────────────────────
    "soy": [
        # English
        "soy", "soybean", "tofu", "edamame", "tempeh", "miso", "tamari",
        "soy sauce", "soya",
        # Turkish
        "soya", "soya fasulyesi",
        # E-codes (lecithin is commonly from soy)
        "e322",   # Lecithin (usually soy-derived)
        "lecithin",
    ],

    # ─── EGGS ─────────────────────────────────────────────────────────────────
    "eggs": [
        # English
        "egg", "albumen", "globulin", "ovomucoid", "mayonnaise", "meringue",
        "lysozyme",
        # Turkish
        "yumurta",
        # E-codes
        "e1105",  # Lysozyme (from egg white)
    ],

    # ─── FISH ─────────────────────────────────────────────────────────────────
    "fish": [
        # English
        "fish", "cod", "salmon", "tuna", "tilapia", "anchovy", "sardine",
        "mackerel", "halibut", "bass", "flounder",
        # Turkish
        "balık", "morina", "somon", "ton balığı", "hamsi", "sardalya",
        # E-codes
        "e471",   # Mono- and diglycerides (can be fish-derived)
    ],

    # ─── SHELLFISH ────────────────────────────────────────────────────────────
    "shellfish": [
        # English
        "shrimp", "crab", "lobster", "prawn", "oyster", "clam", "scallop",
        "squid", "mussel",
        # Turkish
        "karides", "yengeç", "ıstakoz", "istiridye", "midye", "kalamar",
    ],

    # ─── GELATIN (Non-Vegan, Often Hidden) ────────────────────────────────────
    "gelatin": [
        # English
        "gelatin", "gelatine", "collagen",
        # Turkish
        "jelatin",
        # E-codes
        "e441",   # Gelatin
        "e542",   # Bone phosphate (edible bone phosphate — animal origin)
    ],

    # ─── CARMINE / COCHINEAL (Non-Vegan Red Dye) ──────────────────────────────
    "carmine": [
        # English
        "carmine", "cochineal", "carminic acid",
        # Turkish
        "karmin",
        # E-codes
        "e120",   # Cochineal / Carminic acid
    ],

    # ─── OTHER HIDDEN NON-VEGAN E-CODES ───────────────────────────────────────
    "animal_additives": [
        "e120",   # Carmine (insect-derived)
        "e441",   # Gelatin (animal bone/skin)
        "e542",   # Bone phosphate
        "e901",   # Beeswax
        "e904",   # Shellac (insect resin)
        "e910",   # L-cysteine (from feathers/hair)
        "e920",   # L-cysteine hydrochloride
        "e966",   # Lactitol (milk-derived)
        "beeswax", "shellac", "isinglass", "rennet", "lard", "tallow",
        # Turkish
        "bal mumu", "domuz yağı", "iç yağı",
    ],
}
