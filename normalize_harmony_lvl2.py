import re

def clean_dcml_core(label: str):

    if not isinstance(label, str):
        return None

    label = label.strip()

    # 1. eliminar TODO lo estructural DCML
    label = re.sub(r"\|.*", "", label)      # |HC |PAC etc
    label = re.sub(r"[{}\[\]\(\)]", "", label)
    label = re.sub(r"\^", "", label)
    label = re.sub(r"\.", "", label)

    # 2. eliminar alteraciones raras
    label = label.replace("#", "")
    label = label.replace("b", "b")  # se mantiene si quieres

    # 3. eliminar basura residual
    label = label.strip()

    return label if label != "" else None

def normalize_label_lvl2(label: str):

    if not isinstance(label, str):
        return None

    label = label.strip()

    if label == "":
        return None

    # -------------------------
    # 1. acordes especiales
    # -------------------------
    if any(x in label for x in ["Ger", "It", "Fr"]):
        return label

    # -------------------------
    # 2. DOMINANTES SECUNDARIAS (PRIMERO detectamos estructura)
    # -------------------------
    if "/" in label:
        return "V_secondary"

    # -------------------------
    # 3. eliminar inversiones y tensiones
    # -------------------------
    label = re.sub(r"\d+", "", label)
    label = re.sub(r"[+°ø]", "", label)

    # -------------------------
    # 4. normalizar triadas base
    # -------------------------
    mapping = {
        "I": "I",
        "ii": "ii",
        "iii": "iii",
        "IV": "IV",
        "V": "V",
        "vi": "vi",
        "vii": "vii"
    }

    return mapping.get(label, label)