import re

def normalize_label_lvl1(label: str) -> str:
    """
    Nivel 1 de normalizacion.
    Conserva: numeral romano, alteraciones, cualidad basica,
    dominantes secundarias, acordes especiales.
    Elimina: inversiones, tensiones, extensiones, anotaciones
    estructurales DCML, cadencias, modulaciones.
    """
    if not isinstance(label, str):
        return None
    label = label.strip()
    if label == "":
        return None

    # 1. Acordes especiales (antes de cualquier limpieza)
    for special in ["Ger", "It", "Fr"]:
        if special in label:
            return special

    # 2. Prefijos de tonalidad con nombre de nota (Ab., C#., Bb., etc.)
    label = re.sub(r"^[A-G][#b]?\.", "", label)

    # 3. Todo desde | hasta fin (sufijos de cadencia: |PAC, |IAC, |EC...)
    label = re.sub(r"\|.*", "", label)

    # 4. Contenido entre corchetes cerrados y los corchetes mismos
    label = re.sub(r"\[.*?\]", "", label)

    # 5. Corchetes que abren sin cerrar: eliminar desde [ hasta fin
    label = re.sub(r"\[.*", "", label)

    # 6. Llaves y corchetes de cierre sueltos
    label = re.sub(r"[{}\]]", "", label)

    # 6. Prefijos de modulacion con numeral romano (I., ii., V., etc.)
    label = re.sub(r"^[#b]?(?:VII|VI|IV|V|III|II|I|vii|vi|iv|v|iii|ii|i)\.", "", label)

    # 7. Parentesis completos
    label = re.sub(r"\(.*?\)", "", label)

    # 8. Numeros (inversiones y tensiones)
    label = re.sub(r"\d+", "", label)

    # 9. Caret
    label = label.replace("^", "")

    # 10. Simbolos redundantes
    label = re.sub(r"[%+]", "", label)

    # 11. Normalizar dominantes secundarias profundas
    parts = label.split("/")
    if len(parts) > 2:
        label = "/".join(parts[:2])

    # 12. Slash final accidental
    label = label.strip("/")

    label = label.strip()
    if label == "":
        return None
    return label
