import re


# =========================================================
# PROPERTY TYPE CLASSIFIER
# =========================================================

def classify_property(text):
    """
    Classifies a property advertisement as:

    FOR SALE
    FOR RENT
    UNKNOWN

    The function handles common variations found in
    real-world advertisement boards.
    """

    if text is None:
        return "UNKNOWN"

    # -----------------------------------------------------
    # Normalize OCR text
    # -----------------------------------------------------

    text = str(text).upper()

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    # Remove repeated spaces
    text = re.sub(r"\s+", " ", text).strip()


    # =====================================================
    # SALE PATTERNS
    # =====================================================

    sale_patterns = [

        r"\bFOR\s*SALE\b",

        r"\bFORSALE\b",

        r"\bFOR[- ]SALE\b",

        r"\bHOUSE\s+FOR\s+SALE\b",

        r"\bFLAT\s+FOR\s+SALE\b",

        r"\bPROPERTY\s+FOR\s+SALE\b",

        r"\bSALE\b",

        r"\bSELLING\b",

        r"\bSELL\s+NOW\b"
    ]


    # =====================================================
    # RENT PATTERNS
    # =====================================================

    rent_patterns = [

        r"\bFOR\s+RENT\b",

        r"\bFORRENT\b",

        r"\bFOR[- ]RENT\b",

        r"\bHOUSE\s+FOR\s+RENT\b",

        r"\bFLAT\s+FOR\s+RENT\b",

        r"\bPROPERTY\s+FOR\s+RENT\b",

        r"\bRENT\b",

        r"\bTO[- ]LET\b",

        r"\bTOLET\b",

        r"\bAVAILABLE\s+FOR\s+RENT\b",

        r"\bRENTAL\b"
    ]


    # =====================================================
    # CHECK RENT FIRST
    # =====================================================

    for pattern in rent_patterns:

        if re.search(
            pattern,
            text
        ):

            return "FOR RENT"


    # =====================================================
    # CHECK SALE
    # =====================================================

    for pattern in sale_patterns:

        if re.search(
            pattern,
            text
        ):

            return "FOR SALE"


    # =====================================================
    # UNKNOWN
    # =====================================================

    return "UNKNOWN"