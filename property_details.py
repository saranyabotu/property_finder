import re


# =========================================================
# PROPERTY DETAIL EXTRACTION
# =========================================================

def clean_text(text):
    """
    Normalize OCR text for easier pattern matching.
    """

    if text is None:
        return ""

    text = str(text)

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# PRICE EXTRACTION
# =========================================================

def extract_price(text):

    text = clean_text(text)

    # -----------------------------------------------------
    # Lakhs
    # Examples:
    # 47 lakh
    # 47 lakhs
    # 75 LAKH
    # 22LAKHS
    # -----------------------------------------------------

    lakh_match = re.search(
        r"(?:₹|RS\.?|INR)?\s*"
        r"(\d+(?:\.\d+)?)\s*"
        r"(?:L|LAKH|LAKHS)\b",
        text,
        re.IGNORECASE
    )

    if lakh_match:

        number = lakh_match.group(1)

        return f"{number} LAKH"


    # -----------------------------------------------------
    # Rupee amounts
    # Examples:
    # ₹14,000
    # Rs.14000
    # RS 14000
    # 14000
    # -----------------------------------------------------

    rupee_match = re.search(
        r"(?:₹|RS\.?|INR)\s*"
        r"(\d[\d,]*(?:\.\d+)?)",
        text,
        re.IGNORECASE
    )

    if rupee_match:

        number = rupee_match.group(1)

        return f"₹{number}"


    # -----------------------------------------------------
    # Price followed by number
    # -----------------------------------------------------

    price_context = re.search(
        r"(?:PRICE|RENT|RATE|COST)"
        r"\s*[:=@-]?\s*"
        r"(?:₹|RS\.?|INR)?\s*"
        r"(\d[\d,]*(?:\.\d+)?)"
        r"\s*(L|LAKH|LAKHS)?",
        text,
        re.IGNORECASE
    )

    if price_context:

        number = price_context.group(1)

        unit = price_context.group(2)

        if unit:

            return f"{number} {unit.upper()}"

        return f"₹{number}"


    return "Not detected"


# =========================================================
# AREA EXTRACTION
# =========================================================

def extract_area(text):

    text = clean_text(text)

    # Examples:
    # 1000 SRF
    # 1000 SQFT
    # 1200 SQ FT
    # 1000 SFT
    # 1000 SQUARE FEET

    patterns = [

        r"(\d+(?:\.\d+)?)\s*(?:SQ\.?\s*FT|SQFT|SFT)\b",

        r"(\d+(?:\.\d+)?)\s*(?:SQ\.?\s*FEET)\b",

        r"(\d+(?:\.\d+)?)\s*(?:SRF)\b",

        r"(?:AREA)\s*[:=-]?\s*"
        r"(\d+(?:\.\d+)?)\s*"
        r"(?:SQ\.?\s*FT|SQFT|SFT|SRF)?"
    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            value = match.group(1)

            return f"{value} sq.ft"


    return "Not detected"


# =========================================================
# BHK EXTRACTION
# =========================================================

def extract_bhk(text):

    text = clean_text(text)

    # -----------------------------------------------------
    # Direct BHK
    # -----------------------------------------------------

    match = re.search(
        r"\b(\d+)\s*BHK\b",
        text,
        re.IGNORECASE
    )

    if match:

        return f"{match.group(1)} BHK"


    # -----------------------------------------------------
    # Bedroom count
    # -----------------------------------------------------

    match = re.search(
        r"\b(\d+)\s*"
        r"(?:BEDROOMS?|BED\s*ROOMS?)\b",
        text,
        re.IGNORECASE
    )

    if match:

        return f"{match.group(1)} BHK"


    return "Not detected"


# =========================================================
# LOCATION EXTRACTION
# =========================================================

def extract_location(text):

    text = clean_text(text)

    # -----------------------------------------------------
    # First try explicit location labels
    # -----------------------------------------------------

    location_patterns = [

        r"(?:LOCATION|LOC|AREA)\s*[:=-]\s*"
        r"([A-Za-z][A-Za-z .'-]{2,50})",

        r"(?:AT)\s+"
        r"([A-Za-z][A-Za-z .'-]{2,50})",

        r"(?:NEAR)\s+"
        r"([A-Za-z][A-Za-z .'-]{2,50})"
    ]


    for pattern in location_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            location = match.group(1).strip()

            # Remove trailing property-related words
            location = re.sub(
                r"\b(?:CALL|PRICE|RENT|BEDROOMS?|"
                r"BATHROOMS?|KITCHEN|PHONE)\b.*$",
                "",
                location,
                flags=re.IGNORECASE
            ).strip()

            if len(location) >= 3:

                return location


    # -----------------------------------------------------
    # Known place names that appear in the dataset
    # -----------------------------------------------------

    known_locations = [

        "NELAMANGALA",

        "VIZIANAGARAM",

        "AYYANNAPETA",

        "PUPPALAGUDA",

        "MANIKONDA",

        "KARELIBAUG",

        "VIJAYAWADA",

        "MARUTHI NAGAR",

        "MOOKANDAPALLI",

        "HOSUR",

        "RK BEACH",

        "RK BEACH AREA"
    ]


    upper_text = text.upper()


    # Prefer longer place names first
    known_locations.sort(
        key=len,
        reverse=True
    )


    for location in known_locations:

        if location in upper_text:

            return location.title()


    return "Not detected"


# =========================================================
# MAIN EXTRACTION FUNCTION
# =========================================================

def extract_property_details(text):

    text = clean_text(text)


    price = extract_price(
        text
    )


    area = extract_area(
        text
    )


    bhk = extract_bhk(
        text
    )


    location = extract_location(
        text
    )


    return {

        "price": price,

        "area": area,

        "bhk": bhk,

        "location": location
    }