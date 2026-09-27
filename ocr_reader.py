import easyocr

# Create OCR reader once
reader = easyocr.Reader(['en'])


def read_text(image_path, output_path=None):
    """
    Read text from the given image.

    Parameters:
        image_path: path of the image to read
        output_path: optional file to save OCR text

    Returns:
        Extracted text as a single string
    """

    result = reader.readtext(image_path)

    extracted_text = []

    for detection in result:
        text = detection[1]
        extracted_text.append(text)

    final_text = "\n".join(extracted_text)

    # Save OCR result if an output file is provided
    if output_path:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(final_text)

    return final_text


# Test the OCR reader
if __name__ == "__main__":
    image_path = "dataset/images/image_002.png"
    output_path = "backend/ocr_result_002.txt"

    text = read_text(image_path, output_path)

    print("OCR result:")
    print(text)
    print("\nOCR result saved successfully!")