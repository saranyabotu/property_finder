import cv2
import os

# Image to annotate
image_path = "dataset/images/image_014.png"

# Load image
image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

# Get image dimensions
image_height, image_width = image.shape[:2]

display_image = image.copy()

drawing = False
start_x = 0
start_y = 0


def save_annotation(x1, y1, x2, y2):

    # Keep coordinates inside image
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    # Correct coordinate order
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    # Convert to YOLO format
    center_x = ((left + right) / 2) / image_width
    center_y = ((top + bottom) / 2) / image_height

    box_width = (right - left) / image_width
    box_height = (bottom - top) / image_height

    # Create labels folder
    os.makedirs("dataset/labels", exist_ok=True)

    # Automatically create correct label filename
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    label_file = f"dataset/labels/{image_name}.txt"

    # Save YOLO annotation
    with open(label_file, "w") as file:
        file.write(
            f"0 {center_x:.6f} {center_y:.6f} "
            f"{box_width:.6f} {box_height:.6f}\n"
        )

    print("\n====================================")
    print("ANNOTATION SAVED SUCCESSFULLY!")
    print("====================================")
    print("Image:", image_name)
    print("Label file:", label_file)
    print(
        f"YOLO: 0 {center_x:.6f} {center_y:.6f} "
        f"{box_width:.6f} {box_height:.6f}"
    )


def mouse_callback(event, x, y, flags, param):

    global drawing
    global start_x, start_y
    global display_image

    # Mouse button pressed
    if event == cv2.EVENT_LBUTTONDOWN:

        drawing = True

        start_x = max(0, min(x, image_width - 1))
        start_y = max(0, min(y, image_height - 1))

    # Mouse moving
    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:

            current_x = max(0, min(x, image_width - 1))
            current_y = max(0, min(y, image_height - 1))

            display_image = image.copy()

            cv2.rectangle(
                display_image,
                (start_x, start_y),
                (current_x, current_y),
                (0, 255, 0),
                5
            )

    # Mouse button released
    elif event == cv2.EVENT_LBUTTONUP:

        drawing = False

        end_x = max(0, min(x, image_width - 1))
        end_y = max(0, min(y, image_height - 1))

        display_image = image.copy()

        cv2.rectangle(
            display_image,
            (start_x, start_y),
            (end_x, end_y),
            (0, 255, 0),
            5
        )

        print("\nBounding box:")
        print("Top-left:", start_x, start_y)
        print("Bottom-right:", end_x, end_y)

        # Automatically save
        save_annotation(
            start_x,
            start_y,
            end_x,
            end_y
        )


# Create window
window_name = "Property Advertisement Annotation"

cv2.namedWindow(window_name)

cv2.setMouseCallback(
    window_name,
    mouse_callback
)

print("====================================")
print("PROPERTY ADVERTISEMENT ANNOTATION")
print("====================================")
print("Image:", image_path)
print("Width:", image_width)
print("Height:", image_height)
print()
print("Click and drag around the ENTIRE")
print("property advertisement.")
print()
print("Release the mouse to save automatically.")
print("Press Q to close the window.")
print("====================================")


while True:

    cv2.imshow(window_name, display_image)

    key = cv2.waitKey(20) & 0xFF

    if key == ord("q"):
        break


cv2.destroyAllWindows()