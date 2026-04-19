import cv2
import numpy as np
import glob
import os


# ---------------------------------------------------------------------------
# Script purpose:
# - Remove background and artifacts from CBIS mammography PNG images
# - Keep only the main breast region using thresholding and morphology
# - Save cleaned images to a new folder
# ---------------------------------------------------------------------------



input_folder = "output_png"
output_folder = "clean_cbis_png"

os.makedirs(output_folder, exist_ok=True)


image_files = sorted(glob.glob(os.path.join(input_folder, "*.png")))
print(f"Found {len(image_files)} images in the folder'{input_folder}'.")


# Process each image independently

for img_path in image_files:
    base_name = os.path.basename(img_path)
    out_path = os.path.join(output_folder, base_name)

    if os.path.exists(out_path):
        print(f"{base_name} already existing.")
        continue

    
    img = cv2.imread(img_path)
    if img is None:
        print(f" reading error {img_path}, skip image.")
        continue

    # Original image dimensions
    hh, ww = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove a fixed border to avoid scanner edges and artifacts
    gray = gray[40:hh-40, 40:ww-40]

    # Restore original size by padding with black borders
    gray = cv2.copyMakeBorder(gray, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=0)

    # Apply Otsu thresholding to separate foreground from background
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]


    # Morphological operations to clean the binary mask
    # - Close: fill holes inside the breast region
    # - Open: remove small isolated artifacts

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)


    # Find external contours in the cleaned binary mask

    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    if len(contours) == 0:
        print(f"No contour found for {img_path}, skip.")
        continue

    # Select the largest contour
    big_contour = max(contours, key=cv2.contourArea)


    # Create a binary mask from the largest contour
    mask = np.zeros((hh, ww), dtype=np.uint8)
    cv2.drawContours(mask, [big_contour], 0, 255, cv2.FILLED)

    # Dilate the mask to ensure full coverage of breast boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (305, 305))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Apply the mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imwrite(out_path, result)
    print(f"Saved: {out_path}")

print(f"\nAll images saved in the folder '{output_folder}'.")
