import cv2
import numpy as np
import glob
import os

# Cartelle di input e output
input_folder = "output_png"
output_folder = "clean_cbis_png"

# Crea la cartella di output se non esiste
os.makedirs(output_folder, exist_ok=True)

# Ottieni tutte le immagini .png
image_files = sorted(glob.glob(os.path.join(input_folder, "*.png")))
print(f"Trovate {len(image_files)} immagini nella cartella '{input_folder}'.")

for img_path in image_files:
    base_name = os.path.basename(img_path)
    out_path = os.path.join(output_folder, base_name)

    # ✅ Se il file è già stato elaborato, lo salta
    if os.path.exists(out_path):
        print(f"⏭️  {base_name} già esistente, salto.")
        continue

    # Leggi immagine
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Errore nel leggere {img_path}, salto l'immagine.")
        continue

    hh, ww = img.shape[:2]

    # Converti in grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ritaglia 40 pixel su tutti i lati
    gray = gray[40:hh-40, 40:ww-40]

    # Aggiungi bordo nero di 40 pixel
    gray = cv2.copyMakeBorder(gray, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=0)

    # Threshold di Otsu
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

    # Morfologia: chiusura + apertura
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # Contorno più grande
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    if len(contours) == 0:
        print(f"⚠️ Nessun contorno trovato per {img_path}, salto.")
        continue
    big_contour = max(contours, key=cv2.contourArea)

    # Maschera
    mask = np.zeros((hh, ww), dtype=np.uint8)
    cv2.drawContours(mask, [big_contour], 0, 255, cv2.FILLED)

    # Dilata la maschera
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (305, 305))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Applica la maschera
    result = cv2.bitwise_and(img, img, mask=mask)

    # Salva il risultato nella nuova cartella
    cv2.imwrite(out_path, result)
    print(f"✅ Salvato: {out_path}")

print(f"\n🎯 Tutte le immagini elaborate e salvate nella cartella '{output_folder}'.")
