import cv2
import numpy as np

def main():
    imagePath = 'immagine.png'

    img = cv2.imread(imagePath)

    if img is None:
        print(f"Errore: Impossibile caricare l'immagine da {imagePath}. Assicurati che il percorso sia corretto.")
        return

    squareSize = 20

    height, width, _ = img.shape

    whiteSquareX1 = (width // 2) - (squareSize // 2) - 150 
    whiteSquareY1 = (height // 2) - (squareSize // 2)
    whiteSquareX2 = whiteSquareX1 + squareSize
    whiteSquareY2 = whiteSquareY1 + squareSize

    blackSquareX1 = (width // 2) - (squareSize // 2) + 150 
    blackSquareY1 = (height // 2) - (squareSize // 2)
    blackSquareX2 = blackSquareX1 + squareSize
    blackSquareY2 = blackSquareY1 + squareSize

    displayImg = img.copy()

    cv2.rectangle(displayImg, (whiteSquareX1, whiteSquareY1),
                    (whiteSquareX2, whiteSquareY2), (255, 255, 255), 2)

    cv2.rectangle(displayImg, (blackSquareX1, blackSquareY1),
                    (blackSquareX2, blackSquareY2), (0, 0, 0), 2)

    cv2.imshow('Immagine con Quadrati', displayImg)

    print("Premi 'c' per calcolare le medie RGB, oppure 'q' per uscire.")

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('c'):
            whiteSquarePixels = img[whiteSquareY1:whiteSquareY2, whiteSquareX1:whiteSquareX2]
            blackSquarePixels = img[blackSquareY1:blackSquareY2, blackSquareX1:blackSquareX2]

            if whiteSquarePixels.size > 0:
                meanRgbWhite = np.mean(whiteSquarePixels, axis=(0, 1))
                print(f"Media RGB Quadrato Bianco: {meanRgbWhite}")
            else:
                print("Il quadrato bianco non contiene pixel validi.")

            if blackSquarePixels.size > 0:
                meanRgbBlack = np.mean(blackSquarePixels, axis=(0, 1))
                print(f"Media RGB Quadrato Nero: {meanRgbBlack}")
            else:
                print("Il quadrato nero non contiene pixel validi.")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()