import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Errore: Impossibile aprire la telecamera.")
        return

    squareSize = 20

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Errore: Impossibile leggere il frame dalla telecamera.")
            break

        height, width, _ = frame.shape

        whiteSquareX1 = (width // 2) - (squareSize // 2) - 150
        whiteSquareY1 = (height // 2) - (squareSize // 2)
        whiteSquareX2 = whiteSquareX1 + squareSize
        whiteSquareY2 = whiteSquareY1 + squareSize

        blackSquareX1 = (width // 2) - (squareSize // 2) + 150
        blackSquareY1 = (height // 2) - (squareSize // 2)
        blackSquareX2 = blackSquareX1 + squareSize
        blackSquareY2 = blackSquareY1 + squareSize

        cv2.rectangle(frame, (whiteSquareX1, whiteSquareY1),
                      (whiteSquareX2, whiteSquareY2), (255, 255, 255), 2)

        cv2.rectangle(frame, (blackSquareX1, blackSquareY1),
                      (blackSquareX2, blackSquareY2), (0, 0, 0), 2)

        cv2.imshow('Camera Feed con Quadrati', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            whiteSquarePixels = frame[whiteSquareY1:whiteSquareY2, whiteSquareX1:whiteSquareX2]
            blackSquarePixels = frame[blackSquareY1:blackSquareY2, blackSquareX1:blackSquareX2]

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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()