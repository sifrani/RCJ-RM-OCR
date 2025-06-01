import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


imgMatrix = []


def getNeighbours(x, y, image):
    xMinus1, yMinus1, xPlus1, yPlus1 = x - 1, y - 1, x + 1, y + 1
    return [image[xMinus1][y], image[xMinus1][yPlus1], image[x][yPlus1], image[xPlus1][yPlus1],
            image[xPlus1][y], image[xPlus1][yMinus1], image[x][yMinus1], image[xMinus1][yMinus1]]

def getTransitions(neighbours):
    n = neighbours + neighbours[0:1]
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

def zhangSuen(image):
    imageThinned = image.copy()
    changing1 = changing2 = 1
    while changing1 or changing2:
        changing1 = []
        rows, columns = imageThinned.shape
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                p2, p3, p4, p5, p6, p7, p8, p9 = n = getNeighbours(x, y, imageThinned)
                if (imageThinned[x][y] == 1 and
                    2 <= sum(n) <= 6 and
                    getTransitions(n) == 1 and
                    p2 * p4 * p6 == 0 and
                    p4 * p6 * p8 == 0):
                    changing1.append((x, y))
        for x, y in changing1:
            imageThinned[x][y] = 0
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                p2, p3, p4, p5, p6, p7, p8, p9 = n = getNeighbours(x, y, imageThinned)
                if (imageThinned[x][y] == 1 and
                    2 <= sum(n) <= 6 and
                    getTransitions(n) == 1 and
                    p2 * p4 * p8 == 0 and
                    p2 * p6 * p8 == 0):
                    changing2.append((x, y))
        for x, y in changing2:
            imageThinned[x][y] = 0
    return imageThinned

def checkLetter(group, imgMatrix, maxX, minX, maxY, minY):
    binaryMatrix = np.zeros((maxY-minY +1, maxX-minX+1), dtype=np.uint8)

    avg = 0


    for px in group:
        r, g, b = imgMatrix[px[1]][px[0]]
        avg += int(r) + int(g) + int(b)

    avg /= len(group)

    for px in group:
        r, g, b = imgMatrix[px[1]][px[0]]
        if int(r) + int(g) + int(b) < avg:
            binaryMatrix[px[1]-minY, px[0]-minX] = 1

    skeleton = zhangSuen(binaryMatrix)

    return skeleton

def serafinSimone(skeleton):
    rows, cols = skeleton.shape
    vertexies = []

    dirs = [(0, 1),(-1, 1),(-1, 0),(-1, -1),(0, -1),( 1, -1),(1, 0),(1, 1)]


    for x in range(1,rows-1):
        for y in range(1,cols-1):
            if skeleton[x, y]:
                # In OpenCV, modify directly imgMatrix if you want to see the changes
                # imgMatrix[x][y] = (50,0,100) # This won't work directly for visualization in OpenCV
                l = 0
                i = 0
                while i < len(dirs):
                    nx, ny = x+dirs[i][0], y+dirs[i][1]
                    if 0 <= nx < rows and 0 <= ny < cols and skeleton[nx, ny]:
                        l += 1
                        i += 1
                        while i < len(dirs) and 0 <= x+dirs[i][0] < rows and 0 <= y+dirs[i][1] < cols and skeleton[(x+dirs[i][0],y+dirs[i][1])]:
                            i+=1
                    i+=1

                # Check for wrap-around for the last and first neighbor
                if skeleton[x+dirs[0][0],y+dirs[0][1]] and skeleton[(x+dirs[len(dirs)-1][0],y+dirs[len(dirs)-1][1])]:
                    l-=1

                if l%2 == 1:
                    vertexies.append((x,y))
                    # imgMatrix[x][y] = (250,0,100) # This won't work directly for visualization in OpenCV


    def rlerp(a, b, tStart=0.2, tEnd=0.8):
        x0, y0 = a
        x1, y1 = b

        punti = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                punti.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                    # passo diagonale → aggiungi passo verticale
                    punti.append((x + sx, y - sy))
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                punti.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                    # passo diagonale → aggiungi passo orizzontale
                    punti.append((x - sx, y + sy))
                y += sy

        punti.append((x, y))

        # Rimuovi duplicati preservando l'ordine
        visti = set()
        connessi = []
        for p in punti:
            if p not in visti:
                connessi.append(p)
                visti.add(p)

        # Calcola gli indici LERP
        n = len(connessi)
        startIndex = int(tStart * n)
        endIndex = int(tEnd * n)

        return connessi[startIndex:endIndex]


    if len(vertexies) == 6:
        return "H"
    if len(vertexies) == 2:
        for x,y in rlerp(vertexies[0], vertexies[1], 0.2, 0.8):
            # Ensure coordinates are within bounds before accessing skeleton
            if 0 <= x < rows and 0 <= y < cols and skeleton[x][y]:
                return "S"
        return "U"

    return None

def convertImageToMatrix(imagePath):
    # OpenCV reads images in BGR format
    img = cv2.imread(imagePath)
    return img

def showImageAndMatrixInOpenCV(imagePath):
    global imgMatrix

    imgMatrix = convertImageToMatrix(imagePath)
    if imgMatrix is None:
        return

    height, width, _ = imgMatrix.shape
    print(width, height)

    # Create a copy of the original image for drawing
    displayImage = imgMatrix.copy()

    # --- Start of your original gray area selection logic ---
    imgBools = np.zeros((width, height), dtype=bool)
    imgBoolsUnderscore = np.zeros((width, height), dtype=bool)

    rng = 54
    for x in range(width):
        for y in range(height):
            # OpenCV loads as BGR, so imgMatrix[y][x] gives [B, G, R]
            b, g, r = imgMatrix[y][x]
            if abs(int(r) - int(g)) + abs(int(g) - int(b)) + abs(int(b) - int(r)) < rng:
                imgBoolsUnderscore[x][y] = True
    # This part applies a 5x5 window check, similar to a morphological operation
    for x_ in range(2, width - 2):
        for y_ in range(2, height - 2):
            # Check if all pixels in the 5x5 neighborhood are True in imgBoolsUnderscore
            allTrueInWindow = True
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if not imgBoolsUnderscore[x_ + dx][y_ + dy]:
                        allTrueInWindow = False
                        break
                if not allTrueInWindow:
                    break
            if allTrueInWindow:
                imgBools[x_][y_] = True

    def bfs(startX, startY, currentImgBools):
        pipe = deque([(startX, startY)])
        currentImgBools[startX][startY] = False
        g = [(startX, startY)]

        # 8-directional connectivity
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while pipe:
            cx, cy = pipe.popleft()
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                # Ensure nx, ny are within bounds of currentImgBools
                if 0 <= nx < currentImgBools.shape[0] and 0 <= ny < currentImgBools.shape[1] and currentImgBools[nx][ny]:
                    currentImgBools[nx][ny] = False
                    pipe.append((nx, ny))
                    g.append((nx, ny))
        return g

    groups = []
    # Create a mutable copy for BFS to modify
    imgBoolsMutable = imgBools.copy()

    for i in range(width):
        for j in range(height):
            if imgBoolsMutable[i][j]:
                g = bfs(i, j, imgBoolsMutable)
                if g:
                    groups.append(g)
    # --- End of your original gray area selection logic ---

    for g in groups:
        if len(g) > 100:
            minX, maxX = width, 0
            minY, maxY = height, 0

            # Calculate bounding box for the current group
            for px in g:
                minX = min(minX, px[0])
                maxX = max(maxX, px[0])
                minY = min(minY, px[1])
                maxY = max(maxY, px[1])

            # Draw bounding box on the display image
            # OpenCV rectangle: (img, (x1,y1), (x2,y2), color (BGR), thickness)
            cv2.rectangle(displayImage, (minX, minY), (maxX, maxY), (0, 255, 0), 1)

            # Pass the group, original imgMatrix, and bounding box to checkLetter
            sk = checkLetter(g, imgMatrix, maxX, minX, maxY, minY)

            # Check if skeletonization resulted in a non-empty skeleton before passing to serafinSimone
            if sk.size > 0 and np.any(sk == 1):
                v = serafinSimone(sk)
                if v:
                    print(f"Detected: {v} in bounding box ({minX}, {minY}, {maxX-minX+1}, {maxY-minY+1})")
            else:
                print(f"No skeleton found for a group in bounding box ({minX}, {minY}, {maxX-minX+1}, {maxY-minY+1})")

    # Display the final image with bounding boxes
    cv2.imshow('Image with Detected Letters', displayImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


imagePath = 'immagine5.png'
showImageAndMatrixInOpenCV(imagePath)