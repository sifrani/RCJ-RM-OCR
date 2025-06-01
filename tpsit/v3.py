import cv2
import numpy as np
from collections import deque
import threading
import time

latestFrame = None
frameLock = threading.Lock()
runningThreads = True

ocrResults = []
resultsLock = threading.Lock()

def getNeighbours(x, y, image):
    x1, y1, x_plus_1, y_plus_1 = x - 1, y - 1, x + 1, y + 1
    return [image[x1][y], image[x1][y_plus_1], image[x][y_plus_1], image[x_plus_1][y_plus_1],
            image[x_plus_1][y], image[x_plus_1][y1], image[x][y1], image[x1][y1]]

def countTransitions(neighbours):
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
                    countTransitions(n) == 1 and
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
                    countTransitions(n) == 1 and
                    p2 * p4 * p8 == 0 and
                    p2 * p6 * p8 == 0):
                    changing2.append((x, y))
        for x, y in changing2:
            imageThinned[x][y] = 0
    return imageThinned

def checkLetter(group, originalImageMatrix, maxX, minX, maxY, minY):
    binaryMatrix = np.zeros((maxY - minY + 1, maxX - minX + 1), dtype=np.uint8)

    for px in group:
        b, g, r = originalImageMatrix[px[1]][px[0]]
        if int(r) + int(g) + int(b) < 300: 
            binaryMatrix[px[1] - minY, px[0] - minX] = 1
    return binaryMatrix

def serafinSimone(skeleton):
    rows, cols = skeleton.shape
    vertexies = []

    dirs = [(0, 1),(-1, 1),(-1, 0),(-1, -1),(0, -1),( 1, -1),(1, 0),(1, 1)]

    for x in range(rows):
        for y in range(cols):
            if skeleton[x, y]:
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
                
                # Check for wrap-around for the last and first neighbor with boundary checks
                if (0 <= x+dirs[0][0] < rows and 0 <= y+dirs[0][1] < cols and skeleton[x+dirs[0][0],y+dirs[0][1]]) and \
                   (0 <= x+dirs[len(dirs)-1][0] < rows and 0 <= y+dirs[len(dirs)-1][1] < cols and skeleton[(x+dirs[len(dirs)-1][0],y+dirs[len(dirs)-1][1])]):
                    l-=1
               
                if l%2 == 1:
                    vertexies.append((x,y))
    
    def rlerp(a, b, t_start=0.2, t_end=0.8):    
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
                    punti.append((x - sx, y + sy))
                y += sy

        punti.append((x, y))
        visti = set()
        connessi = []
        for p in punti:
            if p not in visti:
                connessi.append(p)
                visti.add(p)

        n = len(connessi)
        startIndex = int(t_start * n)
        endIndex = int(t_end * n)

        return connessi[startIndex:endIndex]


    if len(vertexies) == 6:
        return "H"
    if len(vertexies) == 2:
        for x,y in rlerp(vertexies[0], vertexies[1], 0.2, 0.8):
            if 0 <= x < rows and 0 <= y < cols and skeleton[x][y]:
                return "S"
        return "U"

    return None

def cameraCaptureThread():
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Errore: Impossibile aprire la webcam.")
        runningThreads = False
        return

    print("Thread della webcam avviato.")
    while runningThreads:
        ret, frame = cap.read()
        if not ret:
            print("Errore: Impossibile leggere il frame dalla webcam.")
            break
        
        with frameLock:
            global latestFrame
            latestFrame = frame.copy() 

    cap.release()
    print("Thread della webcam terminato.")

def ocrProcessingThread():
    global latestFrame, runningThreads, frameLock, ocrResults, resultsLock

    print("Thread di elaborazione OCR avviato.")
    while runningThreads:
        currentFrameToProcess = None
        with frameLock:
            if latestFrame is not None:
                currentFrameToProcess = latestFrame.copy()

        if currentFrameToProcess is not None:
            height, width, _ = currentFrameToProcess.shape

            imgBools = np.zeros((width, height), dtype=bool)
            imgBools_ = np.zeros((width, height), dtype=bool)

            rng = 50
            for x in range(width):
                for y in range(height):
                    b, g, r = currentFrameToProcess[y][x]
                    if abs(int(r) - int(g)) + abs(int(g) - int(b)) + abs(int(b) - int(r)) < rng:
                        imgBools_[x][y] = True

            for x_ in range(2, width - 2):
                for y_ in range(2, height - 2):
                    allTrueInWindow = True
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            if not imgBools_[x_ + dx][y_ + dy]:
                                allTrueInWindow = False
                                break
                        if not allTrueInWindow:
                            break
                    if allTrueInWindow:
                        imgBools[x_][y_] = True

            def bfs(startX, startY, currentImgBoolsForBfs):
                pipe = deque([(startX, startY)])
                currentImgBoolsForBfs[startX][startY] = False
                g = [(startX, startY)]
                
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                
                while pipe:
                    cx, cy = pipe.popleft()
                    for dx, dy in dirs:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < currentImgBoolsForBfs.shape[0] and 0 <= ny < currentImgBoolsForBfs.shape[1] and currentImgBoolsForBfs[nx][ny]:
                            currentImgBoolsForBfs[nx][ny] = False
                            pipe.append((nx, ny))
                            g.append((nx, ny))
                return g

            groups = []
            imgBoolsMutableForBfs = imgBools.copy() 
            
            for i in range(width):
                for j in range(height):
                    if imgBoolsMutableForBfs[i][j]:
                        g = bfs(i, j, imgBoolsMutableForBfs)
                        if g:
                            groups.append(g)
            
            currentOcrResults = []
            for g in groups:
                if len(g) > 10: 
                    minX, maxX = width, 0
                    minY, maxY = height, 0

                    for px in g:
                        minX = min(minX, px[0])
                        maxX = max(maxX, px[0])
                        minY = min(minY, px[1])
                        maxY = max(maxY, px[1])
                    
                    sk = checkLetter(g, currentFrameToProcess, maxX, minX, maxY, minY)
                    
                    detectedLetter = None
                    if sk.size > 0 and np.any(sk == 1):
                        detectedLetter = serafinSimone(sk)
                    
                    if detectedLetter:
                        currentOcrResults.append(((minX, minY, maxX, maxY), detectedLetter))
            
            with resultsLock:
                ocrResults.clear() 
                ocrResults.extend(currentOcrResults) 

        time.sleep(1)

    print("Thread di elaborazione OCR terminato.")

def mainDisplayLoop():
    global latestFrame, runningThreads, frameLock, ocrResults, resultsLock

    print("Thread di visualizzazione principale avviato.")
    while runningThreads:
        currentFrame = None
        with frameLock:
            if latestFrame is not None:
                currentFrame = latestFrame.copy()

        if currentFrame is not None:
            displayFrame = currentFrame.copy()

            currentResultsToDisplay = []
            with resultsLock:
                currentResultsToDisplay.extend(ocrResults)

            for (minX, minY, maxX, maxY), text in currentResultsToDisplay:
                cv2.rectangle(displayFrame, (minX, minY), (maxX, maxY), (0, 255, 0), 1)
                
                textX = minX
                textY = minY - 10 if minY - 10 > 10 else minY + 20 
                
                cv2.putText(displayFrame, text, (textX, textY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Webcam Feed & OCR Results', displayFrame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            runningThreads = False
            break
        
    print("Thread di visualizzazione principale terminato.")

if __name__ == '__main__':
    cameraThread = threading.Thread(target=cameraCaptureThread)
    ocrThread = threading.Thread(target=ocrProcessingThread)
    
    cameraThread.start()
    ocrThread.start()

    mainDisplayLoop()

    cameraThread.join()
    ocrThread.join()
    cv2.destroyAllWindows()
    print("Programma terminato.")