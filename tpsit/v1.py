try:
    import cum
except:
    ...

import pygame
import numpy as np
from PIL import Image
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
    binaryMatrix = np.zeros((maxX-minX, maxY-minY), dtype=np.uint8)

    for px in group:
        r, g, b = imgMatrix[px[1]][px[0]]
        if int(r) + int(g) + int(b) < 300:
            binaryMatrix[px[1]-minY, px[0]-minX] = 1

    skeleton = zhangSuen(binaryMatrix)

    """
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(binary_array, cmap=plt.cm.gray)
    ax[0].set_title("Binary from group")
    ax[0].axis("off")

    ax[1].imshow(skeleton, cmap=plt.cm.gray)
    ax[1].set_title("Zhang-Suen Skeleton")
    ax[1].axis("off")
    plt.show()
    """
    return skeleton

def serafinSimone(skeleton):
    rows, cols = skeleton.shape
    vertexies = []

    dirs = [(0, 1),(-1, 1),(-1, 0),(-1, -1),(0, -1),( 1, -1),(1, 0),(1, 1)]


    for x in range(rows):
        for y in range(cols):
            if skeleton[x, y]:
                imgMatrix[x][y] = (50,0,100)
                l = 0
                i = 0
                while i < len(dirs):
                    if skeleton[(x+dirs[i][0],y+dirs[i][1])]:
                        l += 1
                        i += 1
                        while i < len(dirs) and skeleton[(x+dirs[i][0],y+dirs[i][1])]:
                            i+=1
                    i+=1
                if skeleton[(x+dirs[0][0],y+dirs[0][1])] and skeleton[(x+dirs[len(dirs)-1][0],y+dirs[len(dirs)-1][1])]:
                    l-=1

                if l%2 == 1:
                    vertexies.append((x,y))
                    imgMatrix[x][y] = (250,0,100)


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
            if skeleton[x][y]:
                return "S"
        return "U"

    return None

def convertImageToMatrix(imagePath):
    img = Image.open(imagePath)
    imgArray = np.array(img)
    imgArrayInverted = np.flipud(imgArray)
    return imgArrayInverted

def showImageAndMatrixInPygame(imagePath):
    global imgMatrix
    pygame.init()

    imgMatrix = convertImageToMatrix(imagePath)

    height, width, _ = imgMatrix.shape
    print(width,height)
    screen = pygame.display.set_mode((width, height))

    imgBools = [[False for y in range(height)] for x in range(width)]
    imgBoolsUnderscore = [[False for y in range(height)] for x in range(width)]


    for x in range(width):
        for y in range(height):
            r,g,b = imgMatrix[y][x]

            """
            r+=1
            g+=1
            b+=1

            rng = 0.1
            if abs(r/g-1) < rng and abs(g/b-1) < rng and abs(b/r-1) < rng:
                imgBoolsUnderscore[x][y] = True
            """

            rng = 40
            if abs(int(r)-int(g)) + abs(int(g)-int(b)) + abs(int(b)-int(r)) < rng:
                imgBoolsUnderscore[x][y] = True


    for x_ in range(2, width-2):
        for y_ in range(2, height-2):
            if imgBoolsUnderscore[x_][y_] == True and not False in [imgBoolsUnderscore[x_+x][y_+y] for x in range(-2, 3) for y in range(-2, 3)]:
                imgBools[x_][y_] = True


    def bfs(x, y):
        pipe = deque([(x, y)])
        imgBools[x][y] = False
        g = [(x, y)]

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while pipe:
            cx, cy = pipe.popleft()
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < len(imgBools) and 0 <= ny < len(imgBools[0]) and imgBools[nx][ny]:
                    imgBools[nx][ny] = False
                    pipe.append((nx, ny))
                    g.append((nx, ny))

        return g

    groups = []

    for i in range(len(imgBools)):
        for j in range(len(imgBools[0])):
            if imgBools[i][j]:
                g = bfs(i, j)
                if g:
                    groups.append(g)


    for g in groups:
        if len(g) > 10:

            maxX, minX, maxY, minY = 0,999999,0,999999

            vertexes = [0,0,0,0]
            vertexesDis = [999999,999999,999999,999999]

            def dist(a,b):
                return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**(0.5)

            for px in g:
                if px[0] < minX:
                    minX = px[0]
                if px[1] < minY:
                    minY = px[1]
                if px[0] > maxX:
                    maxX = px[0]
                if px[1] > maxY:
                    maxY = px[1]

            for px in g:
                if dist(px, (maxX, maxY)) < vertexesDis[0]:
                    vertexesDis[0] = dist(px, (maxX, maxY))
                    vertexes[0] = px
                if dist(px, (minX, maxY)) < vertexesDis[1]:
                    vertexesDis[1] = dist(px, (minX, maxY))
                    vertexes[1] = px
                if dist(px, (minX, minY)) < vertexesDis[2]:
                    vertexesDis[2] = dist(px, (minX, minY))
                    vertexes[2] = px
                if dist(px, (maxX, minY)) < vertexesDis[3]:
                    vertexesDis[3] = dist(px, (maxX, minY))
                    vertexes[3] = px


            pygame.draw.rect(screen, (0, 0, 100),
                             [width - maxX, minY, maxX-minX+1, maxY-minY+1], 1)


            for v in vertexes:
                pygame.draw.circle(screen, (0, 255, 0),
                                   [width-v[0], v[1]], 4, 0)

            sk = checkLetter(g, imgMatrix, maxX, minX, maxY, minY)
            v = serafinSimone(sk)
            print(v)

    imgSurface = pygame.surfarray.make_surface(imgMatrix)

    imgSurface = pygame.transform.rotate(imgSurface, -90)

    imgSurface = pygame.transform.scale(imgSurface, (width, height))

    screen.blit(imgSurface, (0, 0))


    pygame.display.flip()

    running = True
    while running:

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

imagePath = 'immagine4.png'
showImageAndMatrixInPygame(imagePath)