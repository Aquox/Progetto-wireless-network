import csv
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

stop = False




image_folder = 'images'
video_name = 'video.avi'

images = [img for img in os.listdir("./../Output/frames") if img.endswith(".png")]
print(images)
frame = cv2.imread(os.path.join("./../Output/frames", images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter("./../Output/video.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join("./../Output/frames", image)))

cv2.destroyAllWindows()
video.release()

plt.plot(1)
plt.show()


Xpixel = 3840
Ypixel = 1920
MatrixPixels = np.zeros((Xpixel, Ypixel),dtype = 'bool')

#fovX dipende dal pitch di partenza
fovX = 100
fovY = 80

listOfFile = glob.glob("./*.csv")
RectangularData = []
for i in range (len(listOfFile)):
    RectangularData.append(np.zeros((1800, 8)))


verifica = (np.zeros((50)))
verifica2 = (np.zeros((50)))


# leggo tutti i file e per ogni frame vado a calcolare il quadrato di fov utente 2D
for i in range(len(listOfFile)):
    with open(listOfFile[i], newline="", encoding="ISO-8859-1") as filecsv:
        lettore = csv.reader(filecsv, delimiter=",")
        header = next (lettore)
        l = 0
        while True:
            try:
                riga = next(lettore)
            except:
                break
            # rad = (gradi * pi)/180
            rad = float(riga[6])*np.pi/180
            if(l==185):
                verifica[i]= float(riga[4])
                verifica2[i]= float(riga[5])
            RectangularData[i][l][0] = fovX/2 * np.cos(rad) + fovY/2 * np.sin(rad) + float(riga[4])
            RectangularData[i][l][1] = fovX/2 * np.sin(rad) - fovY/2 * np.cos(rad) + float(riga[5])
            RectangularData[i][l][2] = fovX/2 * np.cos(rad) - fovY/2 * np.sin(rad) + float(riga[4])
            RectangularData[i][l][3] = fovX/2 * np.sin(rad) + fovY/2 * np.cos(rad) + float(riga[5])
            RectangularData[i][l][4] = - fovX/2 * np.cos(rad) - fovY/2 * np.sin(rad) + float(riga[4])
            RectangularData[i][l][5] = - fovX/2 * np.sin(rad) + fovY/2 * np.cos(rad) + float(riga[5])
            RectangularData[i][l][6] = - fovX/2 * np.cos(rad) + fovY/2 * np.sin(rad) + float(riga[4])
            RectangularData[i][l][7] = - fovX/2 * np.sin(rad) - fovY/2 * np.cos(rad) + float(riga[5])
            l = l + 1

coordToPixelX = 360/Xpixel
coordToPixelY = 180/Ypixel

#plt.plot(verifica, verifica2, 'ro')
#plt.axis([-180, 180, -90, 90])
#plt.show()

#x1Rounded = x2Rounded = max((int(RectangularData[0][0][0]+180)/coordToPixelX),(int((RectangularData[0][0][2]+180)/coordToPixelX)))
#y1Rounded = y4Rounded = min((int((RectangularData[0][0][1]+90)/coordToPixelY)),int((RectangularData[0][0][7]+90)/coordToPixelY))
#x3Rounded = x4Rounded = min((int((RectangularData[0][0][4]+180)/coordToPixelX)),int((RectangularData[0][0][6]+180)/coordToPixelX))
#y2Rounded = y3Rounded = max((int((RectangularData[0][0][3]+90)/coordToPixelY)),int((RectangularData[0][0][5]+90)/coordToPixelY))


pixelFrames = []


for j in range(1800):
    for i in range (50):
        ## if valori non sballati
        x1Rounded = x2Rounded = max((int((RectangularData[i][j][0] + 180) / coordToPixelX)),
                                    (int((RectangularData[i][j][2] + 180) / coordToPixelX)))
        y1Rounded = y4Rounded = min((int((RectangularData[i][j][1] + 90) / coordToPixelY)),
                                    int((RectangularData[i][j][7] + 90) / coordToPixelY))
        x3Rounded = x4Rounded = min((int((RectangularData[i][j][4] + 180) / coordToPixelX)),
                                    int((RectangularData[i][j][6] + 180) / coordToPixelX))
        y2Rounded = y3Rounded = max((int((RectangularData[i][j][3] + 90) / coordToPixelY)),
                                    int((RectangularData[i][j][5] + 90) / coordToPixelY))
        if y1Rounded < int(20/coordToPixelY):
            if x3Rounded < 0 or x1Rounded > Xpixel-1:
                if x3Rounded < 0:
                    if y1Rounded < 0:
                        MatrixPixels[0:x1Rounded, 0:y2Rounded] = np.ones((x1Rounded, y2Rounded), dtype = 'bool')
                        MatrixPixels[Xpixel-x1Rounded:Xpixel, 0:y2Rounded] = np.ones((x1Rounded,
                                                                                  y2Rounded), dtype='bool')
                        MatrixPixels[0:Xpixel, 0:int(22 / coordToPixelY)] = np.ones((Xpixel, int(22 / coordToPixelY)),
                                                                                    dtype = 'bool')
                    else:
                        MatrixPixels[0:x1Rounded, y1Rounded:y2Rounded] = np.ones((x1Rounded,
                                                                                y2Rounded-y1Rounded), dtype = 'bool')
                        MatrixPixels[Xpixel - x1Rounded:Xpixel, y1Rounded:y2Rounded] = np.ones((x1Rounded,
                                                                                y2Rounded - y1Rounded), dtype='bool')
                        MatrixPixels[0:Xpixel, 0:int(22/coordToPixelY)] = np.ones((Xpixel, int(22/coordToPixelY)),
                                                                                    dtype = 'bool')
                if x1Rounded > Xpixel-1:
                    if y1Rounded < 0:
                        MatrixPixels[x3Rounded:Xpixel, 0:y2Rounded] = np.ones((Xpixel - x3Rounded, y2Rounded),
                                                                              dtype = 'bool')
                        MatrixPixels[0:x1Rounded-Xpixel, 0:y2Rounded] = np.ones((x1Rounded - Xpixel,
                                                                                  y2Rounded), dtype='bool')
                        MatrixPixels[0:Xpixel, 0:int(22 / coordToPixelY)] = np.ones((Xpixel, int(22 / coordToPixelY)),
                                                                                    dtype = 'bool')
                    else:
                        MatrixPixels[x3Rounded:Xpixel, y1Rounded:y2Rounded] = np.ones((Xpixel - x3Rounded,
                                                                                y2Rounded-y1Rounded), dtype = 'bool')
                        MatrixPixels[0:x1Rounded-Xpixel, y1Rounded:y2Rounded] = np.ones((x1Rounded - Xpixel,
                                                                                       y2Rounded - y1Rounded),
                                                                                      dtype='bool')
                        MatrixPixels[0:Xpixel, 0:int(22/coordToPixelY)] = np.ones((Xpixel, int(22/coordToPixelY)),
                                                                                    dtype = 'bool')
            else:
                if y1Rounded < 0:
                    MatrixPixels[x3Rounded:x1Rounded, 0:y2Rounded] = np.ones((x1Rounded - x3Rounded,
                                                                                y2Rounded), dtype = 'bool')
                    MatrixPixels[0:Xpixel, 0:int(22 / coordToPixelY)] = np.ones((Xpixel, int(22 / coordToPixelY)),
                                                                                dtype = 'bool')
                else:
                    MatrixPixels[x3Rounded:x1Rounded, y1Rounded:y2Rounded] = np.ones((x1Rounded-x3Rounded,
                                                                                  y2Rounded-y1Rounded), dtype = 'bool')
                    MatrixPixels[0:Xpixel, 0:int(22/coordToPixelY)] = np.ones((Xpixel, int(22/coordToPixelY)),
                                                                                dtype = 'bool')
        if y2Rounded > int(160/coordToPixelY):
            if x3Rounded < 0 or x1Rounded > Xpixel - 1:
                if x3Rounded < 0:
                    if y2Rounded > Ypixel-1:
                        MatrixPixels[0:x1Rounded, y1Rounded:Ypixel-1] = np.ones((x1Rounded, Ypixel-1-y1Rounded),
                                                                                dtype = 'bool')
                        MatrixPixels[Xpixel - x1Rounded:Xpixel, y1Rounded:Ypixel-1] = np.ones((x1Rounded,
                                                                                Ypixel - 1 - y1Rounded), dtype='bool')
                        MatrixPixels[0:Xpixel, int(Ypixel - 22 / coordToPixelY):Ypixel-1] = np.ones((Xpixel,
                                                                            int(22 / coordToPixelY)), dtype = 'bool')
                    else:
                        MatrixPixels[0:x1Rounded, y1Rounded:y2Rounded] = np.ones((x1Rounded, y2Rounded-y1Rounded),
                                                                                 dtype = 'bool')
                        MatrixPixels[Xpixel - x1Rounded:Xpixel, y1Rounded:y2Rounded] = np.ones((x1Rounded,
                                                                                                y2Rounded - y1Rounded),
                                                                                               dtype='bool')
                        MatrixPixels[0:Xpixel, int(Ypixel - 22 / coordToPixelY):Ypixel-1] = np.ones((Xpixel,
                                                                            int(22 / coordToPixelY)), dtype = 'bool')
                if x1Rounded > Xpixel - 1:
                    if y2Rounded > Ypixel-1:
                        MatrixPixels[x3Rounded:Xpixel, y1Rounded:Ypixel-1] = np.ones((Xpixel - x3Rounded,
                                                                                Ypixel-1-y1Rounded), dtype = 'bool')
                        MatrixPixels[0:x1Rounded - Xpixel, y1Rounded:Ypixel-1] = np.ones((x1Rounded - Xpixel,
                                                                                Ypixel - 1 - y1Rounded), dtype='bool')
                        MatrixPixels[0:Xpixel, int(Ypixel - 22 / coordToPixelY):Ypixel-1] = np.ones((Xpixel,
                                                                             int(22 / coordToPixelY)), dtype = 'bool')
                    else:
                        MatrixPixels[x3Rounded:Xpixel, y1Rounded:y2Rounded] = np.ones((Xpixel - x3Rounded,
                                                                                 y2Rounded-y1Rounded), dtype = 'bool')
                        MatrixPixels[0:x1Rounded - Xpixel, y1Rounded:y2Rounded] = np.ones((x1Rounded - Xpixel,
                                                                                           y2Rounded - y1Rounded),
                                                                                          dtype='bool')
                        MatrixPixels[0:Xpixel, int(Ypixel - 22 / coordToPixelY):Ypixel-1] = np.ones((Xpixel,
                                                                            int(22 / coordToPixelY)), dtype = 'bool')
            else:
                if y2Rounded > Ypixel-1:
                    MatrixPixels[x3Rounded:x1Rounded, y1Rounded:Ypixel-1] = np.ones((x1Rounded - x3Rounded,
                                                                                   Ypixel-1-y1Rounded), dtype = 'bool')
                    MatrixPixels[0:Xpixel, int(Ypixel - 22 / coordToPixelY):Ypixel-1] = np.ones((Xpixel,
                                                                              int(22 / coordToPixelY)), dtype = 'bool')
                else:
                    MatrixPixels[x3Rounded:x1Rounded, y1Rounded:y2Rounded] = np.ones((x1Rounded-x3Rounded,
                                                                                  y2Rounded-y1Rounded), dtype = 'bool')
                    MatrixPixels[0:Xpixel, int(Ypixel - 22 / coordToPixelY):Ypixel-1] = np.ones((Xpixel,
                                                                              int(22 / coordToPixelY)), dtype = 'bool')

        if y1Rounded >= int(20/coordToPixelY) and y2Rounded <= int(160/coordToPixelY):
            if x3Rounded < 0 or x1Rounded > Xpixel - 1:
                if x3Rounded < 0:
                    MatrixPixels[0:x1Rounded, y1Rounded:y2Rounded] = np.ones((x1Rounded, y2Rounded - y1Rounded),
                                                                             dtype='bool')
                    MatrixPixels[Xpixel - x1Rounded:Xpixel, y1Rounded:y2Rounded] = np.ones((x1Rounded,
                                                                                            y2Rounded - y1Rounded),
                                                                                           dtype='bool')
                if x1Rounded > Xpixel - 1:
                    MatrixPixels[x3Rounded:Xpixel, y1Rounded:y2Rounded] = np.ones((Xpixel - x3Rounded,
                                                                                   y2Rounded - y1Rounded), dtype='bool')
                    MatrixPixels[0:x1Rounded - Xpixel, y1Rounded:y2Rounded] = np.ones((x1Rounded - Xpixel,
                                                                                       y2Rounded - y1Rounded),
                                                                                      dtype='bool')
            else:
                MatrixPixels[x3Rounded:x1Rounded, y1Rounded:y2Rounded] = np.ones((x1Rounded - x3Rounded,
                                                                                  y2Rounded - y1Rounded), dtype='bool')

    #print (np.count_nonzero(MatrixPixels == 1))
    pixelFrames.append(np.count_nonzero(MatrixPixels == 1))
    fig, ax = plt.subplots()
    mat = ax.imshow(MatrixPixels.T, cmap='GnBu', interpolation='nearest')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(21, 9)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    fig.savefig("./../Output/Frames/frame"+str(j)+".png", dpi=50, bbox_inches='tight')
    plt.close('all')
    MatrixPixels = np.zeros((Xpixel, Ypixel), dtype='bool' )


#  creo una cartella dove salvare i dati in output
if not os.path.exists("./../Output"):
    os.makedirs("./../Output")

Ygraphpercentage = []
Ygraphmoregranularity = []

# disegno un grafico  x: #frame  y: #percentuale pixel visti
for i in range (1800):
    Ygraphpercentage.append(pixelFrames[i]/(Xpixel*Ypixel))

plt.plot(Ygraphpercentage)
fig = plt.gcf()
fig.set_size_inches(21, 9)
fig.savefig("./../Output/Graficoprova2414.png", dpi=100)
plt.show()

# disegno un grafico a granularità maggiore raggruppando i frame di un secondo (30)
# x: #secondi y: #percentuale pixel visti
for i in range(60):
    Ygraphmoregranularity.append(sum(Ygraphpercentage[i*30:i*30+30])/30)

plt.plot(Ygraphmoregranularity)
fig = plt.gcf()
fig.set_size_inches(21, 9)
fig.savefig("./../Output/awfawfa.png", dpi=100)
plt.show()



