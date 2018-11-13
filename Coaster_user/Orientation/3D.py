import csv
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import math


stop = False


Xpixel = 3600
Ypixel = 2048
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
            if (j == 185):
                print("ciao")
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
    MatrixPixels = np.zeros((Xpixel, Ypixel), dtype='bool')

print(pixelFrames)
pixelFramesPercentile =  []
for i in range (1800):
    pixelFramesPercentile.append(pixelFrames[i]/(Xpixel*Ypixel))

print (sum(pixelFramesPercentile)/len(pixelFramesPercentile))

plt.plot(pixelFramesPercentile)
plt.show()




# A[3:5, 3:5] = B


# y = []
#
# #  leggo i vari file e salvo tutti i tiles riga per riga in una lista di set per non avere doppioni.
# #  Valido per qualsiasi tipo di file con le caratteristiche appropiate
# for i in range(len(listOfFile)):
#     with open(listOfFile[i], newline="", encoding="ISO-8859-1") as filecsv:
#         lettore = csv.reader(filecsv,delimiter=",")
#         header = next(lettore)
#         l=0
#         while True:
#             if not stop:
#                 listofset.append(set())
#             try:
#                 riga = next(lettore)
#             except:
#                 break
#             for m in range(1,(len(riga))):
#                 listofset[l].add(riga[m])
#             l=l+1
#         stop = True
#
# #  creo una cartella dove salvare i dati in output
# if not os.path.exists("allData"):
#     os.makedirs("allData")
#
#
#
# #disegno un grafico  x: #frame  y: #tiles visti , con errore std. dev.
# for i in range(1800):
#     y.append(len(listofset[i]))
# mediay = sum (y) / len(y)
# sumfordev = 0
# for i in range(1800):
#     temp = (y[i] - mediay)
#     sumfordev = sumfordev + temp*temp
# stdev = math.sqrt(sumfordev/1800)
# print(stdev)
# plt.plot(y)
# plt.axis([0, 1800, 0, 225])
# plt.xticks([200*k for k in range (10)])
# x = list(range(1,1801))
# err = [stdev] * 1800
# print(x)
# plt.errorbar(x, y, yerr=(err,err), ecolor="red")
# plt.title("Frame e tiles")
# plt.xlabel("# Frame")
# plt.ylabel("# Tiles visualizzati nel frame")
# plt.savefig("./allData/example.png")
# plt.show()
#
#
#
# #  creo il file con numero di frame, percentuale di visione del frame e i tiles visti, l'ultima riga è il totale di
# #  tiles visti nell'insieme del video.
# #  La percentuale non è valida se i tiles per frame sono diversi da 200 in totale.
#
# with open("./allData/all_tile_per_frame.csv", 'w', newline='') as filecsv:
#     wr = csv.writer(filecsv)
#     wr.writerow(["frame","% tiles visti","tiles"])
#     totale = 0
#     for x in range(1, 1801):
#         setToList = list(listofset[x-1])
#         setToList.insert(0, x)
#         totale = totale + len(listofset[x-1])
#         setToList.insert(1, len(listofset[x-1])/200)
#         wr.writerow(setToList)
#     wr.writerow(["total that can be " + str(totale/(1800*200))])

