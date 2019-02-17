import csv
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gc


sizes = [153,160,114,117,138,126,50,60,149,51]
folders = ["Coaster","Coaster2","Diving","Drive","Game","Landscape","Pacman","Panel","Ride","Sport"] #

for f in range(len(folders)):
    print(f)
    # Dati di input
    Xpixel = 3840
    Ypixel = 1920
    fovX = 100
    fovY = 100

    # images = [img for img in os.listdir("./../Output/" + folders[f] + "/Frame") if img.endswith(".png")]
    # images2 = [img for img in os.listdir("./../Output/" + folders[f] + "/Fotogrammi") if img.endswith(".png")]
    # frame = cv2.imread(os.path.join("./../Output/" + folders[f] + "/Frame", images[0]), cv2.IMREAD_UNCHANGED)
    # frame2 = cv2.imread(os.path.join("./../Output/" + folders[f] + "/Fotogrammi", images2[0]), cv2.IMREAD_UNCHANGED)
    # # b_channel, g_channel, r_channel = cv2.split(frame2)
    # # alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # creating a dummy alpha channel image.
    # print (frame[120,450,0])
    # print(frame[120, 450, 1])
    # print(frame[120, 450, 2])
    # print(frame[120, 450, 3])
    #
    # print(frame[0, 0, 0])
    # print(frame[0, 0, 1])
    # print(frame[0, 0, 2])
    # print(frame[0, 0, 3])
    #
    # height, width, layers = frame.shape
    # video = cv2.VideoWriter("./../Output/" + folders[f] + "/frame.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 30,
    #                         (width, height))
    # for i in range(len(images)):
    #     src1 = cv2.imread(os.path.join("./../Output/" + folders[f] + "/Frame", images[i]),cv2.IMREAD_UNCHANGED)
    #     src2 = cv2.imread(os.path.join("./../Output/" + folders[f] + "/Fotogrammi", images2[i]),cv2.IMREAD_UNCHANGED)
    #     # b_channel, g_channel, r_channel = cv2.split(src2)
    #     # src2_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    #     # print(src1)
    #     # print(src2_BGRA)
    #     dst = cv2.addWeighted(src1, 1, src2_BGRA, 1, 0)
    #     video.write(dst)
    # cv2.destroyAllWindows()
    # video.release()
    #
    # plt.plot(1)
    # plt.show()


    # Variabili e matrici di supporto
    MatrixPixels = np.zeros((Xpixel, Ypixel),dtype = 'bool')
    # MatrixPixelsOtherside = np.zeros((Xpixel, Ypixel),dtype = 'bool')
    MatrixRealFrame = np.zeros((Xpixel, Ypixel),dtype='bool')
    MatrixProva = np.zeros((Xpixel,Ypixel))
    fovXpixel = int(Xpixel/360*fovX)
    fovYpixel = int(Ypixel/180*fovY)


    listOfFile = glob.glob("./../Dataset/"+folders[f]+"/Orientation/*.csv")
    CoordinateData = []
    for i in range (len(listOfFile)):
        CoordinateData.append(np.zeros((1800, 2)))

    for i in range(int(Ypixel/2 - fovXpixel/2)):
        MatrixPixels[int(Xpixel/4 + (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2) : int(3*Xpixel/4 - (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2) , Ypixel-1-i-int(fovXpixel/2):Ypixel-i-int(fovXpixel/2)] = \
            np.ones((int(3*Xpixel/4 - (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2)-int(Xpixel/4 + (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2), 1), dtype='bool')
    for i in range(int(Ypixel/2 - fovXpixel/2)):
        MatrixPixels[int(Xpixel/4 + (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2) : int(3*Xpixel/4 - (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2) , i+int(fovXpixel/2):i+int(fovXpixel/2)+1] = \
            np.ones((int(3*Xpixel/4 - (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2)-int(Xpixel/4 + (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2), 1), dtype='bool')

    MatrixPixels[int(Xpixel/4):int(3*Xpixel/4) , int(Ypixel-fovXpixel/2):Ypixel] = np.ones((int(Xpixel/2),Ypixel - int(Ypixel-fovXpixel/2)),dtype='bool')
    MatrixPixels[int(Xpixel/4):int(3*Xpixel/4) , 0:int(fovXpixel/2)] = np.ones((int(Xpixel/2),int(fovXpixel/2)),dtype='bool')



    fig, ax = plt.subplots()
    mat = ax.imshow(np.rot90(MatrixPixels), cmap='GnBu', interpolation='nearest')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(21, 9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    fig.savefig("./../Output/"+folders[f]+"/frameadadwdadasdawdawdada.png", dpi=250, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close('all')


    plt.plot(1)
    plt.show()

    # for i in range(int(Ypixel/2 - fovXpixel/2)):
    #     MatrixPixelsOtherside[0 : int(Xpixel/4 - (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2) , Ypixel-1-i-int(fovXpixel/2):Ypixel-i-int(fovXpixel/2)] = \
    #         np.ones((int(Xpixel/4 - (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2), 1), dtype='bool')
    #     MatrixPixelsOtherside[int(3*Xpixel / 4 + (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2): int(Xpixel), Ypixel - 1 - i - int(fovXpixel / 2):Ypixel - i - int(fovXpixel / 2)] = \
    #         np.ones((int( Xpixel  - (i * (Ypixel - fovXpixel) / (Ypixel / 2 - fovXpixel / 2)) / 2) - int(3*Xpixel / 4 ), 1), dtype='bool')
    #
    # for i in range(int(Ypixel/2 - fovXpixel/2)):
    #     MatrixPixelsOtherside[ 0 : int(Xpixel/4 - (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2) , i+int(fovXpixel/2):i+int(fovXpixel/2)+1] = \
    #         np.ones((int(Xpixel/4 - (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2), 1), dtype='bool')
    #     MatrixPixelsOtherside[int(3*Xpixel / 4 + (i * (Ypixel - fovXpixel) / (Ypixel / 2 - fovXpixel / 2)) / 2): int( Xpixel ),i + int(fovXpixel / 2):i + int(fovXpixel / 2) + 1] = \
    #         np.ones((int(Xpixel) - int( 3*Xpixel / 4 + (i * (Ypixel - fovXpixel) / (Ypixel / 2 - fovXpixel / 2)) / 2), 1), dtype='bool')
    # MatrixPixelsOtherside[0:int(Xpixel/4) , int(Ypixel-fovXpixel/2):Ypixel] = np.ones((int(Xpixel/4),Ypixel - int(Ypixel-fovXpixel/2)),dtype='bool')
    # MatrixPixelsOtherside[int(3*Xpixel/4):Xpixel , int(Ypixel-fovXpixel/2):Ypixel] = np.ones((int(Xpixel/4),Ypixel - int(Ypixel-fovXpixel/2)),dtype='bool')
    # MatrixPixelsOtherside[0:int(Xpixel/4) , 0:int(fovXpixel/2)] = np.ones((int(Xpixel/4),int(fovXpixel/2)),dtype='bool')
    # MatrixPixelsOtherside[int(3*Xpixel/4):Xpixel , 0:int(fovXpixel/2)] = np.ones((int(Xpixel/4),int(fovXpixel/2)),dtype='bool')


    # leggo tutti i file e ricavo le coordinate della visualizzazione dei vari utenti
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
                CoordinateData[i][l][0] = riga[4]
                CoordinateData[i][l][1] = riga[5]
                l = l + 1


    # Variabili di supporto al 3D
    coordToPixelX = 360/Xpixel
    coordToPixelY = 180/Ypixel
    pixelFrames = []

    # Variabile di supporto per il baricentro x,y [0,1] ; le distanze da esso [2,52]
    coordBaricentro = np.zeros((1800,52))

    coordXold = []
    coordYold = []
    distancesTraveled = []
    for l in range(50):
        coordXold.append(0)
        coordYold.append(0)
        distancesTraveled.append(0)

    # Ciclo per ogni frame per ricavare le varie immagini e i vari dati
    for j in range(1800):

        coordX = []
        coordY = []

        # Ci sono 50 utenti quindi analizzo i dati per ogni utente
        for i in range (50):

            xtoPixel = int((CoordinateData[i][j][0]+180)/coordToPixelX)
            ytoPixel = int((CoordinateData[i][j][1]+90)/coordToPixelY)
            coordX.append(xtoPixel)
            # y invertito su asse centrale (ypixel/2)
            coordY.append((int(Ypixel / 2) - ytoPixel) * 2 + ytoPixel)

            if(j != 0):
                distancesTraveled[i] = distancesTraveled[i] + np.sqrt(min((coordXold[i]-coordX[i])**2,(coordXold[i]-coordX[i]+Xpixel)**2,(coordXold[i]-coordX[i]-Xpixel)**2) + min((coordYold[i]-coordY[i])**2,(coordYold[i]-coordY[i]+Ypixel)**2,(coordYold[i]-coordY[i]-Ypixel)**2))

            coordXold[i] = xtoPixel
            coordYold[i] = (int(Ypixel / 2) - ytoPixel) * 2 + ytoPixel

            # disegno sulla matrice del frame i pixel visualizzati dall'utente
            if int(ytoPixel + fovYpixel/2) > Ypixel:
                if int(xtoPixel + Xpixel/4) > Xpixel:
                    # apposto
                    MatrixRealFrame[0:int(xtoPixel + Xpixel / 4 - Xpixel), ytoPixel - int(fovYpixel / 2): Ypixel] = np.sum(
                        [MatrixRealFrame[0:int(xtoPixel + Xpixel / 4 - Xpixel), ytoPixel - int(fovYpixel / 2): Ypixel], MatrixPixels[int(3 * Xpixel / 4 - (xtoPixel + Xpixel / 4 - Xpixel)):int(3 * Xpixel / 4), ytoPixel - int(fovYpixel / 2): Ypixel]]
                        , axis=0, dtype='bool')
                    MatrixRealFrame[int(xtoPixel - Xpixel / 4): Xpixel, ytoPixel - int(fovYpixel / 2): Ypixel] = np.sum(
                        [MatrixRealFrame[int(xtoPixel - Xpixel / 4): Xpixel, ytoPixel - int(fovYpixel / 2): Ypixel], MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4) - (int(xtoPixel + Xpixel / 4) - Xpixel), ytoPixel - int(fovYpixel / 2): Ypixel]]
                        , axis=0, dtype='bool')
                    MatrixRealFrame[int(xtoPixel + Xpixel / 4 - Xpixel):int(xtoPixel - Xpixel / 4), int(Ypixel-(ytoPixel + fovYpixel/2 - Ypixel)):Ypixel] = np.sum(
                        [MatrixRealFrame[int(xtoPixel + Xpixel / 4 - Xpixel):int(xtoPixel - Xpixel / 4), int(Ypixel-(ytoPixel + fovYpixel/2 - Ypixel)):Ypixel], MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4), int(Ypixel-(ytoPixel + fovYpixel/2 - Ypixel)):Ypixel]]
                        , axis=0, dtype='bool')
                if int(xtoPixel - Xpixel/4) < 0 :
                    # da vedere
                    MatrixRealFrame[0:int(xtoPixel + Xpixel / 4), ytoPixel - int(fovYpixel / 2): Ypixel] = np.sum(
                        [MatrixRealFrame[0:int(xtoPixel + Xpixel / 4), ytoPixel - int(fovYpixel / 2): Ypixel], MatrixPixels[int(Xpixel / 4 - (xtoPixel - Xpixel / 4)):int(3 * Xpixel / 4), ytoPixel - int(fovYpixel / 2): Ypixel]]
                        , axis=0, dtype='bool')
                    MatrixRealFrame[int(Xpixel + (xtoPixel - Xpixel / 4)): Xpixel, ytoPixel - int(fovYpixel / 2): Ypixel] = np.sum(
                        [MatrixRealFrame[int(Xpixel + (xtoPixel - Xpixel / 4)): Xpixel, ytoPixel - int(fovYpixel / 2): Ypixel], MatrixPixels[int(Xpixel / 4):int(Xpixel / 4 - (xtoPixel - Xpixel / 4)), ytoPixel - int(fovYpixel / 2): Ypixel]]
                        , axis=0, dtype='bool')
                    MatrixRealFrame[int(xtoPixel + Xpixel / 4):int(Xpixel + (xtoPixel - Xpixel / 4)), int(Ypixel-(ytoPixel + fovYpixel/2 - Ypixel)):Ypixel] = np.sum(
                        [MatrixRealFrame[int(xtoPixel + Xpixel / 4):int(Xpixel + (xtoPixel - Xpixel / 4)), int(Ypixel-(ytoPixel + fovYpixel/2 - Ypixel)):Ypixel], MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4), int(Ypixel-(ytoPixel + fovYpixel/2 - Ypixel)):Ypixel]]
                        , axis=0, dtype='bool')
                if int(xtoPixel + Xpixel/4) <= Xpixel  and  int(xtoPixel - Xpixel/4) >= 0 :
                    # apposto
                    MatrixRealFrame[int(xtoPixel - Xpixel / 4):int(xtoPixel + Xpixel / 4), ytoPixel - int(fovYpixel / 2): Ypixel] = np.sum(
                        [MatrixRealFrame[int(xtoPixel - Xpixel / 4):int(xtoPixel + Xpixel / 4), ytoPixel - int(fovYpixel / 2): Ypixel], MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4), ytoPixel - int(fovYpixel / 2): Ypixel]]
                        , axis=0, dtype='bool')
                    MatrixRealFrame[0:int(xtoPixel - Xpixel / 4), int(Ypixel-(ytoPixel + fovYpixel/2 - Ypixel)):Ypixel] = np.sum(
                        [MatrixRealFrame[0:int(xtoPixel - Xpixel / 4), int(Ypixel-(ytoPixel + fovYpixel/2 - Ypixel)):Ypixel], MatrixPixels[int(3 * Xpixel / 4 - (xtoPixel - Xpixel / 4)):int(3 * Xpixel / 4), int(Ypixel-(ytoPixel + fovYpixel/2 - Ypixel)):Ypixel]]
                        , axis=0, dtype='bool')
                    MatrixRealFrame[int(xtoPixel + Xpixel / 4): Xpixel, int(Ypixel-(ytoPixel + fovYpixel/2 - Ypixel)):Ypixel] = np.sum(
                        [MatrixRealFrame[int(xtoPixel + Xpixel / 4): Xpixel, int(Ypixel-(ytoPixel + fovYpixel/2 - Ypixel)):Ypixel], MatrixPixels[int(Xpixel / 4):int(Xpixel / 4 + (Xpixel - (xtoPixel + Xpixel / 4))), int(Ypixel-(ytoPixel + fovYpixel/2 - Ypixel)):Ypixel]]
                        , axis=0, dtype='bool')
            if int(ytoPixel - fovYpixel/2) < 0 :
                if int(xtoPixel + Xpixel/4) > Xpixel:
                    # da vedere
                    MatrixRealFrame[0:int(xtoPixel + Xpixel / 4 - Xpixel), 0 : ytoPixel + int(fovYpixel / 2)] = np.sum(
                        [MatrixRealFrame[0:int(xtoPixel + Xpixel / 4 - Xpixel), 0 : ytoPixel + int(fovYpixel / 2)], MatrixPixels[int(3 * Xpixel / 4 - (xtoPixel + Xpixel / 4 - Xpixel)):int(3 * Xpixel / 4), 0 : ytoPixel + int(fovYpixel / 2)]]
                        , axis=0, dtype='bool')
                    MatrixRealFrame[int(xtoPixel - Xpixel / 4): Xpixel, 0 : ytoPixel + int(fovYpixel / 2)] = np.sum(
                        [MatrixRealFrame[int(xtoPixel - Xpixel / 4): Xpixel, 0 : ytoPixel + int(fovYpixel / 2)], MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4) - (int(xtoPixel + Xpixel / 4) - Xpixel), 0 : ytoPixel + int(fovYpixel / 2)]]
                        , axis=0, dtype='bool')
                    MatrixRealFrame[int(xtoPixel + Xpixel / 4 - Xpixel):int(xtoPixel - Xpixel / 4), 0:abs(ytoPixel - int(fovYpixel / 2))] = np.sum(
                        [MatrixRealFrame[int(xtoPixel + Xpixel / 4 - Xpixel):int(xtoPixel - Xpixel / 4), 0:abs(ytoPixel - int(fovYpixel / 2))], MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4), 0:abs(ytoPixel - int(fovYpixel / 2))]]
                        , axis=0, dtype='bool')
                if int(xtoPixel - Xpixel/4) < 0 :
                    # apposto
                    MatrixRealFrame[0:int(xtoPixel + Xpixel / 4), 0 : ytoPixel + int(fovYpixel / 2)] = np.sum(
                        [MatrixRealFrame[0:int(xtoPixel + Xpixel / 4),0 : ytoPixel + int(fovYpixel / 2)], MatrixPixels[int(Xpixel / 4 - (xtoPixel - Xpixel / 4)):int(3 * Xpixel / 4), 0 : ytoPixel + int(fovYpixel / 2)]]
                        , axis=0, dtype='bool')
                    MatrixRealFrame[int(Xpixel + (xtoPixel - Xpixel / 4)): Xpixel, 0 : ytoPixel + int(fovYpixel / 2)] = np.sum(
                        [MatrixRealFrame[int(Xpixel + (xtoPixel - Xpixel / 4)): Xpixel, 0 : ytoPixel + int(fovYpixel / 2)], MatrixPixels[int(Xpixel / 4):int(Xpixel / 4 - (xtoPixel - Xpixel / 4)), 0 : ytoPixel + int(fovYpixel / 2)]]
                        , axis=0, dtype='bool')
                    MatrixRealFrame[int(xtoPixel + Xpixel / 4):int(Xpixel + (xtoPixel - Xpixel / 4)), 0:abs(ytoPixel - int(fovYpixel / 2))] = np.sum(
                        [MatrixRealFrame[int(xtoPixel + Xpixel / 4):int(Xpixel + (xtoPixel - Xpixel / 4)), 0:abs(ytoPixel - int(fovYpixel / 2))], MatrixPixels[int(Xpixel / 4 ):int(3 * Xpixel / 4), 0:abs(ytoPixel - int(fovYpixel / 2))]]
                        , axis=0, dtype='bool')
                if int(xtoPixel + Xpixel/4) <= Xpixel  and  int(xtoPixel - Xpixel/4) >= 0 :
                    # apposto
                    MatrixRealFrame[int(xtoPixel - Xpixel / 4):int(xtoPixel + Xpixel / 4), 0: ytoPixel + int(fovYpixel / 2)] = np.sum(
                        [MatrixRealFrame[int(xtoPixel - Xpixel / 4):int(xtoPixel + Xpixel / 4), 0: ytoPixel + int(fovYpixel / 2)], MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4), 0: ytoPixel + int(fovYpixel / 2)]]
                        , axis=0, dtype='bool')
                    MatrixRealFrame[0:int(xtoPixel - Xpixel / 4),0:abs(ytoPixel - int(fovYpixel / 2)) ] = np.sum(
                        [MatrixRealFrame[0:int(xtoPixel - Xpixel / 4), 0:abs(ytoPixel - int(fovYpixel / 2))], MatrixPixels[int(3*Xpixel/4 - (xtoPixel - Xpixel / 4) ):int(3*Xpixel/4), 0:abs(ytoPixel - int(fovYpixel / 2))]]
                        , axis=0,dtype='bool')
                    MatrixRealFrame[int(xtoPixel + Xpixel / 4): Xpixel, 0:abs(ytoPixel - int(fovYpixel / 2))] = np.sum(
                        [MatrixRealFrame[int(xtoPixel + Xpixel / 4): Xpixel, 0:abs(ytoPixel - int(fovYpixel / 2))], MatrixPixels[int(Xpixel / 4) :int( Xpixel / 4 + (Xpixel -(xtoPixel + Xpixel / 4))), 0:abs(ytoPixel - int(fovYpixel / 2))]]
                        , axis=0, dtype='bool')
            if int(ytoPixel + fovYpixel/2) <= Ypixel  and  int(ytoPixel - fovYpixel/2) >= 0 :
                if int(xtoPixel + Xpixel/4) > Xpixel :
                    # apposto
                    MatrixRealFrame[0:int(xtoPixel + Xpixel / 4 - Xpixel), ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)] = np.sum(
                        [MatrixRealFrame[0:int(xtoPixel + Xpixel / 4 - Xpixel), ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)], MatrixPixels[int(3*Xpixel / 4 - (xtoPixel + Xpixel / 4 - Xpixel)):int(3 * Xpixel / 4), ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)]]
                        , axis=0, dtype='bool')
                    MatrixRealFrame[int(xtoPixel - Xpixel / 4): Xpixel, ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)] = np.sum(
                        [MatrixRealFrame[int(xtoPixel - Xpixel / 4): Xpixel, ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)], MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4) - (int(xtoPixel + Xpixel / 4) - Xpixel), ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)]]
                        , axis=0, dtype='bool')
                if int(xtoPixel - Xpixel/4) < 0 :
                    # apposto
                    MatrixRealFrame[0:int(xtoPixel + Xpixel / 4), ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)] = np.sum(
                        [MatrixRealFrame[0:int(xtoPixel + Xpixel / 4), ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)], MatrixPixels[int( Xpixel / 4 - (xtoPixel - Xpixel / 4)):int(3 * Xpixel / 4), ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)]]
                        , axis=0, dtype='bool')
                    MatrixRealFrame[int(Xpixel+(xtoPixel - Xpixel / 4)): Xpixel, ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)] = np.sum(
                        [MatrixRealFrame[int(Xpixel+(xtoPixel - Xpixel / 4)): Xpixel, ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)], MatrixPixels[int(Xpixel / 4):int(Xpixel / 4 - (xtoPixel - Xpixel / 4)), ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)]]
                        , axis=0, dtype='bool')
                if int(xtoPixel + Xpixel/4) <= Xpixel  and  int(xtoPixel - Xpixel/4) >= 0:
                    # apposto
                    MatrixRealFrame[int(xtoPixel - Xpixel / 4):int(xtoPixel+Xpixel/4), ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)] = np.sum(
                        [MatrixRealFrame[int(xtoPixel - Xpixel / 4):int(xtoPixel+Xpixel/4), ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)], MatrixPixels[int(Xpixel / 4):int(3*Xpixel/4), ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)]]
                        , axis=0, dtype='bool')

            # for l in range (fovYpixel):
            #     if ytoPixel+int(fovYpixel/2)-l > Ypixel:
            #         MatrixRealFrame[0:Xpixel,Ypixel - ((ytoPixel+int(fovYpixel/2)-l) - Ypixel) -1:Ypixel - ((ytoPixel+int(fovYpixel/2)-l) - Ypixel)  ] = np.sum([MatrixRealFrame[0:Xpixel,Ypixel - ((ytoPixel+int(fovYpixel/2)-l) - Ypixel) -1 :Ypixel - ((ytoPixel+int(fovYpixel/2)-l) - Ypixel) ], np.roll(MatrixPixelsOtherside[0:Xpixel,Ypixel - ((ytoPixel+int(fovYpixel/2)-l) - Ypixel) -1:Ypixel - ((ytoPixel+int(fovYpixel/2)-l) - Ypixel)],int(Xpixel/2) + xtoPixel)], axis=0, dtype='bool')
            #     if ytoPixel+int(fovYpixel/2)-l < 0:
            #         MatrixRealFrame[0:Xpixel, abs(ytoPixel+int(fovYpixel/2)-l):abs(ytoPixel+int(fovYpixel/2)-l)+1] = np.sum([MatrixRealFrame[0:Xpixel, abs(ytoPixel+int(fovYpixel/2)-l):abs(ytoPixel+int(fovYpixel/2)-l)+1], np.roll(MatrixPixelsOtherside[0:Xpixel,abs(ytoPixel+int(fovYpixel/2)-l):abs(ytoPixel+int(fovYpixel/2)-l)+1],int(Xpixel/2) + xtoPixel)], axis=0, dtype='bool')
            #     if ytoPixel+int(fovYpixel/2)-l >=0 and ytoPixel-int(fovYpixel/2)-l <=Ypixel:
            #         MatrixRealFrame[0:Xpixel, ytoPixel + int(fovYpixel / 2) - l - 1:ytoPixel + int(fovYpixel / 2) - l] = np.sum([MatrixRealFrame[0:Xpixel, ytoPixel + int(fovYpixel/2) - l - 1:ytoPixel + int(fovYpixel/2) - l ], np.roll(MatrixPixels[0:Xpixel,ytoPixel+int(fovYpixel/2)-l -1:ytoPixel+int(fovYpixel/2)-l],int(Xpixel/2) + xtoPixel)], axis=0, dtype='bool')

        # Calcolo il baricentro e le distanze dei punti dal baricentro

        xBari = sum(coordX)/50
        yBari = sum(coordY)/50

        # Devo ricalcolare il baricentro, il primo baricentro serve per poter calcolare quello reale. Nel primo non si tiene conto della questione sfera
        coordYmod = []
        coordXmod = []
        for l in range (50):
            minX = min( np.sqrt((xBari-coordX[l])**2), np.sqrt((xBari-coordX[l]+Xpixel)**2), np.sqrt((xBari-coordX[l]-Xpixel)**2) )
            minY = min( np.sqrt((yBari-coordY[l])**2), np.sqrt((yBari-coordY[l]+Ypixel)**2), np.sqrt((yBari-coordY[l]-Ypixel)**2) )
            if (minX == np.sqrt((xBari-coordX[l])**2)):
                coordXmod.append(coordX[l])
            if (minX == np.sqrt((xBari-coordX[l]+Xpixel)**2)):
                coordXmod.append(coordX[l]+Xpixel)
            if (minX == np.sqrt((xBari-coordX[l]-Xpixel)**2)):
                coordXmod.append(coordX[l]-Xpixel)
            if (minY == np.sqrt((yBari-coordY[l])**2)):
                coordYmod.append(coordY[l])
            if (minY == np.sqrt((yBari-coordY[l]+Ypixel)**2)):
                coordYmod.append(coordY[l]+Ypixel)
            if (minY == np.sqrt((yBari-coordY[l]-Ypixel)**2)):
                coordYmod.append(coordY[l]-Ypixel)

        xBari = sum(coordXmod) / 50
        yBari = sum(coordYmod) / 50
        coordBaricentro[j,0] = xBari
        coordBaricentro[j,1] = yBari
        for l in range (2,52):
            # sqrt ( (x2-x1)^2 + (y2-x1)^2 )
            coordBaricentro[j,l] = np.sqrt(min((xBari-coordX[l-2])**2,(xBari-coordX[l-2]+Xpixel)**2,(xBari-coordX[l-2]-Xpixel)**2) + min((yBari-coordY[l-2])**2,(yBari-coordY[l-2]+Ypixel)**2,(yBari-coordY[l-2]-Ypixel)**2))

        # Disegno il grafico per un frame e salvo l'immagine
        pixelFrames.append(np.count_nonzero(MatrixRealFrame == 1))
        # fig, ax = plt.subplots()
        # print(MatrixRealFrame[0,0,0])
        # print(MatrixRealFrame[0, 0,1])
        # print(MatrixRealFrame[0, 0, 2])
        # print(MatrixRealFrame[0, 0, 3])
        # print(MatrixRealFrame[1920, 960, 0])
        # print(MatrixRealFrame[1920, 960, 1])
        # print(MatrixRealFrame[1920, 960, 2])
        # print(MatrixRealFrame[1920, 960, 3])
        #
        # MatrixProva = np.invert(np.rot90(MatrixRealFrame)).astype('int32')
        # MatrixProva = MatrixProva * 255
        # print(MatrixProva[0,0,0])
        # print(MatrixProva[0, 0, 1])
        # print(MatrixProva[0, 0, 2])
        # print(MatrixProva[0, 0, 3])
        # print(MatrixProva[960,1920,0])
        # print(MatrixProva[960, 1920, 1])
        # print(MatrixProva[960,1920,2])
        # print(MatrixProva[960,1920,3])
        #
        #
        # mat = ax.imshow((MatrixProva), cmap='Greys', interpolation='nearest')
        # plt.axis('off')
        # fig = plt.gcf()
        # fig.set_size_inches(21, 9)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # plt.plot(coordX, coordY, 'ro')
        # plt.plot([xBari], [yBari], marker='o', color="green")
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # if j<10:
        #     fig.savefig("./../Output/"+folders[f]+"/Frame/frame000"+str(j)+".png", dpi=50, bbox_inches='tight', transparent=True, pad_inches=0)
        #     plt.close('all')
        #     plt.close(fig)
        #     plt.clf()
        #     plt.close()
        # if j>=10 and j<100:
        #     fig.savefig("./../Output/"+folders[f]+"/Frame/frame00"+str(j)+".png", dpi=50, bbox_inches='tight', transparent=True, pad_inches=0)
        #     plt.close('all')
        #     plt.close(fig)
        #     plt.clf()
        #     plt.close()
        # if j>=100 and j<1000:
        #     fig.savefig("./../Output/"+folders[f]+"/Frame/frame0"+str(j)+".png", dpi=50, bbox_inches='tight', transparent=True, pad_inches=0)
        #     plt.close('all')
        #     plt.close(fig)
        #     plt.clf()
        #     plt.close()
        # if j>=1000:
        #     fig.savefig("./../Output/"+folders[f]+"/Frame/frame"+str(j)+".png", dpi=50, bbox_inches='tight', transparent=True, pad_inches=0)
        #     plt.close('all')
        #     plt.close(fig)
        #     plt.clf()
        #     plt.close()
        # plt.close('all')
        # plt.close(fig)
        # plt.clf()
        # plt.close()
        # gc.collect()
        MatrixRealFrame = np.zeros((Xpixel, Ypixel), dtype='bool' )


    # Variabili di supporto per calcolare la percentuale
    Ygraphpercentage = []
    Ygraphmoregranularity = []

    # disegno un grafico  x: #frame  y: #percentuale pixel visti
    for i in range (1800):
        Ygraphpercentage.append(pixelFrames[i]/(Xpixel*Ypixel))
    plt.plot(Ygraphpercentage)
    plt.axis([0, 1800, 0, 1])
    plt.title("Percentuale di visualizzazione per frame")
    plt.xlabel("# Frame")
    plt.ylabel("% visualizzato")
    fig = plt.gcf()
    fig.set_size_inches(21, 9)
    fig.savefig("./../Output/"+folders[f]+"/"+folders[f]+"_"+"Ad-Hoc_Percentuale_In_Aria_Per_Frame.png", dpi=100)
    plt.close('all')
    with open("./../Output/"+folders[f]+"/"+folders[f]+"_"+"Ad-Hoc_ResocontoMemoriaPerFrame.csv", 'w', newline='') as filecsv:
        wr = csv.writer(filecsv)
        wr.writerow(["#frame","KB Frame","In aria (KB)","Risparmio (KB)"])
        for x in range(1800):
            wr.writerow([x,(sizes[f]/1800)*1024,Ygraphpercentage[x]*((sizes[f]/1800)*1024),(1-Ygraphpercentage[x])*((sizes[f]/1800)*1024)])

    # disegno un grafico a granularità maggiore raggruppando i frame di un secondo (30)
    # x: #secondi y: #percentuale pixel visti
    for i in range(60):
        Ygraphmoregranularity.append(sum(Ygraphpercentage[i*30:i*30+30])/30)
    plt.plot(Ygraphmoregranularity)
    plt.axis([0, 60, 0, 1])
    plt.title("Percentuale di visualizzazione per secondo")
    plt.xlabel("# Secondi")
    plt.ylabel("% visualizzato")
    fig = plt.gcf()
    fig.set_size_inches(21, 9)
    fig.savefig("./../Output/"+folders[f]+"/"+folders[f]+"_"+"Ad-Hoc_Percentuale_In_Aria_Per_Secondo.png", dpi=100)
    plt.close('all')


    # Creo il video con tutti i frame
    images = [img for img in os.listdir("./../Output/"+folders[f]+"/Frame") if img.endswith(".png")]
    frame = cv2.imread(os.path.join("./../Output/"+folders[f]+"/Frame", images[0]),cv2.IMREAD_UNCHANGED)
    height, width, layers = frame.shape
    video = cv2.VideoWriter("./../Output/"+folders[f]+"/"+folders[f]+"_Ad-Hoc_Frames.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 30,(width,height))
    for image in images:
        video.write(cv2.imread(os.path.join("./../Output/"+folders[f]+"/Frame", image)))
    cv2.destroyAllWindows()
    video.release()

    # Salvo su un file di testo la percentuale totale
    text_file = open("./../Output/"+folders[f]+"/"+folders[f]+"_"+"Ad-Hoc_Percentuale_Utilizzo_Totale.txt", "w")
    text_file.write(str(sum(Ygraphpercentage)/1800))
    text_file.close()

    # Calcolo le distanze totali dei punti dai loro baricentri per utente
    summationDistances = []
    summationMedia = []
    summationVarianza = []
    text_file = open("./../Output/" + folders[f] +"/"+folders[f]+"_"+"Ad-Hoc_Distances_Media_Varianza.txt", "w")

    for i in range(50):
        summation = 0
        varianza = 0
        for j in range (1800):
            summation = summation + coordBaricentro [j,i+2]
        summationDistances.append(summation)
        summationMedia.append(summation/1800)
        for j in range (1800):
            varianza = varianza + ((coordBaricentro [j,i+2] - summationMedia[i] ) ** 2)
        summationVarianza.append(varianza/1800)
        text_file.write(str(i) + " - " + str(summation)+" - "+str(summationMedia[i])+" - "+str(summationVarianza[i]) + "\n")
    text_file.close()

    text_file = open("./../Output/" + folders[f] +"/"+folders[f]+"_"+"Ad-Hoc_DistancesTraveled.txt", "w")
    for i in range(50):
        text_file.write(str(i) + " - " +str(distancesTraveled[i]) + "\n")
    text_file.close()

    # Calcolo il risparmio di spazio
    text_file = open("./../Output/" + folders[f] +"/"+folders[f]+"_"+"Ad-Hoc_DimensionsSave.txt", "w")
    text_file.write("%0.2f" % ((sum(Ygraphpercentage) / 1800) * sizes[f]) + " / " + str(sizes[f]))
    text_file.close()








    # pixelFrames = []
    #
    # # Ciclo per ogni frame per ricavare le varie immagini e i vari dati
    # a = 0
    # for j in range(1800):
    #
    #     coordX = []
    #     coordY = []
    #     coordXsep = []
    #     coordYsep = []
    #     a = 0
    #     # Ci sono 50 utenti quindi analizzo i dati per ogni utente
    #     for i in range(50):
    #         if summationDistances[i] < 1000000:
    #             a = a+1
    #             xtoPixel = int((CoordinateData[i][j][0] + 180) / coordToPixelX)
    #             ytoPixel = int((CoordinateData[i][j][1] + 90) / coordToPixelY)
    #             coordX.append(xtoPixel)
    #             # y invertito su asse centrale (ypixel/2)
    #             coordY.append((int(Ypixel / 2) - ytoPixel) * 2 + ytoPixel)
    #
    #             # disegno sulla matrice del frame i pixel visualizzati dall'utente
    #             if int(ytoPixel + fovYpixel / 2) > Ypixel:
    #                 if int(xtoPixel + Xpixel / 4) > Xpixel:
    #                     # apposto
    #                     MatrixRealFrame[0:int(xtoPixel + Xpixel / 4 - Xpixel),
    #                     ytoPixel - int(fovYpixel / 2): Ypixel] = np.sum(
    #                         [MatrixRealFrame[0:int(xtoPixel + Xpixel / 4 - Xpixel), ytoPixel - int(fovYpixel / 2): Ypixel],
    #                             MatrixPixels[int(3 * Xpixel / 4 - (xtoPixel + Xpixel / 4 - Xpixel)):int(3 * Xpixel / 4),
    #                             ytoPixel - int(fovYpixel / 2): Ypixel]]
    #                             , axis=0, dtype='bool')
    #                     MatrixRealFrame[int(xtoPixel - Xpixel / 4): Xpixel, ytoPixel - int(fovYpixel / 2): Ypixel] = np.sum(
    #                             [MatrixRealFrame[int(xtoPixel - Xpixel / 4): Xpixel, ytoPixel - int(fovYpixel / 2): Ypixel],
    #                              MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4) - (int(xtoPixel + Xpixel / 4) - Xpixel),
    #                              ytoPixel - int(fovYpixel / 2): Ypixel]]
    #                             , axis=0, dtype='bool')
    #                     MatrixRealFrame[int(xtoPixel + Xpixel / 4 - Xpixel):int(xtoPixel - Xpixel / 4),
    #                         int(Ypixel - (ytoPixel + fovYpixel / 2 - Ypixel)):Ypixel] = np.sum(
    #                             [MatrixRealFrame[int(xtoPixel + Xpixel / 4 - Xpixel):int(xtoPixel - Xpixel / 4),
    #                              int(Ypixel - (ytoPixel + fovYpixel / 2 - Ypixel)):Ypixel],
    #                              MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4),
    #                              int(Ypixel - (ytoPixel + fovYpixel / 2 - Ypixel)):Ypixel]]
    #                             , axis=0, dtype='bool')
    #                 if int(xtoPixel - Xpixel / 4) < 0:
    #                         # da vedere
    #                     MatrixRealFrame[0:int(xtoPixel + Xpixel / 4), ytoPixel - int(fovYpixel / 2): Ypixel] = np.sum(
    #                             [MatrixRealFrame[0:int(xtoPixel + Xpixel / 4), ytoPixel - int(fovYpixel / 2): Ypixel],
    #                              MatrixPixels[int(Xpixel / 4 - (xtoPixel - Xpixel / 4)):int(3 * Xpixel / 4),
    #                              ytoPixel - int(fovYpixel / 2): Ypixel]]
    #                             , axis=0, dtype='bool')
    #                     MatrixRealFrame[int(Xpixel + (xtoPixel - Xpixel / 4)): Xpixel,
    #                         ytoPixel - int(fovYpixel / 2): Ypixel] = np.sum(
    #                             [MatrixRealFrame[int(Xpixel + (xtoPixel - Xpixel / 4)): Xpixel,
    #                              ytoPixel - int(fovYpixel / 2): Ypixel],
    #                              MatrixPixels[int(Xpixel / 4):int(Xpixel / 4 - (xtoPixel - Xpixel / 4)),
    #                              ytoPixel - int(fovYpixel / 2): Ypixel]]
    #                             , axis=0, dtype='bool')
    #                     MatrixRealFrame[int(xtoPixel + Xpixel / 4):int(Xpixel + (xtoPixel - Xpixel / 4)),
    #                         int(Ypixel - (ytoPixel + fovYpixel / 2 - Ypixel)):Ypixel] = np.sum(
    #                             [MatrixRealFrame[int(xtoPixel + Xpixel / 4):int(Xpixel + (xtoPixel - Xpixel / 4)),
    #                              int(Ypixel - (ytoPixel + fovYpixel / 2 - Ypixel)):Ypixel],
    #                              MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4),
    #                              int(Ypixel - (ytoPixel + fovYpixel / 2 - Ypixel)):Ypixel]]
    #                             , axis=0, dtype='bool')
    #                 if int(xtoPixel + Xpixel / 4) <= Xpixel and int(xtoPixel - Xpixel / 4) >= 0:
    #                     # apposto
    #                     MatrixRealFrame[int(xtoPixel - Xpixel / 4):int(xtoPixel + Xpixel / 4),
    #                         ytoPixel - int(fovYpixel / 2): Ypixel] = np.sum(
    #                             [MatrixRealFrame[int(xtoPixel - Xpixel / 4):int(xtoPixel + Xpixel / 4),
    #                              ytoPixel - int(fovYpixel / 2): Ypixel],
    #                              MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4), ytoPixel - int(fovYpixel / 2): Ypixel]]
    #                             , axis=0, dtype='bool')
    #                     MatrixRealFrame[0:int(xtoPixel - Xpixel / 4),
    #                         int(Ypixel - (ytoPixel + fovYpixel / 2 - Ypixel)):Ypixel] = np.sum(
    #                             [MatrixRealFrame[0:int(xtoPixel - Xpixel / 4),
    #                              int(Ypixel - (ytoPixel + fovYpixel / 2 - Ypixel)):Ypixel],
    #                              MatrixPixels[int(3 * Xpixel / 4 - (xtoPixel - Xpixel / 4)):int(3 * Xpixel / 4),
    #                              int(Ypixel - (ytoPixel + fovYpixel / 2 - Ypixel)):Ypixel]]
    #                             , axis=0, dtype='bool')
    #                     MatrixRealFrame[int(xtoPixel + Xpixel / 4): Xpixel,
    #                         int(Ypixel - (ytoPixel + fovYpixel / 2 - Ypixel)):Ypixel] = np.sum(
    #                             [MatrixRealFrame[int(xtoPixel + Xpixel / 4): Xpixel,
    #                              int(Ypixel - (ytoPixel + fovYpixel / 2 - Ypixel)):Ypixel],
    #                              MatrixPixels[int(Xpixel / 4):int(Xpixel / 4 + (Xpixel - (xtoPixel + Xpixel / 4))),
    #                              int(Ypixel - (ytoPixel + fovYpixel / 2 - Ypixel)):Ypixel]]
    #                             , axis=0, dtype='bool')
    #             if int(ytoPixel - fovYpixel / 2) < 0:
    #                 if int(xtoPixel + Xpixel / 4) > Xpixel:
    #                         # da vedere
    #                     MatrixRealFrame[0:int(xtoPixel + Xpixel / 4 - Xpixel), 0: ytoPixel + int(fovYpixel / 2)] = np.sum(
    #                             [MatrixRealFrame[0:int(xtoPixel + Xpixel / 4 - Xpixel), 0: ytoPixel + int(fovYpixel / 2)],
    #                              MatrixPixels[int(3 * Xpixel / 4 - (xtoPixel + Xpixel / 4 - Xpixel)):int(3 * Xpixel / 4),
    #                              0: ytoPixel + int(fovYpixel / 2)]]
    #                             , axis=0, dtype='bool')
    #                     MatrixRealFrame[int(xtoPixel - Xpixel / 4): Xpixel, 0: ytoPixel + int(fovYpixel / 2)] = np.sum(
    #                             [MatrixRealFrame[int(xtoPixel - Xpixel / 4): Xpixel, 0: ytoPixel + int(fovYpixel / 2)],
    #                              MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4) - (int(xtoPixel + Xpixel / 4) - Xpixel),
    #                              0: ytoPixel + int(fovYpixel / 2)]]
    #                             , axis=0, dtype='bool')
    #                     MatrixRealFrame[int(xtoPixel + Xpixel / 4 - Xpixel):int(xtoPixel - Xpixel / 4),
    #                         0:abs(ytoPixel - int(fovYpixel / 2))] = np.sum(
    #                             [MatrixRealFrame[int(xtoPixel + Xpixel / 4 - Xpixel):int(xtoPixel - Xpixel / 4),
    #                              0:abs(ytoPixel - int(fovYpixel / 2))],
    #                              MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4), 0:abs(ytoPixel - int(fovYpixel / 2))]]
    #                             , axis=0, dtype='bool')
    #                 if int(xtoPixel - Xpixel / 4) < 0:
    #                         # apposto
    #                     MatrixRealFrame[0:int(xtoPixel + Xpixel / 4), 0: ytoPixel + int(fovYpixel / 2)] = np.sum(
    #                             [MatrixRealFrame[0:int(xtoPixel + Xpixel / 4), 0: ytoPixel + int(fovYpixel / 2)],
    #                              MatrixPixels[int(Xpixel / 4 - (xtoPixel - Xpixel / 4)):int(3 * Xpixel / 4),
    #                              0: ytoPixel + int(fovYpixel / 2)]]
    #                             , axis=0, dtype='bool')
    #                     MatrixRealFrame[int(Xpixel + (xtoPixel - Xpixel / 4)): Xpixel,
    #                         0: ytoPixel + int(fovYpixel / 2)] = np.sum(
    #                             [MatrixRealFrame[int(Xpixel + (xtoPixel - Xpixel / 4)): Xpixel,
    #                              0: ytoPixel + int(fovYpixel / 2)],
    #                              MatrixPixels[int(Xpixel / 4):int(Xpixel / 4 - (xtoPixel - Xpixel / 4)),
    #                              0: ytoPixel + int(fovYpixel / 2)]]
    #                             , axis=0, dtype='bool')
    #                     MatrixRealFrame[int(xtoPixel + Xpixel / 4):int(Xpixel + (xtoPixel - Xpixel / 4)),
    #                         0:abs(ytoPixel - int(fovYpixel / 2))] = np.sum(
    #                             [MatrixRealFrame[int(xtoPixel + Xpixel / 4):int(Xpixel + (xtoPixel - Xpixel / 4)),
    #                              0:abs(ytoPixel - int(fovYpixel / 2))],
    #                              MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4), 0:abs(ytoPixel - int(fovYpixel / 2))]]
    #                             , axis=0, dtype='bool')
    #                 if int(xtoPixel + Xpixel / 4) <= Xpixel and int(xtoPixel - Xpixel / 4) >= 0:
    #                         # apposto
    #                     MatrixRealFrame[int(xtoPixel - Xpixel / 4):int(xtoPixel + Xpixel / 4),
    #                         0: ytoPixel + int(fovYpixel / 2)] = np.sum(
    #                             [MatrixRealFrame[int(xtoPixel - Xpixel / 4):int(xtoPixel + Xpixel / 4),
    #                              0: ytoPixel + int(fovYpixel / 2)],
    #                              MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4), 0: ytoPixel + int(fovYpixel / 2)]]
    #                             , axis=0, dtype='bool')
    #                     MatrixRealFrame[0:int(xtoPixel - Xpixel / 4), 0:abs(ytoPixel - int(fovYpixel / 2))] = np.sum(
    #                             [MatrixRealFrame[0:int(xtoPixel - Xpixel / 4), 0:abs(ytoPixel - int(fovYpixel / 2))],
    #                              MatrixPixels[int(3 * Xpixel / 4 - (xtoPixel - Xpixel / 4)):int(3 * Xpixel / 4),
    #                              0:abs(ytoPixel - int(fovYpixel / 2))]]
    #                             , axis=0, dtype='bool')
    #                     MatrixRealFrame[int(xtoPixel + Xpixel / 4): Xpixel, 0:abs(ytoPixel - int(fovYpixel / 2))] = np.sum(
    #                             [MatrixRealFrame[int(xtoPixel + Xpixel / 4): Xpixel, 0:abs(ytoPixel - int(fovYpixel / 2))],
    #                              MatrixPixels[int(Xpixel / 4):int(Xpixel / 4 + (Xpixel - (xtoPixel + Xpixel / 4))),
    #                              0:abs(ytoPixel - int(fovYpixel / 2))]]
    #                             , axis=0, dtype='bool')
    #             if int(ytoPixel + fovYpixel / 2) <= Ypixel and int(ytoPixel - fovYpixel / 2) >= 0:
    #                 if int(xtoPixel + Xpixel / 4) > Xpixel:
    #                         # apposto
    #                     MatrixRealFrame[0:int(xtoPixel + Xpixel / 4 - Xpixel),
    #                         ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)] = np.sum(
    #                             [MatrixRealFrame[0:int(xtoPixel + Xpixel / 4 - Xpixel),
    #                              ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)],
    #                              MatrixPixels[int(3 * Xpixel / 4 - (xtoPixel + Xpixel / 4 - Xpixel)):int(3 * Xpixel / 4),
    #                              ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)]]
    #                             , axis=0, dtype='bool')
    #                     MatrixRealFrame[int(xtoPixel - Xpixel / 4): Xpixel,
    #                         ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)] = np.sum(
    #                             [MatrixRealFrame[int(xtoPixel - Xpixel / 4): Xpixel,
    #                              ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)],
    #                              MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4) - (int(xtoPixel + Xpixel / 4) - Xpixel),
    #                              ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)]]
    #                             , axis=0, dtype='bool')
    #                 if int(xtoPixel - Xpixel / 4) < 0:
    #                         # apposto
    #                     MatrixRealFrame[0:int(xtoPixel + Xpixel / 4),
    #                         ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)] = np.sum(
    #                             [MatrixRealFrame[0:int(xtoPixel + Xpixel / 4),
    #                              ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)],
    #                              MatrixPixels[int(Xpixel / 4 - (xtoPixel - Xpixel / 4)):int(3 * Xpixel / 4),
    #                              ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)]]
    #                             , axis=0, dtype='bool')
    #                     MatrixRealFrame[int(Xpixel + (xtoPixel - Xpixel / 4)): Xpixel,
    #                         ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)] = np.sum(
    #                             [MatrixRealFrame[int(Xpixel + (xtoPixel - Xpixel / 4)): Xpixel,
    #                              ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)],
    #                              MatrixPixels[int(Xpixel / 4):int(Xpixel / 4 - (xtoPixel - Xpixel / 4)),
    #                              ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)]]
    #                             , axis=0, dtype='bool')
    #                 if int(xtoPixel + Xpixel / 4) <= Xpixel and int(xtoPixel - Xpixel / 4) >= 0:
    #                         # apposto
    #                     MatrixRealFrame[int(xtoPixel - Xpixel / 4):int(xtoPixel + Xpixel / 4),
    #                         ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)] = np.sum(
    #                             [MatrixRealFrame[int(xtoPixel - Xpixel / 4):int(xtoPixel + Xpixel / 4),
    #                              ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)],
    #                              MatrixPixels[int(Xpixel / 4):int(3 * Xpixel / 4),
    #                              ytoPixel - int(fovYpixel / 2): ytoPixel + int(fovYpixel / 2)]]
    #                             , axis=0, dtype='bool')
    #         else :
    #             xtoPixel = int((CoordinateData[i][j][0] + 180) / coordToPixelX)
    #             ytoPixel = int((CoordinateData[i][j][1] + 90) / coordToPixelY)
    #             coordXsep.append(xtoPixel)
    #             # y invertito su asse centrale (ypixel/2)
    #             coordYsep.append((int(Ypixel / 2) - ytoPixel) * 2 + ytoPixel)
    #         # for l in range (fovYpixel):
    #         #     if ytoPixel+int(fovYpixel/2)-l > Ypixel:
    #         #         MatrixRealFrame[0:Xpixel,Ypixel - ((ytoPixel+int(fovYpixel/2)-l) - Ypixel) -1:Ypixel - ((ytoPixel+int(fovYpixel/2)-l) - Ypixel)  ] = np.sum([MatrixRealFrame[0:Xpixel,Ypixel - ((ytoPixel+int(fovYpixel/2)-l) - Ypixel) -1 :Ypixel - ((ytoPixel+int(fovYpixel/2)-l) - Ypixel) ], np.roll(MatrixPixelsOtherside[0:Xpixel,Ypixel - ((ytoPixel+int(fovYpixel/2)-l) - Ypixel) -1:Ypixel - ((ytoPixel+int(fovYpixel/2)-l) - Ypixel)],int(Xpixel/2) + xtoPixel)], axis=0, dtype='bool')
    #         #     if ytoPixel+int(fovYpixel/2)-l < 0:
    #         #         MatrixRealFrame[0:Xpixel, abs(ytoPixel+int(fovYpixel/2)-l):abs(ytoPixel+int(fovYpixel/2)-l)+1] = np.sum([MatrixRealFrame[0:Xpixel, abs(ytoPixel+int(fovYpixel/2)-l):abs(ytoPixel+int(fovYpixel/2)-l)+1], np.roll(MatrixPixelsOtherside[0:Xpixel,abs(ytoPixel+int(fovYpixel/2)-l):abs(ytoPixel+int(fovYpixel/2)-l)+1],int(Xpixel/2) + xtoPixel)], axis=0, dtype='bool')
    #         #     if ytoPixel+int(fovYpixel/2)-l >=0 and ytoPixel-int(fovYpixel/2)-l <=Ypixel:
    #         #         MatrixRealFrame[0:Xpixel, ytoPixel + int(fovYpixel / 2) - l - 1:ytoPixel + int(fovYpixel / 2) - l] = np.sum([MatrixRealFrame[0:Xpixel, ytoPixel + int(fovYpixel/2) - l - 1:ytoPixel + int(fovYpixel/2) - l ], np.roll(MatrixPixels[0:Xpixel,ytoPixel+int(fovYpixel/2)-l -1:ytoPixel+int(fovYpixel/2)-l],int(Xpixel/2) + xtoPixel)], axis=0, dtype='bool')
    #
    #     # Calcolo il baricentro e le distanze dei punti dal baricentro
    #
    #     print (a)
    #
    #     xBari=0
    #     yBari=0
    #     for i in range (a):
    #         if summationDistances[i] < 1000000:
    #             xBari = xBari + coordX[i]
    #             yBari = yBari + coordY[i]
    #         xBari = xBari/a
    #         yBari = yBari/a
    #
    #     # Devo ricalcolare il baricentro, il primo baricentro serve per poter calcolare quello reale. Nel primo non si tiene conto della questione sfera
    #     coordYmod = []
    #     coordXmod = []
    #     for l in range(a):
    #         minX = min(np.sqrt((xBari - coordX[l]) ** 2), np.sqrt((xBari - coordX[l] + Xpixel) ** 2),
    #                    np.sqrt((xBari - coordX[l] - Xpixel) ** 2))
    #         minY = min(np.sqrt((yBari - coordY[l]) ** 2), np.sqrt((yBari - coordY[l] + Ypixel) ** 2),
    #                    np.sqrt((yBari - coordY[l] - Ypixel) ** 2))
    #         if (minX == np.sqrt((xBari - coordX[l]) ** 2)):
    #             coordXmod.append(coordX[l])
    #         if (minX == np.sqrt((xBari - coordX[l] + Xpixel) ** 2)):
    #             coordXmod.append(coordX[l] + Xpixel)
    #         if (minX == np.sqrt((xBari - coordX[l] - Xpixel) ** 2)):
    #             coordXmod.append(coordX[l] - Xpixel)
    #         if (minY == np.sqrt((yBari - coordY[l]) ** 2)):
    #             coordYmod.append(coordY[l])
    #         if (minY == np.sqrt((yBari - coordY[l] + Ypixel) ** 2)):
    #             coordYmod.append(coordY[l] + Ypixel)
    #         if (minY == np.sqrt((yBari - coordY[l] - Ypixel) ** 2)):
    #             coordYmod.append(coordY[l] - Ypixel)
    #
    #     for i in range (a):
    #         if summationDistances[i] < 1000000:
    #             xBari = xBari + coordXmod[i]
    #             yBari = yBari + coordYmod[i]
    #         xBari = xBari/a
    #         yBari = yBari/a
    #
    #     coordBaricentro[j, 0] = xBari
    #     coordBaricentro[j, 1] = yBari
    #     for l in range(2, a+2):
    #         # sqrt ( (x2-x1)^2 + (y2-x1)^2 )
    #         coordBaricentro[j, l] = np.sqrt(min((xBari - coordX[l - 2]) ** 2, (xBari - coordX[l - 2] + Xpixel) ** 2,
    #                                             (xBari - coordX[l - 2] - Xpixel) ** 2) + min(
    #             (yBari - coordY[l - 2]) ** 2, (yBari - coordY[l - 2] + Ypixel) ** 2,
    #             (yBari - coordY[l - 2] - Ypixel) ** 2))
    #
    #     # Disegno il grafico per un frame e salvo l'immagine
    #     pixelFrames.append(np.count_nonzero(MatrixRealFrame == 1))
    #     fig, ax = plt.subplots()
    #     mat = ax.imshow((np.rot90(MatrixRealFrame)), cmap='GnBu', interpolation='nearest')
    #     plt.axis('off')
    #     fig = plt.gcf()
    #     fig.set_size_inches(21, 9)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['bottom'].set_visible(False)
    #     ax.spines['left'].set_visible(False)
    #     plt.plot(coordX, coordY, 'ro')
    #     plt.plot(coordXsep, coordYsep, 'yo')
    #     plt.plot([xBari], [yBari], marker='o', color="green")
    #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #     plt.margins(0, 0)
    #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #     if j < 10:
    #         fig.savefig("./../Output/" + folders[f] + "/ModFrame/Modframe000" + str(j) + ".png", dpi=50, bbox_inches='tight',
    #                     transparent=True, pad_inches=0)
    #     if j >= 10 and j < 100:
    #         fig.savefig("./../Output/" + folders[f] + "/ModFrame/Modframe00" + str(j) + ".png", dpi=50, bbox_inches='tight',
    #                     transparent=True, pad_inches=0)
    #     if j >= 100 and j < 1000:
    #         fig.savefig("./../Output/" + folders[f] + "/ModFrame/Modframe0" + str(j) + ".png", dpi=50, bbox_inches='tight',
    #                     transparent=True, pad_inches=0)
    #     if j >= 1000:
    #         fig.savefig("./../Output/" + folders[f] + "/ModFrame/Modframe" + str(j) + ".png", dpi=50, bbox_inches='tight',
    #                     transparent=True, pad_inches=0)
    #     plt.close('all')
    #     MatrixRealFrame = np.zeros((Xpixel, Ypixel), dtype='bool')
    #
    # # Variabili di supporto per calcolare la percentuale
    # Ygraphpercentage = []
    # Ygraphmoregranularity = []
    #
    # # disegno un grafico  x: #frame  y: #percentuale pixel visti
    # for i in range(1800):
    #     Ygraphpercentage.append(pixelFrames[i] / (Xpixel * Ypixel))
    # plt.plot(Ygraphpercentage)
    # plt.axis([0, 1800, 0, 1])
    # plt.title("Percentuale di visualizzazione per frame")
    # plt.xlabel("# Frame")
    # plt.ylabel("% visualizzato")
    # fig = plt.gcf()
    # fig.set_size_inches(21, 9)
    # fig.savefig("./../Output/" + folders[f] + "/Mod3DPercentuale.png", dpi=100)
    # plt.close('all')
    #
    # # disegno un grafico a granularità maggiore raggruppando i frame di un secondo (30)
    # # x: #secondi y: #percentuale pixel visti
    # for i in range(60):
    #     Ygraphmoregranularity.append(sum(Ygraphpercentage[i * 30:i * 30 + 30]) / 30)
    # plt.plot(Ygraphmoregranularity)
    # plt.axis([0, 60, 0, 1])
    # plt.title("Percentuale di visualizzazione per secondo")
    # plt.xlabel("# Secondi")
    # plt.ylabel("% visualizzato")
    # fig = plt.gcf()
    # fig.set_size_inches(21, 9)
    # fig.savefig("./../Output/" + folders[f] + "/Mod3DPercentualeGranularitàMaggiore.png", dpi=100)
    # plt.close('all')
    #
    # # Creo il video con tutti i frame
    # images = [img for img in os.listdir("./../Output/" + folders[f] + "/ModFrame") if img.endswith(".png")]
    # frame = cv2.imread(os.path.join("./../Output/" + folders[f] + "/ModFrame", images[0]), cv2.IMREAD_UNCHANGED)
    # height, width, layers = frame.shape
    # video = cv2.VideoWriter("./../Output/" + folders[f] + "/Modframe.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 30,
    #                         (width, height))
    # for image in images:
    #     video.write(cv2.imread(os.path.join("./../Output/" + folders[f] + "/ModFrame", image)))
    # cv2.destroyAllWindows()
    # video.release()
    #
    # # Salvo su un file di testo la percentuale totale
    # text_file = open("./../Output/" + folders[f] + "/Mod3DPercentuale.txt", "w")
    # text_file.write(str(sum(Ygraphpercentage) / 1800))
    # text_file.close()
    #
    # # Calcolo le distanze totali dei punti dai loro baricentri per utente
    # text_file = open("./../Output/" + folders[f] + "/Mod3DDistances.txt", "w")
    # for i in range(a):
    #     summation = 0
    #     if summationDistances[i] < 1000000:
    #         for j in range(1800):
    #             summation = summation + coordBaricentro[j, i + 2]
    #         text_file.write(str(summation) + "\n")
    # text_file.close()
    #
    # # Calcolo il risparmio di spazio
    # text_file = open("./../Output/" + folders[f] + "/Mod3DDimensionSave.txt", "w")
    # text_file.write("%0.2f" % ((sum(Ygraphpercentage) / 1800) * sizes[f]) + " / " + str(sizes[f]))
    # text_file.close()
