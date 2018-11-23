import csv
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

stop = False








Xpixel = 3840
Ypixel = 1920
MatrixPixels = np.zeros((Xpixel, Ypixel),dtype = 'bool')
MatrixPixelsOtherside = np.zeros((Xpixel, Ypixel),dtype = 'bool')
MatrixRealFrame = np.zeros((Xpixel, Ypixel),dtype='bool')

#fovX dipende dal pitch di partenza
fovX = 100
fovY = 100
fovXpixel = int(Xpixel/360*fovX)
fovYpixel = int(Ypixel/180*fovY)

listOfFile = glob.glob("./*.csv")

CoordinateData = []
for i in range (len(listOfFile)):
    CoordinateData.append(np.zeros((1800, 2)))







for i in range(int(Ypixel/2 - fovXpixel/2)):
    MatrixPixels[int(Xpixel/4 + (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2) : int(3*Xpixel/4 - (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2) , Ypixel-1-i-int(fovXpixel/2):Ypixel-i-int(fovXpixel/2)] = \
        np.ones((int(3*Xpixel/4 - (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2)-int(Xpixel/4 + (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2), 1), dtype='bool')

for i in range(int(Ypixel/2 - fovXpixel/2)):
    MatrixPixels[int(Xpixel/4 + (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2) : int(3*Xpixel/4 - (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2) , i+int(fovXpixel/2):i+int(fovXpixel/2)+1] = \
        np.ones((int(3*Xpixel/4 - (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2)-int(Xpixel/4 + (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2), 1), dtype='bool')



for i in range(int(Ypixel/2 - fovXpixel/2)):
    MatrixPixelsOtherside[0 : int(Xpixel/4 - (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2) , Ypixel-1-i-int(fovXpixel/2):Ypixel-i-int(fovXpixel/2)] = \
        np.ones((int(Xpixel/4 - (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2), 1), dtype='bool')
    MatrixPixelsOtherside[int(3*Xpixel / 4 + (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2): int(Xpixel), Ypixel - 1 - i - int(fovXpixel / 2):Ypixel - i - int(fovXpixel / 2)] = \
        np.ones((int( Xpixel  - (i * (Ypixel - fovXpixel) / (Ypixel / 2 - fovXpixel / 2)) / 2) - int(3*Xpixel / 4 ), 1), dtype='bool')

for i in range(int(Ypixel/2 - fovXpixel/2)):
    MatrixPixelsOtherside[ 0 : int(Xpixel/4 - (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2) , i+int(fovXpixel/2):i+int(fovXpixel/2)+1] = \
        np.ones((int(Xpixel/4 - (i*(Ypixel-fovXpixel)/(Ypixel/2-fovXpixel/2))/2), 1), dtype='bool')
    MatrixPixelsOtherside[int(3*Xpixel / 4 + (i * (Ypixel - fovXpixel) / (Ypixel / 2 - fovXpixel / 2)) / 2): int( Xpixel ),i + int(fovXpixel / 2):i + int(fovXpixel / 2) + 1] = \
        np.ones((int(Xpixel) - int( 3*Xpixel / 4 + (i * (Ypixel - fovXpixel) / (Ypixel / 2 - fovXpixel / 2)) / 2), 1), dtype='bool')




MatrixPixels[int(Xpixel/4):int(3*Xpixel/4) , int(Ypixel-fovXpixel/2):Ypixel] = np.ones((int(Xpixel/2),Ypixel - int(Ypixel-fovXpixel/2)),dtype='bool')
MatrixPixels[int(Xpixel/4):int(3*Xpixel/4) , 0:int(fovXpixel/2)] = np.ones((int(Xpixel/2),int(fovXpixel/2)),dtype='bool')

MatrixPixelsOtherside[0:int(Xpixel/4) , int(Ypixel-fovXpixel/2):Ypixel] = np.ones((int(Xpixel/4),Ypixel - int(Ypixel-fovXpixel/2)),dtype='bool')
MatrixPixelsOtherside[int(3*Xpixel/4):Xpixel , int(Ypixel-fovXpixel/2):Ypixel] = np.ones((int(Xpixel/4),Ypixel - int(Ypixel-fovXpixel/2)),dtype='bool')
MatrixPixelsOtherside[0:int(Xpixel/4) , 0:int(fovXpixel/2)] = np.ones((int(Xpixel/4),int(fovXpixel/2)),dtype='bool')
MatrixPixelsOtherside[int(3*Xpixel/4):Xpixel , 0:int(fovXpixel/2)] = np.ones((int(Xpixel/4),int(fovXpixel/2)),dtype='bool')














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

            CoordinateData[i][l][0] = riga[4]
            CoordinateData[i][l][1] = riga[5]

            l = l + 1

coordToPixelX = 360/Xpixel
coordToPixelY = 180/Ypixel




pixelFrames = []


for j in range(1800):
    coordX = []
    coordY = []
    for i in range (50):




        xtoPixel = int((CoordinateData[i][j][0]+180)/coordToPixelX)
        ytoPixel = int((CoordinateData[i][j][1]+90)/coordToPixelY)

        coordX.append(xtoPixel)
        #y invertito su asse centrale (ypixel/2)
        coordY.append((int(Ypixel / 2) - ytoPixel) * 2 + ytoPixel)




        for l in range (fovYpixel):
            # parte superiore da riguardare
            if ytoPixel+int(fovYpixel/2)-l > Ypixel:
                MatrixRealFrame[0:Xpixel,Ypixel - (ytoPixel+int(fovYpixel/2)-Ypixel) - l -1:Ypixel - (ytoPixel+int(fovYpixel/2)-Ypixel) - l ] = np.sum([MatrixRealFrame[0:Xpixel,Ypixel - (ytoPixel+int(fovYpixel/2)-Ypixel) - l -1 :Ypixel - (ytoPixel+int(fovYpixel/2)-Ypixel) - l ], np.roll(MatrixPixelsOtherside[0:Xpixel,Ypixel - (ytoPixel+int(fovYpixel/2)-Ypixel) - l -1:Ypixel - (ytoPixel+int(fovYpixel/2)-Ypixel) - l],int(Xpixel/2) + xtoPixel)], axis=0, dtype='bool')
            if ytoPixel+int(fovYpixel/2)-l < 0:
                MatrixRealFrame[0:Xpixel, abs(ytoPixel+int(fovYpixel/2)-l):abs(ytoPixel+int(fovYpixel/2)-l)+1] = np.sum([MatrixRealFrame[0:Xpixel, abs(ytoPixel+int(fovYpixel/2)-l):abs(ytoPixel+int(fovYpixel/2)-l)+1], np.roll(MatrixPixelsOtherside[0:Xpixel,abs(ytoPixel+int(fovYpixel/2)-l):abs(ytoPixel+int(fovYpixel/2)-l)+1],int(Xpixel/2) + xtoPixel)], axis=0, dtype='bool')
            if ytoPixel+int(fovYpixel/2)-l >=0 and ytoPixel-int(fovYpixel/2)-l <=Ypixel:
                MatrixRealFrame[0:Xpixel, ytoPixel + int(fovYpixel / 2) - l - 1:ytoPixel + int(fovYpixel / 2) - l] = np.sum([MatrixRealFrame[0:Xpixel, ytoPixel + int(fovYpixel/2) - l - 1:ytoPixel + int(fovYpixel/2) - l ], np.roll(MatrixPixels[0:Xpixel,ytoPixel+int(fovYpixel/2)-l -1:ytoPixel+int(fovYpixel/2)-l],int(Xpixel/2) + xtoPixel)], axis=0, dtype='bool')


    pixelFrames.append(np.count_nonzero(MatrixRealFrame == 1))
    fig, ax = plt.subplots()
    mat = ax.imshow((np.rot90(MatrixRealFrame)), cmap='GnBu', interpolation='nearest')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(21, 9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.plot(coordX, coordY, 'ro')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if j<10:
        fig.savefig("./Output/frame000"+str(j)+".png", dpi=50, bbox_inches='tight', transparent=True, pad_inches=0)
    if j>=10 and j<100:
        fig.savefig("./Output/frame00"+str(j)+".png", dpi=50, bbox_inches='tight', transparent=True, pad_inches=0)
    if j>=100 and j<1000:
        fig.savefig("./Output/frame0"+str(j)+".png", dpi=50, bbox_inches='tight', transparent=True, pad_inches=0)
    if j>=1000:
        fig.savefig("./Output/frame"+str(j)+".png", dpi=50, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close('all')
    MatrixRealFrame = np.zeros((Xpixel, Ypixel), dtype='bool' )


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


print(sum(Ygraphpercentage)/1800)

plt.close('all')
# disegno un grafico a granularità maggiore raggruppando i frame di un secondo (30)
# x: #secondi y: #percentuale pixel visti
for i in range(60):
    Ygraphmoregranularity.append(sum(Ygraphpercentage[i*30:i*30+30])/30)

plt.plot(Ygraphmoregranularity)
fig = plt.gcf()
fig.set_size_inches(21, 9)
fig.savefig("./../Output/awfawfa.png", dpi=100)

plt.close('all')


image_folder = 'images'
video_name = 'video.avi'

images = [img for img in os.listdir("./Output") if img.endswith(".png")]
print(images)
frame = cv2.imread(os.path.join("./Output", images[0]),cv2.IMREAD_UNCHANGED)
height, width, layers = frame.shape

video = cv2.VideoWriter("./Output/video.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 30,(width,height))

for image in images:
    video.write(cv2.imread(os.path.join("./Output", image)))

cv2.destroyAllWindows()
video.release()

plt.plot(1)
plt.show()

