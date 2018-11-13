import csv
import os
import glob
import matplotlib.pyplot as plt
import math

listOfFile = glob.glob("./*.csv")
listofset = []
listoflist = []
stop = False

y = []

#  leggo i vari file e salvo tutti i tiles riga per riga in una lista di set per non avere doppioni.
#  Valido per qualsiasi tipo di file con le caratteristiche appropiate
for i in range(len(listOfFile)):
    with open(listOfFile[i], newline="", encoding="ISO-8859-1") as filecsv:
        lettore = csv.reader(filecsv,delimiter=",")
        header = next(lettore)
        l=0
        while True:
            if not stop:
                listofset.append(set())
            try:
                riga = next(lettore)
            except:
                break
            for m in range(1,(len(riga))):
                listofset[l].add(riga[m])
            l=l+1
        stop = True

#  creo una cartella dove salvare i dati in output
if not os.path.exists("./../Output"):
    os.makedirs("./../Output")


Xpixel = 3600
Ypixel = 2048
yyy = []

#disegno un grafico  x: #frame  y: #tiles visti , con errore std. dev.
for i in range(1800):
    y.append(len(listofset[i]))
    yyy.append ((y[i])*192*192/(Xpixel*Ypixel))
print (y)
print(yyy)
mediay = sum (y) / len(y)
sumfordev = 0
for i in range(1800):
    temp = (y[i] - mediay)
    sumfordev = sumfordev + temp*temp
stdev = math.sqrt(sumfordev/1800)
print(stdev)
plt.plot(y)
plt.axis([0, 1800, 0, 225])
plt.xticks([200*k for k in range (10)])
x = list(range(1,1801))
err = [stdev] * 1800
print(x)
plt.errorbar(x, y, yerr=(err,err), ecolor="red")
plt.title("Frame e tiles")
plt.xlabel("# Frame")
plt.ylabel("# Tiles visualizzati nel frame")
plt.savefig("./../Output/Graficoprova.png")
plt.show()
plt.plot(yyy)
plt.show()



#  creo il file con numero di frame, percentuale di visione del frame e i tiles visti, l'ultima riga è il totale di
#  tiles visti nell'insieme del video.
#  La percentuale non è valida se i tiles per frame sono diversi da 200 in totale.

with open("./../Output/all_tile_per_frame.csv", 'w', newline='') as filecsv:
    wr = csv.writer(filecsv)
    wr.writerow(["frame","% tiles visti","tiles"])
    totale = 0
    for x in range(1, 1801):
        setToList = list(listofset[x-1])
        setToList.insert(0, x)
        totale = totale + len(listofset[x-1])
        setToList.insert(1, len(listofset[x-1])/200)
        wr.writerow(setToList)
    wr.writerow(["total that can be " + str(totale/(1800*200))])

