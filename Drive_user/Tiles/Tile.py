import csv
import os
import glob


listOfFile = glob.glob("./*.csv")
listofset = []
stop = False


#  leggo i vari file e salvo tutti i tiles riga per riga in una lista di set per non avere doppioni.
#  Valido per qualsiasi tipo di file con le caratteristiche appropiate
for x in range(len(listOfFile)):
    with open(listOfFile[x], newline="", encoding="ISO-8859-1") as filecsv:
        lettore = csv.reader(filecsv,delimiter=",")
        header = next(lettore)
        y=0
        while True:
            if not stop:
                listofset.append(set())
            try:
                riga = next(lettore)
            except:
                break
            for z in range(1,(len(riga))):
                listofset[y].add(riga[z])
            y=y+1
        stop = True

#  creo una cartella dove salvare i dati in output
if not os.path.exists("allData"):
    os.makedirs("allData")


#  creo il file con numero di frame, percentuale di visione del frame e i tiles visti, l'ultima riga è il totale di
#  tiles visti nell'insieme del video.
#  La percentuale non è valida se i tiles per frame sono diversi da 200 in totale.

with open("./allData/all_tile_per_frame.csv", 'w', newline='') as filecsv:
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

