import subprocess as sp
import numpy as np
import math


##################################
#####  TILEFICATION SUPPORT  #####
##################################

Xpixel = 900                    # Number of pixels in axis X
Ypixel = 450                    # Number of pixels in axis Y
NTilesHorizontal = 10           # Number of tiles we want on axis X
NTilesVertical = 5              # Number of Tiles we want on axis Y

NTotTiles = NTilesVertical*NTilesHorizontal
NPixelTileHorizontal = Xpixel//NTilesHorizontal
NPixelTileVertical = Ypixel//NTilesVertical

# Pick an array of some tiles, and two matrix as support for tilefication
RandomTiles = np.array([1,10,15,17,18,20,30,45,49])
SupportMatrix = np.zeros((NPixelTileHorizontal, NPixelTileVertical,3), dtype ='bool')
MatrixSeen = np.ones((Xpixel, Ypixel,3), dtype='bool')


###########################
#####  FLOWS OPENING  #####
###########################

commandin = [ "ffmpeg",
            '-i', 'rollercoaster.mp4',          # File name
            '-f', 'image2pipe',                 # Output file format
            '-pix_fmt', 'rgb24',                # Pixel format
            '-vcodec', 'rawvideo', '-']         # Codec

pipe = sp.Popen(commandin, stdout = sp.PIPE, bufsize = Xpixel*Ypixel*3*60)

commandout = ["ffmpeg",
           '-y',                                # Overwrite output file if it exists
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-s', '900x450',                     # Size of one frame
           '-pix_fmt', 'rgb24',                 # Pixel format
           '-r', '30',                          # Frames per second
           '-i', '-',                           # Input comes from a pipe
           '-an',                               # No audio
           '-vcodec', 'mpeg4',                  # Codec
           'my_output_videofile.mp4']           # File name in output

pipe2 = sp.Popen(commandout, stdin=sp.PIPE, stderr=sp.PIPE, bufsize = Xpixel*Ypixel*3*60)


################################################
#####  READ - MODIFY/TILEFICATION - WRITE  #####
################################################

for i in range(1800):

    ## Read a frame from Input video
    raw_image = pipe.stdout.read(Xpixel*Ypixel*3)
    image = np.fromstring(raw_image, dtype='uint8')
    image = image.reshape((Ypixel,Xpixel,3))
    image = np.rot90(image, k=-1)


    ## Tilefication
    for j in range (RandomTiles.size):
        startingX = (RandomTiles[j]%NTilesHorizontal)*NPixelTileHorizontal
        startingY = math.fabs((RandomTiles[j]//NTilesHorizontal)-(NTilesVertical-1))*NPixelTileVertical
        image [startingX : startingX+NPixelTileHorizontal , int(startingY) : int(startingY+NPixelTileVertical)] = SupportMatrix


    ## Write a frame in output
    pipe2.stdin.write(np.rot90(image).tobytes())







