# What this is about

This work is an assignment submitted in partial fulfillment of the requirements for the course of Wireless Networks (MSc degree), Department of Mathematics, University of Padua, Italy. The objective is the analysis of a VR dataset provided in [1], where the authors make available some user-centric metadata for different 360°, immersive scenarios. 

In particular, we are interested in computing the amount of bandwidth a Mobile Augmented Reality (MAR) service could save over the air when considering the aggregate users' field of views (FoVs). To this end, different methods are proposed and analyzed: (i) a "tile" method whereby the video is divided into a number of rectangular areas (tiles) of size 192x192 pixels and (ii) an ad-hoc method which (ideally) computes a circle enclosing all users FoVs. The tiles falling inside a user FoV and the area of the enclosing circle, respectively, comprise the actual data sent over the air.


[1] W.-C. Lo et. al, "360° Video Viewing Dataset in Head-Mounted Virtual Reality", in Proc. of the ACM Multimedia Systems Conference (MMSys'17). ACM, New York, NY, USA, pp. 211-216, 2017. 

# Project structure

Code: code written in python, used to compute bandwidth saving amounts for the different methods

Dataset: a reduced version of the dataset containing user orientation data (50 users) and stats on the observed tiles

Output: bandwidth savings for in each of the considered scenarios.
	nameVideo_method_type.extension


# Result overview

Table 1 provides a summary of the actual amount of data that could be sent over the air for each of the considered scenarios and saving functions.

| Video  | Tile | Ad hoc |
| ------------- | ----------- |  ----------- |
| Mega Coaster | 0.7854 | 0.7990 |
| Roller Coaster | 0.82132 | 0.8415 | 
| Driving with | 0.8960 | 0.8617 |
| Shark Shipwreck | 0.9670 | 0.9656 |
| Perils Panel | 0.8441 | 0.8152 |
| Kangaroo Island | 0.8645 | 0.8359 | 
| SFR Sport | 0.7698 | 0.7993 |
| Hog Rider | 0.7607 | 0.7870 |
| Pac-Man | 0.6812 | 0.7466 |
| Chariot Race | 0.8111 | 0.8263 |
