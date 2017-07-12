This ultility helps to convert files in plink TPED format to FAST format.
**** You must provide PLINK tped files, NOT ped files as input ****
**** See http://pngu.mgh.harvard.edu/~purcell/plink/data.shtml#tr ****

To compile, type "make clean" and then "make".

A small example is provided here, the data can be found in the "data" directory.
Run as : ./plink2fast data/chr1 
This will create input files in FAST format in "data" directory.

If you get the error "cannot execute binary file", please compile as shown above. 
