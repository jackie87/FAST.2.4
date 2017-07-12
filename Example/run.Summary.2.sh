#!/bin/bash
# This script will run FAST in mode=summary with LD computed on the fly.
# The input is from files in DATA.summary
# The output is stored in OUT

time ../FAST \
       --summary-file ./DATA.summary/chr1.summary \
       --multipos-file ./DATA.summary/chr1.multiPos.txt \
       --hap-file ./DATA.summary/chr1.hap \
       --pos-file ./DATA.summary/chr1.hap.ra \
       --out-file ./OUT/output5 \
       --gene-set ./DATA.summary/genes.txt \
       --chr 1 \
       --pheno-var 0.1875 \
       --n-sample 2000 \
       --mode summary \
       --maf-cutoff 0 \
       --max-perm 100000\
       --linear-minsnp-perm \
       --linear-minsnp-gene-perm \
       --linear-vegas-perm \
       --linear-gates \
       --linear-gwis-perm \
       --linear-bf-perm \
       --flank 5 \
       --n-threads 1
       
