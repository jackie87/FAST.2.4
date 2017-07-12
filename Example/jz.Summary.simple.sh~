#!/bin/bash
# This script will run FAST in mode=summary with LD computed on the fly and basic set of inputs.
# The input is from files in DATA.summary
# The output is stored in OUT

time ../FAST \
       --summary-file ./DATA.summary/chr1.simple \
       --hap-file ./DATA.summary/chr1.hap \
       --pos-file ./DATA.summary/chr1.hap.ra \
       --out-file ./OUT/output6 \
       --gene-set ./DATA.summary/genes.txt \
       --chr 1 \
       --mode summary \
       --maf-cutoff 0 \
       --max-perm 100000 \
       --n-sample 2000 \
       --logistic-minsnp-perm \
       --logistic-minsnp-gene-perm \
       --logistic-vegas-perm \
       --logistic-gates \
       --logistic-gwis-perm \
       --logistic-bf-perm \
       --flank 5 \
       --n-threads 1
       
