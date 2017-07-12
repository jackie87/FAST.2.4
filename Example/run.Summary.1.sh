#!/bin/bash
# This script will run FAST in mode=summary with pre-computed ld files.
# The input is from files in DATA.summary
# The output is stored in OUT

time ../FAST \
       --summary-file ./DATA.summary/chr1.summary \
       --multipos-file ./DATA.summary/chr1.multiPos.txt \
       --ld-file ./DATA.summary/chr1.ld \
       --allele-file ./DATA.summary/chr1.allele.info \
       --out-file ./OUT/output4 \
       --gene-set ./DATA.summary/genes.txt \
       --chr 1 \
       --pheno-var 0.1875 \
       --n-sample 2000 \
       --mode summary \
       --maf-cutoff 0 \
       --compute-ld 0 \
       --max-perm 100000\
       --linear-minsnp-perm \
       --linear-minsnp-gene-perm \
       --linear-vegas-perm \
       --linear-gates \
       --linear-gwis-perm \
       --linear-bf-perm \
       --flank 5 \
       --n-threads 1
       
