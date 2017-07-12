#!/bin/bash
# This script will run FAST in mode=genotype with linear model.
# The input is from files in DATA.geno
# The output is stored in OUT

time ../FAST \
       --impute2-geno-file ./DATA.geno/chr1.impute2.geno.gz \
       --impute2-info-file ./DATA.geno/chr1.impute2.info.gz \
       --indiv-file ./DATA.geno/id.txt.gz \
       --out-file ./OUT/output.impute2 \
       --trait-file ./DATA.geno/CC.tfam.gz \
       --gene-set ./DATA.geno/genes1.txt.gz \
       --chr 1 \
       --maf-cutoff 0.01 \
       --mode genotype \
       --max-perm 100 \
       --linear-bf-perm \
       --linear-minsnp-perm \
       --linear-minsnp-gene-perm \
       --linear-vegas-perm \
       --linear-gwis-perm \
       --linear-gates \
       --flank 5 \
       --n-threads 1
  

