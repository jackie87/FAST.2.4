#!/bin/bash
# This script will run FAST in mode=genotype with logistic model.
# The input is from files in DATA.geno
# The output is stored in OUT

time ../FAST \
       --tped-file ./DATA.geno/chr1.tped.gz \
       --snpinfo-file ./DATA.geno/chr1.snp.info.gz \
       --mlinfo-file ./DATA.geno/chr1.mlinfo.gz \
       --indiv-file ./DATA.geno/id.txt.gz \
       --out-file ./OUT/output3 \
       --trait-file ./DATA.geno/CC.tfam.gz \
       --gene-set ./DATA.geno/genes1.txt.gz \
       --chr 1 \
       --maf-cutoff 0 \
       --mode genotype \
       --max-perm 100 \
       --logistic-bf-perm \
       --logistic-minsnp-perm \
       --logistic-minsnp-gene-perm \
       --logistic-vegas-perm \
       --logistic-gwis-perm \
       --logistic-gates \
       --flank 5 \
       --n-threads 1
