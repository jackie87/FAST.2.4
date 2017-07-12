#!/bin/bash
PREF=$1
MODEL=$2
ID=$3
OUT_GW=$PREF.combined.$ID.GWiS.$MODEL.txt
OUT_V=$PREF.combined.$ID.Vegas.$MODEL.txt
OUT_BF=$PREF.combined.$ID.BF.$MODEL.txt
OUT_G=$PREF.combined.$ID.Gates.$MODEL.txt
OUT_MS=$PREF.combined.$ID.minSNP.$MODEL.txt
OUT_MSG=$PREF.combined.$ID.minSNP_Gene.$MODEL.txt
OUT_ALL=$PREF.combined.$ID.allSNP.$MODEL.txt

#touch $OUT
#cat /dev/null > $OUT

flag_GW=
flag_V=
flag_BF=
flag_G=
flag_MS=
flag_MSG=
flag_ALL=

for chr in {1..23}
do
  #echo $PREF.chr$chr.GWiS.$MODEL.txt
  if [ -f $PREF.chr$chr.GWiS.$MODEL.txt ]
  then
     if [[ -z $flag_GW ]]
     then
        flag_GW=1
        cp $PREF.chr$chr.GWiS.$MODEL.txt $OUT_GW 
     else
        sed '1d' $PREF.chr$chr.GWiS.$MODEL.txt >> $OUT_GW
     fi
  fi
 
  if [ -f $PREF.chr$chr.Vegas.$MODEL.txt ]
  then
     if [[ -z $flag_V ]]
     then
        flag_V=1
        cp $PREF.chr$chr.Vegas.$MODEL.txt $OUT_V 
     else
        sed '1d' $PREF.chr$chr.Vegas.$MODEL.txt >> $OUT_V
     fi
  fi 

  if [ -f $PREF.chr$chr.BF.$MODEL.txt ]
  then
     if [[ -z $flag_BF ]]
     then
        flag_BF=1
        cp $PREF.chr$chr.BF.$MODEL.txt $OUT_BF
     else
        sed '1d' $PREF.chr$chr.BF.$MODEL.txt >> $OUT_BF
     fi
  fi
 
  if [ -f $PREF.chr$chr.Gates.$MODEL.txt ]
  then
     if [[ -z $flag_G ]]
     then
        flag_G=1
        cp $PREF.chr$chr.Gates.$MODEL.txt $OUT_G 
     else
        sed '1d' $PREF.chr$chr.Gates.$MODEL.txt >> $OUT_G
     fi
  fi
 
  if [ -f $PREF.chr$chr.minSNP.$MODEL.txt ]
  then
     if [[ -z $flag_MS ]]
     then
        flag_MS=1
        cp $PREF.chr$chr.minSNP.$MODEL.txt $OUT_MS
     else
        sed '1d' $PREF.chr$chr.minSNP.$MODEL.txt >> $OUT_MS
     fi
  fi
 
  if [ -f $PREF.chr$chr.minSNP_Gene.$MODEL.txt ]
  then
     if [[ -z $flag_MSG ]]
     then
        flag_MSG=1
        cp $PREF.chr$chr.minSNP_Gene.$MODEL.txt $OUT_MSG
     else
        sed '1d' $PREF.chr$chr.minSNP_Gene.$MODEL.txt >> $OUT_MSG
     fi
  fi 

  if [ -f $PREF.chr$chr.allSNP.$MODEL.txt ]
  then
     if [[ -z $flag_ALL ]]
     then
        flag_ALL=1
        cp $PREF.chr$chr.allSNP.$MODEL.txt $OUT_ALL
     else
        sed '1d' $PREF.chr$chr.allSNP.$MODEL.txt >> $OUT_ALL
     fi
  fi 
done
