#!/bin/bash
usage()
{
cat << EOF

usage: $0 options
This script runs the FAST Utils

OPTIONS:
   -h                                                                               No arguments, just show this message
   -p <prefix-to-file> <type>                                                       QQPlot with 2 arguments, do QQ plots for each method
   -r <prefix-to-file> <type> <pvalue-cutoff-for-gene> < pvalue-cutoff-for-SNP>     GetReport with 4 arguments, print combined report

   type = Linear | Logistic | Summary
EOF
}

FAST_HOME=/users/jzhan/AllFasts/FAST.2.4.mc
echo "Home=$FAST_HOME"

QQPLOT=
REPORT=
VERBOSE=
while getopts “hp:r:v” OPTION
do
     case $OPTION in
         h)
             usage
             exit 1
             ;;
         p)
             QQPLOT=$OPTARG
             ;;
         r)
             REPORT=$OPTARG
             ;;
         v)
             VERBOSE=1
             ;;
         ?)
             usage
             exit
             ;;
     esac
done

if [[ -z $QQPLOT ]] && [[ -z $REPORT ]] 
then
     echo "Either option -p or -r need to be provided"
     usage
     exit 1
fi

if [[ -z $QQPLOT ]]
then
  echo " "
else
  PREF=$2
  MODEL=$3
  if [[ -z $PREF ]]
  then 
     echo "QQPlot option <prefix-to-file> not set"
     usage
     exit 1
  fi 
  if [[ -z $MODEL ]]
  then 
     echo "QQPlot option <type> not set"
     usage
     exit 1
  fi
  echo "To run QQPlot with files : $PREF*$MODEL.txt" 
  R CMD BATCH --slave --no-restore  "--args prefix=\"$PREF\" model=\"$MODEL\" " $FAST_HOME/Utils/QQ/Plot.QQ.r qqplot.log.txt
fi
   
if [[ -z $REPORT ]]
then
  echo " "
else
  PREF=$2
  MODEL=$3
  GP=$4  
  SP=$5  
  if [[ -z $PREF ]]
  then 
     echo "GetReport 1st argument <prefix-to-file> not set"
     usage
     exit 1
  fi 
  if [[ -z $MODEL ]]
  then 
     echo "GetReport 2nd argument <type> not set"
     usage
     exit 1
  fi
  if [[ -z $GP ]]
  then 
     echo "GetReport 3rd argument <pvalue-cutoff-for-gene> not set, using default = 0.0001"
     GP=0.0001
  elif ! [[ "$GP" =~ ^[0-9]+([.][0-9]+)?$  ]]
  then
     echo "GetReport 3rd argument <pvalue-cutoff-for-gene> is not a number !!!"
     usage
     exit 1
  fi
  if [[ -z $SP ]]
  then 
     echo "GetReport 4th argument <pvalue-cutoff-for-SNP> not set, using default = 0.0001"
     SP=0.0001
  elif ! [[ "$SP" =~ ^[0-9]+([.][0-9]+)?$  ]]
  then
     echo "GetReport 4th argument <pvalue-cutoff-for-snp> is not a number !!!"
     usage
     exit 1
  fi
  echo "To run GetReport with files : $PREF*$MODEL.txt"
  ID=$$ 
  $FAST_HOME/Utils/GETREPORT/combine.sh $PREF $MODEL $ID
  $FAST_HOME/Utils/GETREPORT/getReport.pl $PREF.combined.$ID $MODEL $GP $SP
  rm $PREF.combined.$ID*
fi

