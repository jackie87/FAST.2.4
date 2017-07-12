#!/bin/bash
HOME=$(pwd)
echo $HOME > FAST.home

line=$(cat FAST.home)
echo "FAST HOME directory set to $line, please do not remove file FAST.home"

ww=$(echo $line | sed -e 's/\//\\\//g')
echo "Setting FAST home to $ww"
x="s/XXXX/"$ww"/g"
sed -e $x .FAST.utils > FAST.utils.sh
chmod +x FAST.utils.sh
