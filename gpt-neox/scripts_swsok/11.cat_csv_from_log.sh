#!/bin/bash

if [ -z $1 ]; then
	echo "Usage: $0 [log_file]"
	exit 0
fi

if [ ! -f $1 ]; then
	echo "$1 is missing."
	exit 0
fi

#grep -a lm_loss $1 | awk '{printf "%s,%s,%s,%f\n",substr($8, 1, length($8)-1),$2,substr($26,1,length($26)-6),$29}' | tee $1.csv
grep -a lm_loss $1 | awk '{printf "%s,%f\n",$2,$5}' | tee $1.csv
grep -a lm_loss $1 | awk '{printf "%s,%s,%f\n",substr($8,1,length($8)-1),substr($26,1,length($26)-6),$35}' | tee $1.csv
