#!/bin/bash

grep lm_loss logs/gptneox-test_stdout.txt | awk '{print $26}'

#watch -n 10 "grep lm_loss gptneox-test_stdout.txt | awk '{print \$5\$6\" \"\$26}' | tail -n 10"
#grep lm_loss gptneox-test_stdout.txt | awk '{print $5 $6 " " $23"/222.2TFLOPS"  " " $26}'
