#!/bin/bash

for def_prob in .25 .35 .45 .55 .65 .75 .85 .95
do
    sbatch submit-python-job.sh ${def_prob}
done
