#!/bin/bash
for f in *.sh
do sbatch $f
done
