#!/bin/bash

# run.sh

python data/babi/prep.py --task-id 1
python data/babi/prep.py --task-id 2
# python data/babi/prep.py --task-id 6
python data/babi/model.py

# python data/babi/prep.py --task-id 1

