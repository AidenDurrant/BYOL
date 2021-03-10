#!/bin/bash

rm log_files.txt

python src/pretrain.py -c ~/Documents/BYOL/config.conf
python src/finetune.py -c ~/Documents/BYOL/config.conf
python src/test_finetune.py -c ~/Documents/BYOL/config.conf
