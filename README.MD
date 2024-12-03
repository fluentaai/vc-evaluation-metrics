## VC-EVALUATION-METRICS

This repository contains scripts with metrics for evaluating VC systems. 

Steps to setup environment:

1. to build docker: sudo docker compose -f docker-compose.yml up --build --force-recreate
2. change $WORK_DIR variable in env/.env file according to actual path to this repository on your machine
3. [download WESPEAKER model](https://wenet.org.cn/downloads?models=wespeaker&version=voxblink2_samresnet34.zip) and 
then unzip it into ```./models/wespeaker/wespeaker-SimAMResNet34``` directory
4. [download WAVLM model](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view?usp=sharing) and then 
put it into ```./models/UniSpeech``` directory

Steps to evaluate VC system:

1. 

