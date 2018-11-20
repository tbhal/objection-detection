# objection-detection
Object detetction on custom dataset using Tensorflow object detection API

## Installation
Type git clone https://github.com/tensorflow/models.git in the shell it will clone the TensorFlow API or it can be downloaded from [here](https://github.com/tensorflow/models)

## Steps to be followed

- First we create csv file for the images and xml file by using the xml_to_cs. First switch to that directory then type
'''python
python xml_to_csv.py
'''
- Then we have to create trf records
'''python
python generate_trfrecord.py
'''
- Then decide the model type which we have to use for training I used ssd_mobilnet_v1_coco_2017_11_17

-Then we created training folder with object-detection.pbtx and config file

-The after that we have to paste all these folder in model/research/object_detection folder.

- Then swithc to the legacy directory and run
'''python
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
'''
-The training process graph can be seen on Tensorboard using command in shell
'''python
tensorboard --logdir='training'
'''
