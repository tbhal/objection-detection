# objection-detection
Object detetction on custom dataset using Tensorflow object detection API

## Installation
Type `git clone https://github.com/tensorflow/models.git` in the shell it will clone the TensorFlow API or it can be downloaded from [here](https://github.com/tensorflow/models)

## Steps to be followed

- First we create csv file for the images and xml file by using the xml_to_cs. First switch to that directory then type

`python xml_to_csv.py`

- Then we have to create trf records for that run 
`python generate_trfrecord.py`


- Then decide the model type which we have to use for training I used ssd_mobilnet_v1_coco_2017_11_17

- Then we created training folder with object-detection.pbtx and config file

- The after that we have to paste all these folder in **model/research/object_detection folder**.

- Then switch to the legacy directory and run

`python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config`

- The training process graph can be seen on Tensorboard using command in shell

`tensorboard --logdir='training'`
- After this for **Inference Graph** execute the command
`python3 export_inference_graph.py \`
    `--input_type image_tensor \`
    `--pipeline_config_path training/ssd_mobilenet_v1_pets.config \`
    `--trained_checkpoint_prefix training/model.ckpt-(version of your 		model created) \`
    `--output_directory fruits_inference_graph`

- Type this command when an import error occurs in **model/research** directory `export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim` error like this
No module named 'nets'

### Testing the model

- First paste some images in folder of *test_images* in **models/research/object_detection/** directory label them image3,4,5,6.jpg

- Now open the ipynb file for tutorial in the same directory and make these changes

```python
# What model to download.
MODEL_NAME = 'fruit_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1
```
- Delete he code from download image section or do not reun it.

- And in Detection Section, change the TEST_IMAGE_PATHS var to:
```python
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3, 8) ]
```
- Then in Cell option menu click on "Run All"

The result will be images with bounding box

## Error
Still I am getting this error and still unable to resolve it
ValueError: Tensor conversion requested dtype string for Tensor with dtype float32: 'Tensor("arg0:0", shape=(), dtype=float32, device=/device:CPU:0)'

