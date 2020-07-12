#pip install -q -U "tensorflow-gpu==2.0.0b1"
#pip install -q -U tensorflow_hub

import sys
import time 
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds


batch_size = 32
image_size = 224

class_names = {}


# parse arguments
parser = argparse.ArgumentParser ()
parser.add_argument ('arg1', default='./test_images/hard-leaved_pocket_orchid.jpg', help = 'Image Path', type = str)
parser.add_argument('arg2', help='Point to checkpoint file as str.', type=str)
parser.add_argument ('--top_k', default = 5, help = 'Top K classes.', type = int)
#parser.add_argument ('--category_names' , default = 'label_map.json', help = 'Mapping of labels to real names.', type = str)
parser.add_argument ('--category_names' , help = 'Mapping of labels to real names.', type = str)
args = parser.parse_args()
print(args)

image_path = args.arg1
export_path_keras = args.arg2
classes = args.category_names
top_k = args.top_k

# import model
model = tf.keras.experimental.load_from_saved_model(export_path_keras, custom_objects={'KerasLayer':hub.KerasLayer})


if top_k is None: 
    top_k = 5

# Import Label Mapping
if classes is None:
    class_names = None
else:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)

            
def preprocess_image(image): 
    
    image = tf.cast(image, tf.float32)
    image= tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return image


def predict(image_path, model, top_k=5):
    
    # open and preprocess image
    image = Image.open(image_path)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_image(image)
    prob_list = model.predict(image)
   
    # get top k values
    top_values, top_indices = tf.nn.top_k(input=prob_list, k=top_k, sorted=True)
    print("These are the top propabilities",top_values.numpy())#[0])
    if class_names == None:
        top_classes = top_indices.numpy()[0]+1#[str(value) for value in top_indices.cpu().numpy())#[0]]
    else:
        #top_classes = [class_names[str(top_indices.numpy())]]
        top_classes = [class_names[str(value)] for value in top_indices.numpy()[0]+1]
    print('Of these top classes', top_classes)
    return top_values.numpy()[0], top_classes#.numpy()[0]


probs, classes = predict(image_path, model, top_k)
    