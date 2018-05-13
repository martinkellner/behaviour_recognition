from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model

import numpy as np
import time
import os

class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
            base_model = InceptionV3(
                weights='imagenet',
                include_top=True
            )

            # We'll extract features at the final pool layer.
            self.model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('avg_pool').output
            )

        else:
            # Load the model first.
            self.model = load_model(weights)

            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features

start_time = time.time()
dataset_directory_path = '../data/dataset/'
features_directory_path = '../data/features/'
data_dir = os.walk(dataset_directory_path)
first_skip = True

model = Extractor()

for dir in data_dir:
    # Skip header
    if (first_skip):
        first_skip = False
        continue

    sequences_right = []
    sequences_left  = []

    if os.path.isfile(features_directory_path + dir[0][16:] + "_right.npy"):
        print ("Skip: " + features_directory_path + dir[0][16:] + "_right.npy " + "already exists!")
        continue
    else:
        print("Exists")

    if os.path.isfile(features_directory_path + dir[0][16:] + "_left.npy"):
        print ("Skip: File " + features_directory_path + dir[0][16:] + "_left.npy " + "already exists!")
        continue

    for file in dir[2]:
        # Path to the image
        filepath = dir[0] + '/' + file
        # Extract features from ImageNet
        features = model.extract(filepath)
        # print(filepath, " DONE!", end=' ')

        if (file[0]=='r'):
            # Store features extracted from a image captured by RIGHT camera
            sequences_right.append(features)
            # print ('--storing to left array')
        elif (file[0]=='l'):
             # Store features extracted from a image captured by LEFT camera
             sequences_left.append(features)
             # print ('--storing to right array')

    np.save(features_directory_path + dir[0][16:] + "_right", sequences_right)
    np.save(features_directory_path + dir[0][16:] + "_left", sequences_left)

print("Delay: ", time.time() - start_time)