from keras import backend as K
from keras.models import load_model, Model
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np
from json import loads
import os
from matplotlib import pyplot as plt
from ssd512_train import training_preprocessing, val_preprocessing
from configparser import ConfigParser, ExtendedInterpolation

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator


parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("model_config.ini")
params = parser["ssd512_eval"]
# Set a few configuration parameters.

classes = (loads(params["classes"]))

img_height = int(params["image_height"])  # Height of the model input images
img_width = int(params["image_width"])  # Width of the model input images
img_channels = int(params["img_channels"])  # Number of color channels of the model input images

# Number of positive classes
n_classes = len(classes) - 1

model_mode = 'inference'

# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = params["model_path"]

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})
m_input = model.input
m_output = model.output
decoded_predictions = DecodeDetections(confidence_thresh=float(params["confidence_thresh"]),
                                       iou_threshold=float(params["iou_threshold"]),
                                       top_k=200,
                                       nms_max_output_size=400,
                                       coords='centroids',
                                       normalize_coords=True,
                                       img_height=img_height,
                                       img_width=img_width,
                                       name='decoded_predictions')(m_output)

model = Model(inputs=m_input, outputs=decoded_predictions)


# TODO: Set the paths to the dataset here.
fire_dataset_images_dir = params["image_path"]
fire_dataset_annotations_dir = params["annotations"]
fire_dataset_image_set = os.path.join(params["image_sets"], 'test.txt')

dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=params["hdf5_test_path"],
                             images_dir=fire_dataset_images_dir, filenames=fire_dataset_image_set)

evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)
                      
results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=8,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

print("evaluating model: {} at {} confidence threshold and {} iou threshold".format(params["model_path"],params["iou_thresh"],params["confidence_thresh"]))

mean_average_precision, average_precisions, precisions, recalls = results

for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
    print()
    print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))

m = max((n_classes + 1) // 2, 2)
n = 2

fig, cells = plt.subplots(m, n, figsize=(n*8,m*8))
for i in range(m):
    for j in range(n):
        if n*i+j+1 > n_classes: break
        cells[i, j].plot(recalls[n*i+j+1], precisions[n*i+j+1], color='blue', linewidth=1.0)
        cells[i, j].set_xlabel('recall', fontsize=14)
        cells[i, j].set_ylabel('precision', fontsize=14)
        cells[i, j].grid(True)
        cells[i, j].set_xticks(np.linspace(0,1,11))
        cells[i, j].set_yticks(np.linspace(0,1,11))
        cells[i, j].set_title("{}, AP: {:.3f}".format(classes[n*i+j+1], average_precisions[n*i+j+1]), fontsize=16)


