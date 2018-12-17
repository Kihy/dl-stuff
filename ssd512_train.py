import tensorflow as tf
import argparse
from datetime import datetime
import os

from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger, TensorBoard
from keras import backend as K
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

# limit gpu usage
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.8
set_session=tf.Session(config=config)

# parse args
parser= argparse.ArgumentParser()
parser.add_argument("--image_height", help="Height of the model input images", default=512, type=int )
parser.add_argument("--image_width", help="width of the model input images", default=512, type=int )
parser.add_argument("--weights_path", help="Path to pretrained weights", default='weights/VGG_ILSVRC_16_layers_fc_reduced.h5' )
parser.add_argument("--images", help="Path to all images", default='dataset/fire/JPEGImages' )
parser.add_argument("--annotations", help="Path to all annotations", default='dataset/fire/Annotations' )
parser.add_argument("--image_sets", help="Path to imagesets, the image sets must be called train.txt, test.txt and val.txt stored in the path", default='dataset/fire/ImageSets/Main' )
parser.add_argument("--classes", help="Classes for object detection, seperated by ',' ", default="fire")
parser.add_argument("--batch_size", help="Batch size of for training ", default=16, type=int)
parser.add_argument("--checkpoint_path", help="Path to store checkpoints", default='saved_models' )
parser.add_argument("--log_path", help="Path to log files", default='training_log' )
parser.add_argument("--tensorboard_path", help="Path to tensorboard logs", default='tensorboard' )
parser.add_argument("--epoch", help="number of epochs", default=1000, type=int)
parser.add_argument("--steps_per_epoch", help="steps per epoch", default=35, type=int)
args = parser.parse_args()

classes=['background']
classes.extend(args.classes.replace(' ','').split(","))

img_height = args.image_height # Height of the model input images
img_width = args.image_width # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [0,0 , 0] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = len(classes)-1 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05] # The anchor box scaling factors used in the original SSD512 for the Pascal VOC datasets
scales_coco = [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06] # The anchor box scaling factors used in the original SSD512 for the MS COCO datasets
scales = scales_coco
aspect_ratios = [[1.0, 2.0, 0.5],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5],
                [1.0, 2.0, 0.5]]


two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 128, 256, 512] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True


# 1: Build the Keras model.

K.clear_session() # Clear previous models from memory.

model = ssd_512(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels)


# 2: Load some weights into the model.

# TODO: Set the path to the weights you want to load.
# all weights stored under the weights directory
weights_path = args.weights_path

model.load_weights(weights_path, by_name=True)

# 3: Instantiate an optimizer and the SSD loss function and compile the model.
#    If you want to follow the original Caffe implementation, use the preset SGD
#    optimizer, otherwise I'd recommend the commented-out Adam optimizer.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

# Optional: If you have enough memory, consider loading the images into memory or use hdf5 dataset.

train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

# 2: Parse the image and label lists for the training and validation datasets. This can take a while.

# TODO: Set the paths to the datasets here.

# The directories that contain the images.
fire_img= args.images

# The directories that contain the annotations.
fire_anno = args.annotations

# The paths to the image sets.
fire_train  = os.path.join(args.image_sets, 'train.txt')
fire_test = os.path.join(args.image_sets, 'test.txt')
fire_val  = os.path.join(args.image_sets, 'val.txt')

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
# classes = ['background', 'fire']

train_dataset.parse_xml(images_dirs=[fire_img],
                        image_set_filenames=[fire_train],
                        annotations_dirs=[fire_anno],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)
val_dataset.parse_xml(images_dirs=[fire_img],
                      image_set_filenames=[fire_val],
                      annotations_dirs=[fire_anno],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=False,
                      ret=False)


# Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
# speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
# option in the constructor, because in that cas the images are in memory already anyway. If you don't
# want to create HDF5 datasets, comment out the subsequent two function calls.

train_dataset.create_hdf5_dataset(file_path='hdf5/dataset_fire_train_512.h5',
                                  resize=False,
                                  variable_image_size=True,
                                  verbose=True)

val_dataset.create_hdf5_dataset(file_path='hdf5/dataset_fire_test_512.h5',
                                resize=False,
                                variable_image_size=True,
                                verbose=True)

# 3: Set the batch size.

batch_size = args.batch_size # Change the batch size if you like, or if you run into GPU memory issues.

# 4: Set the image transformations for pre-processing and data augmentation options.

# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv10_2_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# Define a learning rate schedule.

def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001

# Define model callbacks.

# TODO: Set the filepath under which you want to save the model.
current_time = datetime.now().strftime('%Y-%m-%d %H:%M').split(" ")
model_checkpoint = ModelCheckpoint(filepath=os.path.join(args.checkpoint_path,'ssd512_fire_{}_{}.h5'.format(current_time[0],current_time[1])),
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=10)
#model_checkpoint.best = 

csv_logger = CSVLogger(filename=os.path.join(args.log_path,'ssd512_fire_{}_{}.csv'.format(current_time[0],current_time[1])),
                       separator=',',
                       append=True)

#learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule, verbose=1)

terminate_on_nan = TerminateOnNaN()

tensorboard=TensorBoard(log_dir=os.path.join(args.tensorboard_path,'ssd512',current_time[0],current_time[1]), write_images=True, write_graph=True)

callbacks = [model_checkpoint,
             csv_logger,
 #            learning_rate_scheduler,
             terminate_on_nan,
             tensorboard]
# Fit model

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch   = 0
final_epoch     = args.epoch
steps_per_epoch = args.steps_per_epoch

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch, 
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)

