from configparser import ConfigParser, ExtendedInterpolation
from json import loads
import os
from datetime import datetime
from math import ceil

import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, CSVLogger, TensorBoard
from keras.optimizers import Adam

from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from keras_loss_function.keras_ssd_loss import SSDLoss
from models.keras_ssd512 import ssd_512
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder

# limit gpu usage
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session = tf.Session(config=config)

# parse args
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("model_config.ini")
params = parser["ssd512_train"]

classes = (loads(params["classes"]))

img_height = int(params["image_height"])  # Height of the model input images
img_width = int(params["img_width"])  # Width of the model input images
img_channels = int(params["img_channels"])  # Number of color channels of the model input images
# The per-channel mean of the images in the dataset.
# Do not change this value if you're using any of the pre-trained weights.
mean_color = loads(params["mean_color"])
# The color channel order in the original SSD is BGR,
# so we'll have the model reverse the color channel order of the input images.
swap_channels = loads(params["swap_channels"])
# Number of positive classes
n_classes = len(classes) - 1
# The anchor box scaling factors used in the original SSD512 for the Pascal VOC datasets
scales_pascal = loads(params["swap_channels"])

scales = loads(params["scales"])
aspect_ratios = loads(params["aspect_ratios"])

two_boxes_for_ar1 = bool(params["two_boxes_for_ar1"])
# The space between two adjacent anchor box center points for each predictor layer.
steps = loads(params["steps"])
# The offsets of the first anchor box center points from the top and left borders of the image
# as a fraction of the step size for each predictor layer.
offsets = loads(params["offsets"])
# Whether or not to clip the anchor boxes to lie entirely within the image boundaries
clip_boxes = bool(params["clip_boxes"])
# The variances by which the encoded target coordinates are divided as in the original implementation
variances = loads(params["variances"])
normalize_coords = bool(params["normalize_coords"])

# 1: Build the Keras model.

K.clear_session()  # Clear previous models from memory.

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
weights_path = params["weights_path"]

model.load_weights(weights_path, by_name=True)

# 3: Instantiate an optimizer and the SSD loss function and compile the model.
#    If you want to follow the original Caffe implementation, use the preset SGD
#    optimizer, otherwise I'd recommend the commented-out Adam optimizer.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

# Optional: If you have enough memory, consider loading the images into memory or use hdf5 dataset.

train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

# 2: Parse the image and label lists for the training and validation datasets. This can take a while.

# TODO: Set the paths to the datasets here.

# The directories that contain the images.
fire_img = params["image_path"]

# The directories that contain the annotations.
fire_annotation = params["annotations"]

# The paths to the image sets.
fire_imagesets = params["image_sets"]
fire_train = os.path.join(fire_imagesets, 'train.txt')
fire_test = os.path.join(fire_imagesets, 'test.txt')
fire_val = os.path.join(fire_imagesets, 'val.txt')

train_dataset.parse_xml(images_dirs=[fire_img],
                        image_set_filenames=[fire_train],
                        annotations_dirs=[fire_annotation],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)

val_dataset.parse_xml(images_dirs=[fire_img],
                      image_set_filenames=[fire_val],
                      annotations_dirs=[fire_annotation],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=False,
                      ret=False)

# Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
# speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
# option in the constructor, because in that cas the images are in memory already anyway. If you don't
# want to create HDF5 datasets, comment out the subsequent two function calls.

train_dataset.create_hdf5_dataset(file_path=params["hdf5_train_path"],
                                  resize=False,
                                  variable_image_size=True,
                                  verbose=True)

val_dataset.create_hdf5_dataset(file_path=params["hdf5_test_path"],
                                resize=False,
                                variable_image_size=True,
                                verbose=True)

# 3: Set the batch size.

batch_size = int(params["batch_size"])  # Change the batch size if you like, or if you run into GPU memory issues.

# 4: Set the image transformations for pre-processing and data augmentation options.

# For the training generator:
def training_preprocessing():
    ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                                img_width=img_width,
                                                background=mean_color)
    return [ssd_data_augmentation]


# For the validation generator:
def val_preprocessing():
    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=img_height, width=img_width)
    return [convert_to_3_channels,resize]

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
                   ]

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
                                         transformations=training_preprocessing(),
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=val_preprocessing(),
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

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

# TODO: Set the file path under which you want to save the model.
current_time = datetime.now().strftime('%Y-%m-%d %H:%M').split(" ")
model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(params["checkpoint_path"], 'ssd512_fire_{}_{}.h5'.format(current_time[0], current_time[1])),
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=10)
# model_checkpoint.best =

csv_logger = CSVLogger(
    filename=os.path.join(params["log_path"], 'ssd512_fire_{}_{}.csv'.format(current_time[0], current_time[1])),
    separator=',',
    append=True)

# learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule, verbose=1)

terminate_on_nan = TerminateOnNaN()

tensorboard = TensorBoard(log_dir=os.path.join(params["tensorboard_path"], 'ssd512', current_time[0], current_time[1]),
                          write_images=True, write_graph=True)

callbacks = [model_checkpoint,
             csv_logger,
             #            learning_rate_scheduler,
             terminate_on_nan,
             tensorboard]
# Fit model

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch = 0
final_epoch = int(params["epoch"])
steps_per_epoch = int(params["steps_per_epoch"])

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size / batch_size),
                              initial_epoch=initial_epoch)
