import matplotlib

matplotlib.use("Agg")
import os
import numpy as np
from ssd512_train import training_preprocessing, val_preprocessing
from configparser import ConfigParser, ExtendedInterpolation

from matplotlib import pyplot as plt

from json import loads
from keras.models import load_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

#remove previous figures
for i in os.listdir("saved_figures/custom"):
    os.remove(os.path.join("saved_figures/custom", i))


parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read("model_config.ini")
params = parser["ssd512_custom"]

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model = load_model(params["model_path"],
                   custom_objects={'AnchorBoxes': AnchorBoxes, 'L2Normalization': L2Normalization, 'SSDLoss': SSDLoss,
                                   'compute_loss': ssd_loss.compute_loss})

classes = (loads(params["classes"]))

img_height = int(params["image_height"])  # Height of the model input images
img_width = int(params["image_width"])  # Width of the model input images
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

# The directories that contain the images.
fire_img = params["image_path"]

# The directories that contain the annotations.
# fire_annotation = params["annotations"]
#
# The paths to the image sets.
fire_imagesets = params["image_sets"]
file_names=params["image_names"]
classes = (loads(params["classes"]))

test_dataset = DataGenerator(load_images_into_memory=False,
                             images_dir=fire_img, filenames=file_names)

# 1: Set the generator for the predictions.

predict_generator = test_dataset.generate(batch_size=test_dataset.get_dataset_size(),
                                          shuffle=True,
                                          transformations=val_preprocessing(img_height, img_width),
                                          label_encoder=None,
                                          returns={'processed_images',
                                                   'filenames',
                                                   'inverse_transform',
                                                   'original_images',
                                                   },
                                          keep_images_without_gt=True)

# 2: Generate samples.
test_dataset_size = test_dataset.get_dataset_size()

print("Number of images in the validation dataset:\t{:>6}".format(test_dataset_size))

batch_images, batch_filenames, batch_inverse_transforms, batch_original_images = next(
    predict_generator)

for i in range(test_dataset.get_dataset_size()):
    print("Image:", batch_filenames[i])

    # 3: Make predictions.

    y_pred = model.predict(batch_images)

    # 4: Decode the raw predictions in `y_pred`.

    y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=float(params["confidence_thresh"]),
                                       iou_threshold=float(params["iou_thresh"]),
                                       top_k=200,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)

    # 5: Convert the predictions for the original image.

    y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_decoded_inv[i])

    # 5: Draw the predicted boxes onto the image

    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, n_classes + 1)).tolist()

    plt.figure(figsize=(20, 12))
    plt.imshow(batch_original_images[i])

    current_axis = plt.gca()


    for box in y_pred_decoded_inv[i]:
        xmin = box[2]
        ymin = box[3]
        xmax = box[4]
        ymax = box[5]
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
    plt.savefig('saved_figures/custom/{}'.format(batch_filenames[i].split("/")[-1]))

