import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from keras.models import load_model
import numpy as np
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

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model=load_model('saved_models/ssd300_fire_2018-12-13_13:39.h5', custom_objects={'AnchorBoxes': AnchorBoxes, 'L2Normalization': L2Normalization, 'SSDLoss': SSDLoss, 'compute_loss':ssd_loss.compute_loss})

img_height = 512 # Height of the model input images
img_width = 512 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
normalize_coords=True
n_classes = 1 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
mean_color=[0,0,0]
scales = [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets

aspect_ratios = [[1.0, 2.0, 0.5],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                [1.0, 2.0, 0.5],
                [1.0, 2.0, 0.5]]

steps = [8, 16, 32, 64, 128, 256, 512] # The space between two adjacent anchor box center points for each predictor layer.


two_boxes_for_ar1 = True
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path='hdf5/dataset_fire_test_512.h5')

train_dataset=DataGenerator(load_images_into_memory=False, hdf5_dataset_path='hdf5/dataset_fire_train_512.h5')

# The directories that contain the images.
fire_img= 'dataset/fire/JPEGImages'

# The directories that contain the annotations.
fire_anno = 'dataset/fire/Annotations/'

fire_val  = 'dataset/fire/ImageSets/Main/val.txt'
fire_train='dataset/fire/ImageSets/Main/train.txt'
classes = ['background', 'fire']



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
                      exclude_difficult=True,
                      ret=False)

# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)


# For the validation Generator
resize = Resize(height=img_height, width=img_width)
convert_to_3_channels = ConvertTo3Channels()

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv10_2_mbox_conf').output_shape[1:3]
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

# 1: Set the generator for the predictions.

predict_generator = val_dataset.generate(batch_size=val_dataset.get_dataset_size(),
                                         shuffle=True,
                                         transformations=[convert_to_3_channels,
                                         resize],
                                         label_encoder=None,
                                         returns={'processed_images',
                                         'filenames',
                                         'inverse_transform',
                                         'original_images',
                                         'original_labels'},
                                         keep_images_without_gt=False)
train_batch_size=10
data_generator = train_dataset.generate(batch_size=train_batch_size,
                                  shuffle=False,
                                  transformations=[ssd_data_augmentation],
                                  label_encoder=None,
                                  returns={'processed_images',
                                  'processed_labels',
                                  'filenames',
                                  'original_images',
                                  'original_labels'},
                                  keep_images_without_gt=False)

# 2: Generate samples.

batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)

for i in range(val_dataset.get_dataset_size()):
    print("Image:", batch_filenames[i])
    print()
    print("Ground truth boxes:\n")
    print(np.array(batch_original_labels[i]))
    
    
    # 3: Make predictions.
    
    y_pred = model.predict(batch_images)
    
    # 4: Decode the raw predictions in `y_pred`.
    
    y_pred_decoded = decode_detections(y_pred,
                                     
                                       confidence_thresh=0.3,
                                       iou_threshold=0.4,
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
    colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
    
    plt.figure(figsize=(20,12))
    plt.imshow(batch_original_images[i])
    
    current_axis = plt.gca()
    
    for box in batch_original_labels[i]:
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        label = '{}'.format(classes[int(box[0])])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})
    
    
    for box in y_pred_decoded_inv[i]:
        
        xmin = box[2]
        ymin = box[3]
        xmax = box[4]
        ymax = box[5]
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
    plt.savefig('saved_figures/{}'.format(batch_filenames[i].split("/")[-1]))

#input samples

processed_images, processed_annotations, filenames, original_images, original_annotations = next(data_generator)

for i in range(train_batch_size):
    colors = plt.cm.hsv(np.linspace(0, 1, len(classes))).tolist() # Set the colors for the bounding boxes

    fig, cell = plt.subplots(1, 2, figsize=(20,16))
    cell[0].imshow(original_images[i])
    cell[1].imshow(processed_images[i])

    for box in original_annotations[i]:
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        color = colors[int(box[0])]
        label = '{}'.format(classes[int(box[0])])
        cell[0].add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
        cell[0].text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
                                        
    
    for box in processed_annotations[i]:
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        color = colors[int(box[0])]
        label = '{}'.format(classes[int(box[0])])
        cell[1].add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
        cell[1].text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

    plt.savefig('saved_figures/input/{}'.format(filenames[i].split("/")[-1]))
