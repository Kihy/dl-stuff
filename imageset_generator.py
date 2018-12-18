from os import listdir, mkdir, rename
from os.path import join, splitext, isdir, isfile
import random


# Path to dataset folder
data_folder="dataset/fire_1720"

folder_names=["JPEGImages", "ImageSets", "Annotations","ImageSets/Main"]
# make folders
for name in folder_names:
    folder=join(data_folder,name)
    if not isdir(folder):
        print("creating directory: "+folder)
        mkdir(folder)

# move file to right place
for filename in listdir(data_folder):
    if isfile(join(data_folder,filename)):
        if filename.endswith(".jpg"):
            rename(join(data_folder,filename),join(data_folder,"JPEGImages",filename))

        elif filename.endswith(".xml"):
            rename(join(data_folder,filename),join(data_folder,"Annotations", filename))

# Path to all images files
IMAGE_PATH = join(data_folder, "JPEGImages")
# Path for the output of the image sets
IMAGESET_PATH = join(data_folder, "ImageSets/Main")
# Split Percentage
train_prob = 0.8
val_prob = 0.1
test_prob = 0.1
# Seeding
random.seed(1)

filenames = [splitext(f)[0] for f in listdir(IMAGE_PATH)]

# Calculate number of files for each
num_file = len(filenames)
train_cutoff = int(train_prob * num_file)
val_cutoff = int((train_prob + val_prob) * num_file)

random.shuffle(filenames)

# open files
train_file = open(join(IMAGESET_PATH, "train.txt"), "w")
test_file = open(join(IMAGESET_PATH, "test.txt"), "w")
val_file = open(join(IMAGESET_PATH, "val.txt"), "w")

train_list = filenames[:train_cutoff]
val_list = filenames[train_cutoff:val_cutoff]
test_list = filenames[val_cutoff:]

train_file.write("\n".join(train_list))
test_file.write("\n".join(test_list))
val_file.write("\n".join(val_list))

train_file.close()
test_file.close()
val_file.close()

print("#train: {}, #val: {}, #test:{}".format(len(train_list), len(val_list), len(test_list)))
