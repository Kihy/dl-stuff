import os
image_dir="dataset/testFire"
out_dir="dataset/testFire/ImageSet"

out_file=open(os.path.join(out_dir,"names.txt"),"w")
for i in os.listdir(image_dir):
    if os.path.isfile(os.path.join(image_dir,i)):
        out_file.write(i+"\n")

out_file.close()
