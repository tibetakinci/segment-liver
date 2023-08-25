from PIL import Image
import os, glob

train_dir = r"../datasets/liverUSfiltered/imgs/train/"
val_dir = r"../datasets/liverUSfiltered/imgs/val/"
label_train_dir = r"../datasets/liverUSfiltered/labels/train/"
label_val_dir = r"../datasets/liverUSfiltered/labels/val/"
train_imgs = glob.glob("".join([train_dir, "*.png"]))
val_imgs = glob.glob("".join([val_dir, "*.png"]))
label_train_imgs = glob.glob("".join([label_train_dir, "*.jpg"]))
label_val_imgs = glob.glob("".join([label_val_dir, "*.jpg"]))

def convert_to_jpg(imgs, imgs_dir):
    if not imgs:
        print("No .png files in directory {}".format(imgs_dir))
        pass
    for i in range(len(imgs)):
        img = Image.open(imgs[i])
        rgb_img = img.convert('RGB')
        rgb_img.save(imgs_dir + "{}.jpg".format(imgs[i][len(imgs_dir):-4]))
        os.remove(imgs[i])

def convert_to_png(imgs, imgs_dir):
    if not imgs:
        print("No .jpg files in directory {}".format(imgs_dir))
        pass
    for i in range(len(imgs)):
        img = Image.open(imgs[i])
        rgb_img = img.convert('RGB')
        rgb_img.save(imgs_dir + "{}.png".format(imgs[i][len(imgs_dir):-4]))
        os.remove(imgs[i])

def convert_imgs_set():
    convert_to_jpg(train_imgs , train_dir)
    convert_to_jpg(val_imgs, val_dir)

def convert_labels_set():
    convert_to_png(label_train_imgs, label_train_dir)
    convert_to_png(label_val_imgs, label_val_dir)

if __name__ == "__main__":
    convert_imgs_set()
    convert_labels_set()
