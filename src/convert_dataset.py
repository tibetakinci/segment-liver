from PIL import Image
import os, glob

train_dir = r"../datasets/liverUS/imgs/train/"
val_dir = r"../datasets/liverUS/imgs/val/"
train_imgs = glob.glob(train_dir + "*.png")
val_imgs = glob.glob(val_dir + "*.png")

def convert_trainset():
    for i in range(len(train_imgs)):
        img = Image.open(train_imgs[i])
        rgb_img = img.convert('RGB')
        rgb_img.save(train_dir + "{}.jpg".format(train_imgs[i][31:-4]))
        os.remove(train_imgs[i])

def convert_valset():
    for i in range(len(val_imgs)):
        img = Image.open(val_imgs[i])
        rgb_img = img.convert('RGB')
        rgb_img.save(val_dir + "{}.jpg".format(val_imgs[i][29:-4]))
        os.remove(val_imgs[i])

def convert_png_to_jpg():
    convert_trainset()
    convert_valset()

if __name__ == "__main__":
    convert_png_to_jpg()