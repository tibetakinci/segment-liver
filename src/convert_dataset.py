from PIL import Image
import os, glob

train_dir = r"../datasets/liverUS/imgs/train/"
val_dir = r"../datasets/liverUS/imgs/val/"
train_imgs = glob.glob("".join([train_dir, "*.png"]))
val_imgs = glob.glob("".join([val_dir, "*.png"]))

def convert_trainset():
    if not train_imgs:
        print("No .png files in image set")
        pass
    for i in range(len(train_imgs)):
        print("control1")
        img = Image.open(train_imgs[i])
        rgb_img = img.convert('RGB')
        rgb_img.save(train_dir + "{}.jpg".format(train_imgs[i][len(train_dir):-4]))
        os.remove(train_imgs[i])

def convert_valset():
    if not val_imgs:
        print("No .png files in validation set")
        pass
    for i in range(len(val_imgs)):
        print("control2")
        img = Image.open(val_imgs[i])
        rgb_img = img.convert('RGB')
        rgb_img.save(val_dir + "{}.jpg".format(val_imgs[i][len(val_dir):-4]))
        os.remove(val_imgs[i])

def convert_png_to_jpg():
    convert_trainset()
    convert_valset()

if __name__ == "__main__":
    convert_png_to_jpg()