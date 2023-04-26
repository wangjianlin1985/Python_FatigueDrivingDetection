import os
import cv2
from functools import partial
from scipy.misc import imresize
from PIL import Image


data_set = "genki4k/files/"
output_dir = "genki4k/files_crip/"


harr_model_path = 'face_det/data/haarcascades'
frontal_model= os.path.join(harr_model_path, 'haarcascade_frontalface_default.xml')
profile_model = os.path.join(harr_model_path, 'haarcascade_profileface.xml')

# 正脸检测的模型
frontal_dector = partial(cv2.CascadeClassifier(frontal_model).detectMultiScale,
                         scaleFactor=1.1,
                         minNeighbors=5,
                         minSize=(100, 100))

# 侧脸检测的模型
profile_dector = partial(cv2.CascadeClassifier(profile_model).detectMultiScale,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(100, 100))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def extract_face(image, save_path):
    image = cv2.imread(image)
    faces = frontal_dector(image)
    for (x, y, z, w) in faces:
        cv2.rectangle(image, (x,y), (x+z, y+w), (0, 255, 0), 1)
        save_img = image[y:y+w,x:x+z]
        
        cv2.imwrite(save_path, save_img)
        save_img = Image.open(save_path)
        save_img = save_img.resize((64,64))
        save_img = save_img.convert('L')
        save_img.save(save_path)

imgs = os.listdir(data_set)
for img in imgs:
    extract_face(data_set+img, output_dir+img)
