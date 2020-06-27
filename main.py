#%%
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import re
from PIL import Image
import shutil
import random
import matplotlib.pyplot as plt

print(tf.__version__)
print(keras.__version__)

#%% #Data
import gdown
url = "https://drive.google.com/uc?id=1dIR9ANjUsV9dWa0pS9J0c2KUGMfpIRG0"
fname = 'oxford_pet.zip'
gdown.download(url, fname, quiet=False)

#%%
!unzip -q oxford_pet.zip -d oxford_pet

!ls oxford_pet
#%%
cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'oxford_pet')
image_dir = os.path.join(data_dir, 'images')
#%%
image_files = [fname for fname in os.listdir(image_dir) if os.path.splitext(fname)[-1] == '.jpg']
print(len(image_files))
#%%
for image_file in image_files:
  image_path = os.path.join(image_dir, image_file)
  image = Image.open(image_path)
  image_mode = image.mode
  if image_mode != 'RGB':
    print(image_file, image_mode)
    image = np.asarray(image)
    print(image.shape)
    os.remove(image_path)

#%%
image_files = [fname for fname in os.listdir(image_dir) if os.path.splitext(fname)[-1] == '.jpg']
print(len(image_files))

class_list = set() #중복된 항목은 더하지 않는 구조 -> set
for image_file in image_files:
  file_name = os.path.splitext(image_file)[0]
  class_name = re.sub('_\d+','',file_name)
  class_list.add(class_name)
class_list = list(class_list)
print(len(class_list))
#%%
class_list.sort()
class_list
#%%
class2idx= {cls : idx for idx, cls in enumerate(class_list)}
class2idx
#%%
class2idx['Bengal']
#%%
#train, validation dir 
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir,'validation')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
#%%
image_files.sort()
image_files[:10]
#%%
cnt = 0
previous_class = ""
for image_file in image_files:
  file_name = os.path.splitext(image_file)[0]
  class_name = re.sub('_\d+','', file_name)
  if class_name == previous_class :
    cnt += 1
  else:
    cnt = 1
  if cnt <= 160:
    cpath = train_dir
  else:
    cpath = val_dir
  image_path = os.path.join(image_dir, image_file)
  shutil.copy(image_path, cpath)
  previous_class = class_name

# %%
