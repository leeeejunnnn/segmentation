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
image_files = [fname for fname in os.listdir() if os.path.splitext(fname)[-1] == '.jpg']
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
train_images = os.listdir(train_dir)
val_images = os.listdir(val_dir)
print(len(train_images), len(val_images))

# %%
IMG_SIZE = 224
tfr_dir = os.path.join(data_dir, 'tfrecord')
os.makedirs(tfr_dir, exist_ok=True)

tfr_train_dir = os.path.join(tfr_dir, 'cls_train.tfr')
tfr_val_dir = os.path.join(tfr_dir, 'cls_val.tfr')

writer_train = tf.io.TFRecordWriter(tfr_train_dir)
writer_val = tf.io.TFRecordWriter(tfr_val_dir)

# %%
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# %%
n_train = 0 
train_files = os.listdir(train_dir)
for train_file in train_files:
  train_path = os.path.join(train_dir, train_file)
  image = Image.open(train_path)
  image = image.resize((IMG_SIZE, IMG_SIZE))
  bimage = image.tobytes()

  file_name = os.path.splitext(train_file)[0]
  class_name = re.sub('_\d+','',file_name)
  class_num = class2idx[class_name]

  example = tf.train.Example(features=tf.train.Features(feature={
    'image' : _bytes_feature(bimage),
    'cls_num' : _int64_feature(class_num)
  }))
  writer_train.write(example.SerializeToString())
  n_train += 1
writer_train.close()
print(n_train)

# %%
n_val = 0 
val_files = os.listdir(val_dir)
for val_file in val_files:
  val_path = os.path.join(val_dir, val_file)
  image = Image.open(val_path)
  image = image.resize((IMG_SIZE, IMG_SIZE))
  bimage = image.tobytes()

  file_name = os.path.splitext(val_file)[0]
  class_name = re.sub('_\d+','',file_name)
  class_num = class2idx[class_name]

  example = tf.train.Example(features=tf.train.Features(feature={
    'image' : _bytes_feature(bimage),
    'cls_num' : _int64_feature(class_num)
  }))
  writer_val.write(example.SerializeToString())
  n_val += 1
writer_val.close()
print(n_val)

# %%
N_CLASS = len(class_list)
N_EPOCHS = 20
N_BATCH = 40
N_TRAIN = n_train
N_VAL = n_val
IMG_SIZE = 224
learning_rate = 0.0001
steps_per_epoch = N_TRAIN // N_BATCH
validation_steps = int(np.ceil(N_VAL / N_BATCH)) # 소수점 올림


# %%
def _pars_function(tfrecord_serialized):
  features = {'image': tf.io.FixedLenFeature([],tf.string),
              'cls_num': tf.io.FixedLenFeature([], tf.int64)
              }
  parsed_features = tf.io.parse_single_example(tfrecord_serialized, features)

  image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
  image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 3])
  image = tf.cast(image, tf.float32)/255
  
  label = tf.cast(parsed_features['cls_num'], tf.int64)

  return image, label

# %%
train_dataset = tf.data.TFRecordDataset(tfr_train_dir)
train_dataset = train_dataset.map(_pars_function)
train_dataset = train_dataset.shuffle(N_TRAIN).prefetch(
  tf.data.experimental.AUTOTUNE).batch(N_BATCH).repeat()

val_dataset = tf.data.TFRecordDataset(tfr_val_dir)
val_dataset = val_dataset.map(_pars_function)
val_dataset = val_dataset.batch(N_BATCH).repeat()

# %%
for image, label in train_dataset.take(5):
  plt.imshow(image[0])
  title = class_list[label[0].numpy()]
  plt.title(title)
  plt.show()

# %%
