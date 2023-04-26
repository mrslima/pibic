# Scripts

## Multilabel Segmentation

```python
# MODIFICAR OS PIXELS DE UMA IMAGEM

import numpy as np
from PIL import Image
import os


from_dir = 'multilabel_segmentation/labels/'
save_to_dir = 'masks/'


def changeImgPixels(from_dir, save_to_dir):
    
    img_arr = os.listdir(from_dir)
    
    for img in img_arr:
        
        img_path = from_dir + img

        img = Image.open(img_path)
        
        img_arr = np.array(img)
        
        img_arr[img_arr == 1] = 1
        img_arr[img_arr == 2] = 1
        img_arr[img_arr == 3] = 1
        img_arr[img_arr == 4] = 1
        
        img_modified = Image.fromarray(img_arr)
        
        if not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)
        
        img_name = img_path[len(from_dir):]
        output =  save_to_dir + img_name
        img_modified.save(output)


changeImgPixels(from_dir, save_to_dir)

```

## Datasets
**SegTHOR**
- [25% do dataset](https://drive.google.com/file/d/13z6cT0g5XB05fIJxLmj3WmRd7j7LNBFB/view?usp=share_link)
- [50% do dataset](https://drive.google.com/file/d/1aVysbkX4SfzN-61yH51TsNjelBJiGqIg/view?usp=share_link)

---

```python
from PIL import Image
import os

path = 'meio_dataset/images/'
move_to = 'meio_dataset/images/'
files = os.listdir(path)


for file in files:
    # if file[-1:] == 'g':
        # os.remove(move_to + file)
    print(move_to + file)
    image = Image.open(path + file)
    image.save(move_to + file[:-4] + ".tiff")
```

```python
import os, shutil

path = 'labels/'
move_to = 'trash/'
files = os.listdir(path)


for file in files:
    if 'color' in file:
        src = path + file
        dst = move_to + file
        shutil.move(src, dst)
```

```python
# pip install tifftools
    
import tifftools

currentFileName = "masks_as_128x128_patches.tif"
currentRevisedFullFileName = "128_patches/" + currentFileName
info = tifftools.read_tiff(currentRevisedFullFileName)
for i, ifd in enumerate(info['ifds']):
  tifftools.write_tiff(ifd, '128_patches/masks/img%s.tif'%(i,))
  

# OR

from PIL import Image
import os

path = 'images/'
move_to = 'images/'
files = os.listdir(path)


for file in files:
    # if file[-1:] == 'g':
        # os.remove(move_to + file)
    print(move_to + file)
    # image = Image.open(path + file)
    # image.save(move_to + file[:-4] + ".tiff")
```

```python
from PIL import Image
import os
import math
from random import randint
import time



image_dir = 'images/'
label_dir = 'labels/'
train_masks = 'train/masks/'
train_images = 'train/images/'
test_masks = 'test/masks/'
test_images = 'test/images/'
val_masks = 'val/masks/'
val_images = 'val/images/'


def train_test_val_splitter(image_dir, label_dir,
                            train_masks, train_images, 
                            test_masks, test_images, 
                            val_images, val_masks,
                            train_qtty=0.7, test_qtty=0.2, val_qtty=0.1):
    start = time.time()
    
    image_files = os.listdir(image_dir)
    label_files = os.listdir(label_dir)
    
    while True:
        seed = randint(0, math.floor(len(image_files)*train_qtty)-len(os.listdir(train_images)))
        actual_image = image_files[seed]
        actual_label = label_files[seed]
        
        image = Image.open(image_dir + actual_image)
        image.save(train_images + actual_image)
        
        label = Image.open(label_dir + actual_label)
        label.save(train_masks + actual_label)
        
        if len(os.listdir(train_images)) == math.floor(len(image_files)*train_qtty):
            break
    
    while True:
        seed = randint(0, math.floor(len(image_files)*test_qtty)-len(os.listdir(test_images)))
        actual_image = image_files[seed]
        actual_label = label_files[seed]
        
        image = Image.open(image_dir + actual_image)
        image.save(test_images + actual_image)
        
        label = Image.open(label_dir + actual_label)
        label.save(test_masks + actual_label)
        
        if len(os.listdir(test_images)) == math.floor(len(image_files)*test_qtty-len(os.listdir(val_images))):
            break
    
    while True:
        seed = randint(0, math.floor(len(image_files)*val_qtty))
        actual_image = image_files[seed]
        actual_label = label_files[seed]
        
        image = Image.open(image_dir + actual_image)
        image.save(val_images + actual_image)
        
        label = Image.open(label_dir + actual_label)
        label.save(val_masks + actual_label)
        
        if len(os.listdir(val_images)) == math.floor(len(image_files)*val_qtty):
            break
    
    end = time.time()
    print('Elapsed time:', end-start)

    

train_test_val_splitter(image_dir, label_dir,
                            train_masks, train_images, 
                            test_masks, test_images, 
                            val_images, val_masks,
                            train_qtty=0.7, test_qtty=0.2, val_qtty=0.1)
```
