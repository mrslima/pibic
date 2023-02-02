# Scripts


```
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

```
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

```
# pip install tifftools
    
import tifftools

currentFileName = "masks_as_128x128_patches.tif"
currentRevisedFullFileName = "128_patches/" + currentFileName
info = tifftools.read_tiff(currentRevisedFullFileName)
for i, ifd in enumerate(info['ifds']):
  tifftools.write_tiff(ifd, '128_patches/masks/img%s.tif'%(i,))
```
