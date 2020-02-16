from PIL import Image
import os
import numpy as np

root = "E:/Code/torch_learning/yly/data/PennFudanPed/PedMasks"
img_path = os.path.join(root,"FudanPed00001_mask.png")
img=Image.open(img_path)
print(img)
mask = np.array(img)
print(mask)
obj_ids = np.unique(mask)
print(obj_ids)

masks = mask == obj_ids[:,None,None]
print(masks)

obj_ids = obj_ids[1:]

#masks = mask == obj_ids[:,None,None]
#print(masks)

num=len(obj_ids)
print(num)
print(masks[0])
