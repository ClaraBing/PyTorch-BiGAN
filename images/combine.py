import os
import numpy as np
import cv2
from glob import glob

import pdb

fdir = 'e200_unnorm_sigma0.1'
frows = glob(os.path.join(fdir, '*iter200.png'))
print('# images:', len(frows))

img_h, img_w = 480, 640
pad_h = 10
pad_w = 10

n_img_per_row = 16
rows = [[]]
for fimg in frows:
  if len(rows[-1]) == n_img_per_row:
    rows += [],
  rows[-1] += cv2.imread(fimg),

n_dummy = n_img_per_row - len(rows[-1])
rows[-1] += [np.ones([img_h, img_w, 3]) * 255 for _ in range(n_dummy)]

trim_l, trim_r = 140, 100
trim_u, trim_d = 20, 20
img_rows = []
for row in rows:
  row = [img[trim_u:-trim_d, trim_l:-trim_r] for img in row]
  img_row = np.concatenate(row, 1)
  img_rows += img_row,
image = np.concatenate(img_rows, 0)
cv2.imwrite(os.path.join(fdir, 'combined.png'), image)



