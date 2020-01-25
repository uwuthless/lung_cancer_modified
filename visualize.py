import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

sys.path.append('../training/')
sys.path.append('../')
sys.path.append('../preprocessing/')
from layers import nms, iou

img = np.load('./1_clean.npy')
pbb = np.load('./1_pbb.npy')

pbb = pbb[pbb[:, 0] > -1]

pbb = nms(pbb, 0.05)
box = pbb[0].astype('int')[1:]

ax = plt.subplot(1, 1, 1)
plt.imshow(img[0, box[0]], 'gray')
plt.axis('off')
rect = patches.Rectangle((box[2] - box[3], box[1] - box[3]), box[3] * 2, box[3] * 2, linewidth=2, edgecolor='red',
                         facecolor='none')
ax.add_patch(rect)
