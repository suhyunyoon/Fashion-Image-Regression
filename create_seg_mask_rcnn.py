import cv2
import os
import sys
import numpy as np
import h5py
import skimage
import colorsys
import tensorflow as tf
import numpy as np
import shutil
import random

# git clone https://github.com/matterport/Mask_RCNN.git
# cd /Mask_RCNN/
# python setup.py install
# must install mask-rcnn pkg
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn.model import MaskRCNN


class DefaultConfig(Config):
    NAME = "Deepfashion2"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 13
    BATCH_SIZE = 16

class resultFormat():
    def __init__(self, label, score, position):
        self.label = label
        self.score = score
        self.position = position

class Mask_RCNN():
    def __init__(self, item_dir='./temp/', weight_file='mask_rcnn_deepfashion2_0100.h5'):
        config = DefaultConfig()

        self.item_dir = item_dir
        self.model = modellib.MaskRCNN(mode="inference", config=config, model_dir=self.item_dir)
        self.model.load_weights(self.item_dir + weight_file, by_name=True)

        self.class_names = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear',
                       'vest',
                       'sling',
                       'shorts', 'trousers', 'skirt', 'short_sleeved_dress', 'long_sleeved_dress',
                       'vest_dress', 'sling_dress', '']

        colors = self.random_colors(len(self.class_names))
        self.class_dict = {
            name: color for name, color in zip(self.class_names, colors)
        }

    def random_colors(self, N):
        np.random.seed(1)
        colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
        return colors

    def apply_mask(self, image, mask, color, alpha=0.5):
        """apply mask to image"""
        for n, c in enumerate(color):
            image[:, :, n] = np.where(
                mask == 1,
                image[:, :, n] * (1 - alpha) + alpha * c,
                image[:, :, n]
            )
        return image

    def exec(self, boxes, masks, ids, names, scores):
        n_instances = boxes.shape[0]
        print(n_instances)
        if not n_instances:
            print('no_instances')
        else:
            assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

        for i in range(n_instances):
            if not np.any(boxes[i]):
                continue
            y1, x1, y2, x2 = boxes[i]
            label = names[ids[i]]
            color = self.class_dict[label]
            score = scores[i] if scores is not None else None
            caption = '{} {:.2f}'.format(label, score) if score else label
            classForm = resultFormat(label, score, [x1, y1, x2, y2])
            print(classForm.label, classForm.score, classForm.position)


    def display_instances(self, image, boxes, masks, ids, names, scores):
        n_instances = boxes.shape[0]
        print(n_instances)
        if not n_instances:
            print('no_instances')
        else:
            assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

        for i in range(n_instances):
            if not np.any(boxes[i]):
                continue
            y1, x1, y2, x2 = boxes[i]
            label = names[ids[i]]
            color = self.class_dict[label]
            score = scores[i] if scores is not None else None
            caption = '{} {:.2f}'.format(label, score) if score else label
            classForm = resultFormat(label, score, [x1, y1, x2, y2])
            print(classForm.label, classForm.score, classForm.position)
            mask = masks[:, :, i]
            image = self.apply_mask(image, mask, color)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

        return image

    # Generating masks
    def gen_masks(self, X, img_size):
        image_w, image_h = img_size

        # check X empty
        num_X = len(X)
        self.model.config.BATCH_SIZE = num_X
        if num_X == 0:
            return np.zeros((0, image_w, image_h, 1), dtype=np.bool)
        # detect
        results = self.model.detect(X, verbose=0)
        # results = [{'masks': np.ones((image_w, image_h, 1))}, {'masks': np.ones((image_w, image_h, 0))}, {'masks': np.ones((image_w, image_h, 1))}]
        r = results

        # make mask array (batch_size)
        mask = np.zeros((num_X, image_w, image_h, 1), dtype=np.bool)
        for i, v in enumerate(r):
            if v['masks'].shape[-1] == 0:
                arr = np.zeros((image_w, image_h, 1), dtype=np.bool)
            else:
                #arr = v['masks']
                arr = v['masks'][:,:,:1]
            #print(mask.shape, np.array([arr]).shape)
            mask[i] = arr

        return mask

    def create_seg(self):
        images_hdf5_name = 'images.hdf5'
        masks_hdf5_name = 'masks.hdf5'
        img_size = (224, 224)
        # batch size 문제있음(2 이상으로 하면 output겹쳐서 나옴)
        img_batch = 40000
        batch_size = 1
        with h5py.File(self.item_dir + images_hdf5_name, 'r') as f, h5py.File(self.item_dir + masks_hdf5_name, 'a') as m:
            num_files = len(f['image'])
            m.create_dataset('mask', (num_files, img_size[0], img_size[1], 1), dtype=np.bool)
            cnt = 0
            for index in range(0, num_files, img_batch):
                num_X = len(f[index])
                mask = np.zeros((num_X, img_size[0], img_size[1], 1), dtype=np.bool)
                for i in range(0, num_X, batch_size):
                    # detect
                    x = self.gen_masks(f[index][i:i + batch_size], img_size)
                    mask[i:i + batch_size] = x

                    if i % 100 == 0:
                        print(i, end=' ')
                # save masks
                m['mask'][cnt * img_batch: (cnt + 1) * img_batch] = mask

                cnt += 1
                print('\nmasks {} Done.'.format(cnt))

        print('Done')

if __name__=="__main__":
    model = Mask_RCNN().create_seg()
