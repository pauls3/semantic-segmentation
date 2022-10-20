
import os
import json

from config import cfg
from runx.logx import logx
from datasets.base_loader import BaseLoader
from datasets.utils import make_dataset_folder
from datasets import uniform


class Loader(BaseLoader):
    num_classes = 19
    ignore_label = 255
    trainid_to_name = {}
    color_mapping = []

    def __init__(self, mode, quality=None, joint_transform_list=None,
                 img_transform=None, label_transform=None, eval_folder=None):

        super(Loader, self).__init__(quality=quality,
                                     mode=mode,
                                     joint_transform_list=joint_transform_list,
                                     img_transform=img_transform,
                                     label_transform=label_transform)

        root = cfg.DATASET.RTISRAIL22
        # config_fn = os.path.join(root, '/home/stanik/rtis_lab/data/RailSem19/rs19-config.json')
        config_fn = '/home/stanik/rtis_lab/data/rtis-rail-2022v2/class_to_id_new.json'
        self.fill_colormap_and_names(config_fn)

        ######################################################################
        # Assemble image lists
        ######################################################################
        # if mode == 'folder':
        #     self.all_imgs = make_dataset_folder(eval_folder)
        # else:
        if mode != 'folder':
            splits = {'train': 'training',
                    'val': 'validation',
                    'test': 'testing'}
            split_name = splits[mode]
            img_ext = 'jpg'
            mask_ext = 'png'

        # print(----------------------------------------)
        # print(mode)
        # print(----------------------------------------)

            if mode == 'train':
                img_root = os.path.join(root, 'trainVal_images')
                mask_root = os.path.join(root, 'trainVal_masks')
            elif mode == 'val':
                img_root = os.path.join(root, 'val_images')
                mask_root = os.path.join(root, 'val_masks')
            else:
                img_root = os.path.join(root, 'test_images')
                mask_root = os.path.join(root, 'test_masks')

            print(img_root)

        # img_root = os.path.join(root, split_name, 'images')
        # mask_root = os.path.join(root, split_name, 'labels')
        
            self.all_imgs = self.find_images(img_root, mask_root, img_ext,
                                mask_ext)
        # if mode == 'folder':
        #     self.all_imgs = make_dataset_folder(eval_folder)
        # else:
        #     self.all_imgs = self.find_images(img_root, mask_root, img_ext,
        #                                  mask_ext)

        else:
            self.all_imgs = make_dataset_folder(eval_folder)

        logx.msg('all imgs {}'.format(len(self.all_imgs)))
        self.centroids = uniform.build_centroids(self.all_imgs,
                                                 self.num_classes,
                                                 self.train,
                                                 cv=cfg.DATASET.CV)
        self.build_epoch()

    def fill_colormap_and_names(self, config_fn):
        """
        Mapillary code for color map and class names

        Outputs
        -------
        self.trainid_to_name
        self.color_mapping
        """
        with open(config_fn) as config_file:
            config = json.load(config_file)
        config_labels = config['labels']

        # calculate label color mapping
        colormap = []
        self.trainid_to_name = {}
        for i in range(0, len(config_labels)):
            colormap = colormap + config_labels[i]['color']
            name = config_labels[i]['name']
            name = name.replace(' ', '_')
            self.trainid_to_name[i] = name
        self.color_mapping = colormap