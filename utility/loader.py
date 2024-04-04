import torch 
import json 
import os 
import numpy as np
import h5py
from PIL import Image
import io
import re

from torch.utils.data import Dataset

class EpicKitchenLoader(Dataset):

    def __init__(self, feature_folder, json_file, split, num_frames, feat_stride, default_fps, num_classes):

        self.feature_folder = feature_folder
        self.json_file = json_file
        self.split = split
        self.num_frames = num_frames
        self.feat_stride = feat_stride
        self.default_fps = default_fps
        self.num_classes = num_classes
        self.label_dict = None
        self.feature_stride = 8

        dict_db, label_dict = self._load_json_db(self.json_file)
        self.label_dict = label_dict

        empty_label_ids = self.find_empty_cls(self.label_dict, num_classes)
        
        self.dict_db = dict_db
        self.label_dict = label_dict
 
        self.data_list = self.dict_db
        self.db_attributes = {
            'dataset_name': 'epic-kitchens-100',
            'tiou_thresholds': np.linspace(0.1, 0.5, 5),
            'empty_label_ids': empty_label_ids
        }



    def find_empty_cls(self, label_dict, num_classes):
        # find categories with out a data sample
        if len(label_dict) == num_classes:
            return []
        empty_label_ids = []
        label_ids = [v for _, v in label_dict.items()]
        for id in range(num_classes):
            if id not in label_ids:
                empty_label_ids.append(id)
        return empty_label_ids

   
    def _load_json_db(self, json_file):

        # load database and select the subset

        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available

        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']

        # fill in the db (immutable afterwards)
        
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue
        

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                num_acts = len(value['annotations'])
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(value['annotations']):
                    segments[idx][0] = act['segment'][0]
                    segments[idx][1] = act['segment'][1]
                    labels[idx] = label_dict[act['label']]
            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        return dict_db, label_dict
    
    def load_images_from_hdf5(self, input_hdf5_file):

        images_dict = {}
        with h5py.File(input_hdf5_file, 'r') as hf:
            for key in hf.keys():
                 images_dict[key] = Image.open(io.BytesIO(np.array(hf[key])))
        return images_dict
   
    def load_video_from_directory(self, directory):

        stacked_images = []
        for filename in os.listdir(directory):
            if filename.endswith(".h5"):
                file_path = os.path.join(directory, filename)
                images_dict = self.load_images_from_hdf5(file_path)
                for key, value in images_dict.items():
                    stacked_images.append(value)
        stacked_tensor = torch.tensor(stacked_images)  # Directly stack the images into a tensor
        return stacked_tensor
    
    def load_video_path_from_directory(self, directory):
        paths = []
        for filename in os.listdir(directory):
            if filename.endswith(".h5"):
                file_path = f'{directory}/{filename}'
                paths.append(file_path)
        return paths

    def extract_last_number(self, filename):
        return int(re.search(r'\d+', filename.split('_')[-1]).group())


    def __getitem__(self, idx):      

        video_item = self.data_list[idx]  #lista P01_01, P01_03, ecc..
        video_id = video_item['id']
        participant = video_id.split('_')[0]
        video_id = video_id.split('_')[1]
        print(participant, video_id)

        video_folder = f'{self.feature_folder}/{participant}/{participant}_{video_id}'

        video_tensor = self.load_video_path_from_directory(video_folder)

        sorted_file_paths = sorted(video_tensor, key= self.extract_last_number)

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                (video_item['segments'] * video_item['fps']- 0.5 * self.num_frames) / self.feat_stride
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : sorted_file_paths,      # T 
                     'segments'        : segments,   #  N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : self.feat_stride,
                     'feat_num_frames' : self.num_frames}
  
        return  data_dict   



    def __len__(self):
        return len(self.data_list)

