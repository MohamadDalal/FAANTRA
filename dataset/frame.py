#!/usr/bin/env python3

"""
File containing classes related to the frame datasets.
"""

#Standard imports
from util.io import load_json
import os
import random
import numpy as np
import copy
import torch
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm
import pickle
import math

#Local imports


#Constants

# Pad the start/end of videos with empty frames
DEFAULT_PAD_LEN = 5
FPS_SN = 25 # Remember to change value in util.io if you change it here

# Main dataset for training and validation
class ActionSpotDataset(Dataset):

    def __init__(
            self,
            classes,                        # dict of class names to idx
            label_file,                     # path to label json
            frame_dir,                      # path to frames
            store_dir,                      # path to store files (with frames path and labels per clip)
            store_mode,                     # 'store' or 'load'
            clip_len,                       # Number of frames per clip
            dataset_len,                    # Number of clips
            label_pad_idx,                  # The value to pad with. Pads both labels and offsets
            n_class,                        # Number of classes including the NONE class. Doesn't seem to be used
            stride=1,                       # Downsample frame rate
            overlap=1,                      # Overlap between clips (in proportion to clip_len)
            excluded_classes = [],          # List of classes to be excluded when creating labels
            radi_smoothing=0,               # Radius of label smoothing in observation labels
            pad_len=DEFAULT_PAD_LEN,        # Number of frames to pad the start
                                            # and end of videos
            dataset = 'finediving',         # Dataset name
            obs_perc=[0.2, 0.3, 0.5],       # Observation percentage
            pred_perc=0.5,                  # Prediction percentage
            n_query=30,                     # Number of queries in model
            anticipate_background = False,  # Anticipate background class instead of EOS
            use_actionness = False,         # Use actionness instead of EOS
            use_anchors = False,            # Use temporal anchors for the model
            cheating_dataset = False,       # Cheating dataset that gives model anticipation frames instead of observed frames
            cheating_range = None           # Range of video to provide when cheating
            # TODO: Add a specific observation percentage when doing test dataset
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._split = label_file.split('/')[-1].split('.')[0]
        self._class_dict = classes
        self._n_class = n_class
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._dataset = dataset
        self._excluded_classes = excluded_classes
        self._store_dir = store_dir
        self._store_mode = store_mode
        assert store_mode in ['store', 'load']
        self._clip_len = clip_len
        assert clip_len > 0
        self._stride = stride
        assert stride > 0
        if overlap != 1:
            self._overlap = int((1-overlap) * clip_len)
        else:
            self._overlap = 1
        assert overlap >= 0 and overlap <= 1
        self._dataset_len = dataset_len
        assert dataset_len > 0
        self._pad_len = 0 if self._dataset == 'soccernetballanticipation' else pad_len  # Makes no sense to pad video since each video is a single clip and each clip is only 750 frames
        assert pad_len >= 0

        self._obs_perc = obs_perc
        self._pred_perc = pred_perc
        self._label_pad_idx = label_pad_idx
        self._n_query = n_query
        self._anticipate_background = anticipate_background
        self._use_actionness = use_actionness
        self._use_anchors = use_anchors
        self._cheating_dataset = cheating_dataset
        self._cheating_range = cheating_range
        assert not (anticipate_background and use_actionness), "Cannot anticipate background and use actionness at the same time"
        if use_anchors: assert (use_anchors and (anticipate_background or use_actionness)), "Cannot use anchors with EOS, needs to use background or actionness"
        self.NONE = 0

        # Label modifications
        self._radi_displacement = 0     # Removed via hard coding
        self._radi_smoothing = radi_smoothing     

        #Frame reader class
        self._frame_reader = FrameReader(frame_dir, dataset = dataset)

        #Variables for SN & SNB label paths if datastes
        if (self._dataset == 'soccernet') | (self._dataset == 'soccernetball') | (self._dataset == 'soccernetballanticipation') :
            global LABELS_SN_PATH
            global LABELS_SNB_PATH
            global LABELS_SNBA_PATH
            LABELS_SN_PATH = frame_dir
            LABELS_SNB_PATH = frame_dir
            LABELS_SNBA_PATH = frame_dir

        #Store or load clips
        if self._store_mode == 'store':
            self._store_clips_anticipation() if self._dataset == 'soccernetballanticipation' else self._store_clips()
        elif self._store_mode == 'load':
            self._load_clips()

        self._total_len = len(self._frame_paths)

    def _store_clips_anticipation(self):
        #Initialize frame paths list
        self._frame_paths = []
        self._labels_store = []
        if self._radi_displacement > 0:
            self._labelsD_store = []
        for video in tqdm(self._labels):
            num_clips = int(video['num_clips'])
            full_video_len = int((video['num_frames']//video['num_clips']))
            # For corner anticipation, use full video length to capture labels at the end of clips
            # Original formula only used first portion: int(full_video_len * ((self._clip_len*self._stride/FPS_SN)+5)/30)
            video_len = full_video_len
            labels_files = load_json(os.path.join(LABELS_SNBA_PATH, video['video'] + '/Labels-ball.json'))['videos']

            for clip_idx in range(0,num_clips):
                labels_file = labels_files[clip_idx]['annotations']['observation'] + labels_files[clip_idx]['annotations']['anticipation']
                for base_idx in range(-self._pad_len * self._stride, max(0, video_len - 1 + (2 * self._pad_len - self._clip_len) * self._stride), self._overlap):

                    frames_paths = self._frame_reader.load_paths(video['video'] + f"/clip_{clip_idx+1}", base_idx, base_idx + self._clip_len * self._stride, stride=self._stride)

                    labels = []
                    if self._radi_displacement >= 0:
                        labelsD = []
                    for event in labels_file:
                        event_frame = int(int(event['position']) / 1000 * FPS_SN) #miliseconds to frames
                        label_idx = (event_frame - base_idx) // self._stride

                        if self._radi_displacement >= 0:
                            if (label_idx >= -self._radi_displacement and label_idx < self._clip_len + self._radi_displacement):
                                label = self._class_dict[event['label']]
                                for i in range(max(0, label_idx - self._radi_displacement), min(self._clip_len, label_idx + self._radi_displacement + 1)):
                                    labels.append({'label': label, 'label_idx': i})
                                    labelsD.append({'displ': i - label_idx, 'label_idx': i})

                        else: #EXCLUDE OR MODIFY FOR RADI OF 0
                            if (label_idx >= -self._dilate_len and label_idx < self._clip_len + self._dilate_len):
                                label = self._class_dict[event['label']]
                                for i in range(max(0, label_idx - self._dilate_len), min(self._clip_len, label_idx + self._dilate_len + 1)):
                                    labels.append({'label': label, 'label_idx': i})

                    if frames_paths[1] != -1: #in case no frames were available
                        self._frame_paths.append(frames_paths)
                        self._labels_store.append(labels)
                        if self._radi_displacement > 0:
                            self._labelsD_store.append(labelsD)

        #Save to store
        store_path = os.path.join(self._store_dir, 'LEN' + str(self._clip_len) + 'DIS' + str(self._radi_displacement) + 'SPLIT' + self._split)

        if not os.path.exists(store_path):
            os.makedirs(store_path)

        with open(store_path + '/frame_paths.pkl', 'wb') as f:
            pickle.dump(self._frame_paths, f)
        with open(store_path + '/labels.pkl', 'wb') as f:
            pickle.dump(self._labels_store, f)
        if self._radi_displacement > 0:
            with open(store_path + '/labelsD.pkl', 'wb') as f:
                pickle.dump(self._labelsD_store, f)
        print('Stored clips to ' + store_path)
        return


    def _store_clips(self):
        #Initialize frame paths list
        self._frame_paths = []
        self._labels_store = []
        if self._radi_displacement > 0:
            self._labelsD_store = []
        for video in tqdm(self._labels):
            video_len = int(video['num_frames'])

            #Different label file for SoccerNet (and we require the half for frames):
            if self._dataset == 'soccernet':
                video_half = int(video['video'][-1])
                labels_file = load_json(os.path.join(LABELS_SN_PATH, "/".join(video['video'].split('/')[:-1]) + '/Labels-v2.json'))['annotations']
            elif self._dataset == 'soccernetball':
                video_half = 1
                labels_file = load_json(os.path.join(LABELS_SNB_PATH, video['video'] + '/Labels-ball.json'))['annotations']
            elif self._dataset == 'soccernetballanticipation':
                video_half = 1
                labels_file = load_json(os.path.join(LABELS_SNBA_PATH, video['video'] + '/Labels-ball.json'))['annotations']
            else:
                labels_file = video['events']

            for base_idx in range(-self._pad_len * self._stride, max(0, video_len - 1 + (2 * self._pad_len - self._clip_len) * self._stride), self._overlap):

                if self._dataset == 'finegym':
                    frames_paths = self._frame_reader.load_paths(video['video'], base_idx, base_idx + self._clip_len * self._stride, stride=self._stride, 
                                        source_info = video['_source_info'])
                else:
                    frames_paths = self._frame_reader.load_paths(video['video'], base_idx, base_idx + self._clip_len * self._stride, stride=self._stride)

                labels = []
                if self._radi_displacement >= 0:
                    labelsD = []
                for event in labels_file:
                    
                    #For SoccerNet dataset different label file
                    if (self._dataset == 'soccernet') | (self._dataset == 'soccernetball'):
                        event_half = int(event['gameTime'][0])
                        if event_half == video_half:
                            event_frame = int(int(event['position']) / 1000 * FPS_SN) #miliseconds to frames
                            label_idx = (event_frame - base_idx) // self._stride

                            if self._radi_displacement >= 0:
                                if (label_idx >= -self._radi_displacement and label_idx < self._clip_len + self._radi_displacement):
                                    label = self._class_dict[event['label']]
                                    for i in range(max(0, label_idx - self._radi_displacement), min(self._clip_len, label_idx + self._radi_displacement + 1)):
                                        labels.append({'label': label, 'label_idx': i})
                                        labelsD.append({'displ': i - label_idx, 'label_idx': i})

                            else: #EXCLUDE OR MODIFY FOR RADI OF 0
                                if (label_idx >= -self._dilate_len and label_idx < self._clip_len + self._dilate_len):
                                    label = self._class_dict[event['label']]
                                    for i in range(max(0, label_idx - self._dilate_len), min(self._clip_len, label_idx + self._dilate_len + 1)):
                                        labels.append({'label': label, 'label_idx': i})
                    
                    #For other datasets
                    else:
                        event_frame = event['frame']
                        label_idx = (event_frame - base_idx) // self._stride

                        if self._radi_displacement >= 0:
                            if (label_idx >= -self._radi_displacement and label_idx < self._clip_len + self._radi_displacement):
                                label = self._class_dict[event['label']]
                                for i in range(max(0, label_idx - self._radi_displacement), min(self._clip_len, label_idx + self._radi_displacement + 1)):
                                    labels.append({'label': label, 'label_idx': i})
                                    labelsD.append({'displ': i - label_idx, 'label_idx': i})
                        else: #EXCLUDE OR MODIFY FOR RADI OF 0
                            if (label_idx >= -self._dilate_len and label_idx < self._clip_len + self._dilate_len):
                                label = self._class_dict[event['label']]
                                for i in range(max(0, label_idx - self._dilate_len), min(self._clip_len, label_idx + self._dilate_len + 1)):
                                    labels.append({'label': label, 'label_idx': i})

                if frames_paths[1] != -1: #in case no frames were available

                    #For SoccerNet only include clips with events
                    if self._dataset == 'soccernet':
                        if len(labels) > 0:
                            self._frame_paths.append(frames_paths)
                            self._labels_store.append(labels)
                            if self._radi_displacement > 0:
                                self._labelsD_store.append(labelsD)
                    else:
                        self._frame_paths.append(frames_paths)
                        self._labels_store.append(labels)
                        if self._radi_displacement > 0:
                            self._labelsD_store.append(labelsD)

        #Save to store
        store_path = os.path.join(self._store_dir, 'LEN' + str(self._clip_len) + 'DIS' + str(self._radi_displacement) + 'SPLIT' + self._split)

        if not os.path.exists(store_path):
            os.makedirs(store_path)

        with open(store_path + '/frame_paths.pkl', 'wb') as f:
            pickle.dump(self._frame_paths, f)
        with open(store_path + '/labels.pkl', 'wb') as f:
            pickle.dump(self._labels_store, f)
        if self._radi_displacement > 0:
            with open(store_path + '/labelsD.pkl', 'wb') as f:
                pickle.dump(self._labelsD_store, f)
        print('Stored clips to ' + store_path)
        return
    
    def _load_clips(self):
        store_path = os.path.join(self._store_dir, 'LEN' + str(self._clip_len) + 'DIS' + str(self._radi_displacement) + 'SPLIT' + self._split)
        
        with open(store_path + '/frame_paths.pkl', 'rb') as f:
            self._frame_paths = pickle.load(f)
        with open(store_path + '/labels.pkl', 'rb') as f:
            self._labels_store = pickle.load(f)
        if self._radi_displacement > 0:
            with open(store_path + '/labelsD.pkl', 'rb') as f:
                self._labelsD_store = pickle.load(f)
        print('Loaded clips from ' + store_path)
        return

    def _get_one(self):
        #Get random index
        idx = random.randint(0, self._total_len - 1)
        # Randomly choose an observation percentage out of list in config
        obs_perc_idx = random.randint(0, len(self._obs_perc) - 1)
        obs_perc = self._obs_perc[obs_perc_idx]

        #Get frame_path and labels dict
        frames_path = self._frame_paths[idx]
        dict_label = self._labels_store[idx]
        if self._radi_displacement > 0:
            dict_labelD = self._labelsD_store[idx]

        # Remove excluded classes from labels
        pop_indices = []
        for i, a in enumerate(dict_label):
            if a["label"] in self._excluded_classes:
                pop_indices.append(i)
        for i in range(len(pop_indices)-1, -1, -1):
            dict_label.pop(pop_indices[i])

        #Load frames
        vid_len = frames_path[5]        # Only works when pad=True in load_frames. Otherwise we need to subtract paths[3] "pad_end" from it
        observed_len = int(obs_perc*vid_len)
        pred_len = int(self._pred_perc*vid_len)
        if self._use_anchors:
            assert pred_len % self._n_query == 0, f"Prediction length ({pred_len}) must be divisible by the number of queries ({self._n_query}) if using anchors"
        all_labels = np.array([action["label"] for action in dict_label])
        all_labels_offsets = np.array([action["label_idx"] for action in dict_label])

        # Create observed dataset output
        if self._cheating_dataset:
            # Get observed frames
            cheating_start = int(self._cheating_range[0]*vid_len)
            cheating_end = int(self._cheating_range[1]*vid_len)
            frames = self._frame_reader.load_frames(frames_path, cheating_start, cheating_end, pad=True, stride=self._stride)
            observed_frames = frames[cheating_start:cheating_end]
            # Create array with ground truth label for each frame
            past_labels = np.zeros(len(observed_frames))
            observed_labels_idx = (cheating_start <= all_labels_offsets) & (all_labels_offsets < cheating_end)
            observed_labels = all_labels[observed_labels_idx]
            observed_labels_offsets = all_labels_offsets[observed_labels_idx]
            for a, o in zip(observed_labels, observed_labels_offsets):
                # Smooth label array if needed
                if self._radi_smoothing > 0:
                    for r in range(self._radi_smoothing+1):
                        past_labels[max(0, o-cheating_start-r)] = a
                        past_labels[min(len(past_labels)-1, o-cheating_start+r)] = a
                else:
                    past_labels[o] = a
        else:
            # Get observed frames
            frames = self._frame_reader.load_frames(frames_path, 0, observed_len, pad=True, stride=self._stride)
            observed_frames = frames[:observed_len]
            # Create array with ground truth label for each frame
            past_labels = np.zeros(len(observed_frames))
            observed_labels_idx = all_labels_offsets < observed_len
            observed_labels = all_labels[observed_labels_idx]
            observed_labels_offsets = all_labels_offsets[observed_labels_idx]
            for a, o in zip(observed_labels, observed_labels_offsets):
                # Smooth label array if needed
                if self._radi_smoothing > 0:
                    for r in range(self._radi_smoothing+1):
                        past_labels[max(0, o-r)] = a
                        past_labels[min(len(past_labels)-1, o+r)] = a
                else:
                    past_labels[o] = a
        assert observed_len+pred_len <= vid_len

        # Create anticipation labels
        future_labels_idx = (observed_len <= all_labels_offsets) & (all_labels_offsets < observed_len+pred_len)
        future_labels = all_labels[future_labels_idx]
        future_labels = future_labels if self._use_actionness else np.append(future_labels, self.NONE)          # We do not use EOS if we are using Actionness
        future_offsets = all_labels_offsets[future_labels_idx]
        # Make the offset start from the first anticipated (future) frame
        if len(future_offsets) > 0:
            future_offsets -= observed_len
            assert future_offsets.min() >= 0
            assert future_offsets.max() < pred_len
        
        # Adjust labels to have same length as model prediction, which is #queries
        diff = self._n_query - len(future_labels)
        if diff > 0:
            # If anticipating background instead of EOS, then we pad with NONE, which now represents background instead of EOS for the rest of the sequence
            # When using actionness there will be no EOS, and we want the rest of the sequence to be padding
            tmp = (np.ones(diff) * self.NONE) if self._anticipate_background else (np.ones(diff) * self._label_pad_idx)
            future_labels = np.concatenate((future_labels, tmp))
            tmp_offsets = np.ones(diff) * self._label_pad_idx if self._use_actionness else np.ones(diff+1) * self._label_pad_idx
            future_offsets = np.concatenate((future_offsets, tmp_offsets))
        elif diff < 0:
            future_labels = future_labels[:self._n_query]
            future_offsets = future_offsets[:self._n_query]
        elif not self._use_actionness:
            tmp_offsets = np.ones(1) * self._label_pad_idx
            future_offsets = np.concatenate((future_offsets, tmp_offsets))
        
        # Create actionness
        actionness = np.where(future_labels == self._label_pad_idx, 0, 1) if self._use_actionness else np.zeros_like(future_labels)

        # Create anchors: Go through action, offset pair. Use offset to determine anchor class. If 2 classes are in same anchor, then cry about it (Take last one)
        # Anchor_Offset: Is the number of frames forward within the anchor, so with 32 frames and 8 anchors it's between 0-3.
        anchor_offset = np.ones_like(future_offsets) * self._label_pad_idx
        anchor_labels = np.ones_like(future_labels) * self.NONE if self._anticipate_background else np.ones_like(future_labels) * self._label_pad_idx
        if self._use_anchors:
            for i in range(len(future_labels)):
                if future_offsets[i] == self._label_pad_idx:
                    continue
                else:
                    index = int(future_offsets[i] // (pred_len // self._n_query))
                    anchor_labels[index] = future_labels[i]
                    anchor_offset[index] = future_offsets[i] % 4

        item = {'frames':observed_frames,
                'past_label':torch.tensor(past_labels, dtype=torch.float32),
                'future_offset':torch.tensor(anchor_offset if self._use_anchors else future_offsets, dtype=torch.float32),
                'future_target':torch.tensor(anchor_labels if self._use_anchors else future_labels, dtype=torch.float32),
                'actionness':torch.tensor(actionness, dtype=torch.float32)}
        return item

    def __getitem__(self, unused):
        ret = self._get_one()

        return ret
    
    def my_collate(self, batch):
        '''custom collate function, gets inputs as a batch, output : batch'''
        '''Makes sure the output comes out in a proper format instead of a dictionary'''

        b_features = [item['frames'] for item in batch]
        b_past_label = [item['past_label'] for item in batch]
        b_trans_future_off = [item['future_offset'] for item in batch]
        b_trans_future_target = [item['future_target'] for item in batch]
        b_actionness = [item['actionness'] for item in batch]

        batch_size = len(batch)

        b_features = torch.nn.utils.rnn.pad_sequence(b_features, batch_first=True, padding_value=0) #[B, S, C]
        b_past_label = torch.nn.utils.rnn.pad_sequence(b_past_label, batch_first=True,
                                                         padding_value=self._label_pad_idx)
        b_trans_future_off = torch.nn.utils.rnn.pad_sequence(b_trans_future_off, batch_first=True,
                                                        padding_value=self._label_pad_idx)
        b_trans_future_target = torch.nn.utils.rnn.pad_sequence(b_trans_future_target, batch_first=True, padding_value=self._label_pad_idx)
        b_actionness = torch.nn.utils.rnn.pad_sequence(b_actionness, batch_first=True, padding_value=self._label_pad_idx)

        batch = [b_features, b_past_label, b_trans_future_off, b_trans_future_target, b_actionness]

        return batch

    def __len__(self):
        return self._dataset_len

    def print_info(self):
        _print_info_helper(self._src_file, self._labels)


# Directly from: https://github.com/arturxe2/T-DEED_v2/blob/main/dataset/frame.py
class FrameReader:

    def __init__(self, frame_dir, dataset):
        self._frame_dir = frame_dir
        self.dataset = dataset

    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path)
        return img
    
    def load_paths(self, video_name, start, end, stride=1, source_info = None):

        if self.dataset == 'finediving':
            video_name = video_name.replace('__', '/')
            path = os.path.join(self._frame_dir, video_name)
            frame0 = sorted(os.listdir(path))[0]
            ndigits = len(frame0[:-4])
            frame0 = int(frame0[:-4])

        if self.dataset == 'tennis':
            frame0 = int(video_name.split('_')[-2])
            video_name = '_'.join(video_name.split('_')[:-2])
            path = os.path.join(self._frame_dir, video_name)

        if self.dataset == 'finegym':
            frame0 = source_info['start_frame'] - source_info['pad'][0]
            video_name = video_name.split('_')[0]  
            path = os.path.join(self._frame_dir, video_name)

        if self.dataset == 'soccernetball' or self.dataset == 'soccernetballanticipation':
            path = os.path.join(self._frame_dir, video_name)

        found_start = -1
        pad_start = 0
        pad_end = 0
        for frame_num in range(start, end, stride):

            if frame_num < 0:
                pad_start += 1
                continue

            if pad_end > 0:
                pad_end += 1
                continue
            
            if self.dataset == 'finediving':
                frame = frame0 + frame_num
                frame_path = os.path.join(path, str(frame).zfill(ndigits) + '.jpg')
                base_path = path

            elif (self.dataset == 'fs_comp') | (self.dataset == 'fs_perf'):
                frame = frame_num
                frame_path = os.path.join(self._frame_dir, video_name, 'frame' + str(frame) + '.jpg')
                base_path = os.path.join(self._frame_dir, video_name)
                ndigits = -1

            elif self.dataset == 'soccernet':
                frame = frame_num
                frame_path = os.path.join(self._frame_dir, video_name, 'frame' + str(frame) + '.jpg')
                base_path = os.path.join(self._frame_dir, video_name)   
                ndigits = -1      

            elif self.dataset == 'tennis':
                frame = frame0 + frame_num
                frame_path = os.path.join(path, 'frame' + str(frame) + '.jpg')
                base_path = path
                ndigits = -1

            elif self.dataset == 'finegym':
                frame = frame0 + frame_num
                frame_path = os.path.join(path, 'frame' + str(frame) + '.jpg')
                base_path = path
                ndigits = -1

            elif self.dataset == 'soccernetball' or 'soccernetballanticipation':
                frame = frame_num
                frame_path = os.path.join(path, 'frame' + str(frame) + '.jpg')
                base_path = path
                ndigits = -1
            
            exist_frame = os.path.exists(frame_path)
            if exist_frame & (found_start == -1):
                found_start = frame

            if not exist_frame:
                pad_end += 1

        ret = [base_path, found_start, pad_start, pad_end, ndigits, (end-start) // stride]

        return ret
    
    def load_frames(self, paths, start_index, end_index, pad=False, stride=1):
        base_path = paths[0]
        start = paths[1]
        pad_start = paths[2]
        pad_end = paths[3]
        ndigits = paths[4]
        length = paths[5]

        ret = []
        if ndigits == -1:
            path = os.path.join(base_path, 'frame')
            zero_image = torch.zeros_like(self.read_frame(path + str(start) + '.jpg'))
            for j in range(length - pad_start - pad_end):
                if (j >= (start_index - pad_start)) and (j < (end_index - pad_start)):
                    ret.append(self.read_frame(path + str(start + j * stride) + '.jpg'))
                else:
                    ret.append(zero_image)

        else:
            path = base_path + '/'
            zero_image = torch.zeros_like(self.read_frame(path + str(start).zfill(ndigits) + '.jpg'))
            for j in range(length - pad_start - pad_end):
                if (j >= (start_index - pad_start)) and (j < (end_index - pad_start)):
                    ret.append(self.read_frame(path + str(start + j * stride).zfill(ndigits) + '.jpg'))
                else:
                    ret.append(zero_image)

        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))

        # Always pad start, but only pad end if requested
        if pad_start > 0 or (pad and pad_end > 0):
            ret = torch.nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, pad_start, pad_end if pad else 0))            

        return ret

# BAA dataset for evaluation. Does not support cheating dataset.
# Unlike ActionSpotVideoDataset, this dataset returns labels with the frames
# Instead of providing a list containing all labels for an entire game
# This is due to using clips instead of games to seperate videos
class ActionAnticipationVideoDataset(Dataset):
    def __init__(
            self,
            classes,                        # dict of class names to idx
            label_file,                     # path to label json
            frame_dir,                      # path to frames
            obs_len,                        # Number of frames to observe
            stride=1,                       # Downsample frame rate
            dataset = 'soccernetballanticipation',
    ):
        if not dataset == 'soccernetballanticipation':
            print("For evaluating on datasets other than the Ball Action Anticipation dataset use the ActionSpotVideoDataset class")
            raise NotImplementedError
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._obs_len = obs_len
        self._pred_len = math.ceil(5*FPS_SN/stride) # Hard coded to 5 seconds
        self._stride = stride
        self._dataset = dataset

        self._frame_reader = FrameReaderVideo(frame_dir, dataset = dataset)

        #Variables for SNBA label paths if datastes
        global LABELS_SNBA_PATH
        LABELS_SNBA_PATH = frame_dir

        # Create lists of clips and their annotations sepereated by video (In this case video is individual splits instead of individual games)
        self._clips = []
        self._annotations = {}
        for l in self._labels:
            has_clip = False
            num_clips = l['num_clips']
            clip_len = int(l['num_frames']/num_clips)
            self._annotations[l['video']] = []
            video_annotations = load_json(os.path.join(LABELS_SNBA_PATH, l['video'] + '/Labels-ball.json'))['videos']
            for c in range(num_clips):
                has_clip = True
                self._clips.append((l['video'] + f'/clip_{c+1}', l['video'], c, clip_len))
                self._annotations[l['video']].append(video_annotations[c]['annotations']['anticipation'])
            assert has_clip, l

    def __len__(self):
        return len(self._clips)

    def __getitem__(self, idx):
        video_name, base_dir, clip_num, clip_end = self._clips[idx]

        # Load the observed frames from the end of the clip, such that they are the frames right behind the anticipation window
        frames = self._frame_reader.load_frames(
                video_name, clip_end - self._obs_len*self._stride, clip_end,
                0, self._obs_len, pad=False, stride=self._stride)
        
        labels = self._annotations[base_dir][clip_num]

        labels_array = np.zeros(self._pred_len) # Hard coded to 5 seconds
        visibility_array = np.zeros_like(labels_array)
        for l in labels:
            labels_array[int((int(l['position'])/1000*FPS_SN/self._stride)-(clip_end/self._stride))] = self._class_dict[l['label']]
            visibility_array[int((int(l['position'])/1000*FPS_SN/self._stride)-(clip_end/self._stride))] = -1 if l['visibility'] == "not shown" else 1

        return {'video': base_dir, 'clip': video_name, 'clip_num': torch.tensor(clip_num), 'frame': frames,
                'label': labels_array, 'visibility': visibility_array}
    
    # Returns relevant information about each video (split)
    @property
    def videos(self):
        return sorted([
            (v['video'], math.ceil(self._pred_len / self._stride),
            v["num_clips"], FPS_SN / self._stride) for v in self._labels])

# BAS dataset for evaluation
class ActionSpotVideoDataset(Dataset):

    def __init__(
            self,
            classes,                        # dict of class names to idx
            label_file,                     # path to label json
            frame_dir,                      # path to frames
            clip_len,                       # Number of frames per clip
            start_observe_index,            # Where to start observing from (For cheating dataset)
            end_observe_index,              # Where to stop observing from  (For cheating dataset)
            overlap_len=0,                  # Number of frames of overlap between clips
            stride=1,                       # Downsample frame rate
            pad_len=DEFAULT_PAD_LEN,        # Number of frames to pad the start and end of videos
            dataset = 'finediving',
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._clip_len = clip_len
        self._start_observe_index = start_observe_index
        self._end_observe_index = end_observe_index
        self._stride = stride
        self._dataset = dataset
        pad_len = 0 if self._dataset == 'soccernetballanticipation' else pad_len

        self._frame_reader = FrameReaderVideo(frame_dir, dataset = dataset)

        # Create list containing the video that each clip comes from
        self._clips = []
        for l in self._labels:
            has_clip = False
            for i in range(
                -pad_len * self._stride,
                max(0, l['num_frames'] - (overlap_len * stride)), \
                # Need to ensure that all clips have at least one frame
                (clip_len - overlap_len) * self._stride
            ):
                has_clip = True
                if self._dataset == 'finegym':
                    self._clips.append((l['video'], i, l['_source_info']))
                else:
                    self._clips.append((l['video'], i))
            assert has_clip, l

        #Variables for SN & SNB label paths if datastes
        if (self._dataset == 'soccernet') | (self._dataset == 'soccernetball'):
            global LABELS_SN_PATH
            global LABELS_SNB_PATH
            LABELS_SN_PATH = frame_dir
            LABELS_SNB_PATH = frame_dir

    def __len__(self):
        return len(self._clips)

    def __getitem__(self, idx):
        if self._dataset == 'finegym':
            video_name, start, source_info = self._clips[idx]
        else:
            video_name, start = self._clips[idx]

        if self._dataset == 'finegym':
            frames = self._frame_reader.load_frames(
                video_name, start,
                start + self._clip_len * self._stride,
                self._start_observe_index, self._end_observe_index,
                pad=True, stride=self._stride, source_info = source_info)
        else:
            frames = self._frame_reader.load_frames(
                video_name, start, start + self._clip_len * self._stride,
                self._start_observe_index, self._end_observe_index,
                pad=True, stride=self._stride)

        return {'video': video_name, 'start': start // self._stride,
                'frame': frames}

    # Returns a list of all labels and visibility for an entire game
    def get_labels(self, video):
        meta = self._labels[self._video_idxs[video]]
        if self._dataset == 'soccernet':
            labels_file = load_json(os.path.join(LABELS_SN_PATH, "/".join(meta['video'].split('/')[:-1]) + '/Labels-v2.json'))['annotations']
            labels_half = int(video[-1])
        elif self._dataset == 'soccernetball':
            labels_file = load_json(os.path.join(LABELS_SNB_PATH, meta['video'] + '/Labels-ball.json'))['annotations']
            labels_half = 1
        else:
            labels_file = meta['events']
            labels_half = 0
        
        num_frames = meta['num_frames']
        num_labels = math.ceil(num_frames / self._stride)

        labels = np.zeros(num_labels, np.int64)
        visibility = np.zeros(num_labels, np.int8)
        for event in labels_file:
            if (self._dataset == 'soccernet') | (self._dataset == 'soccernetball'):
                frame = int(int(event['position']) / 1000 * FPS_SN)
                half = int(event['gameTime'][0])
            else:
                frame = event['frame']
                half = 0
            if (half == labels_half):
                if (frame < num_frames):
                    labels[frame // self._stride] = self._class_dict[event['label']]
                    visibility[frame // self._stride] = -1 if event["visibility"] == "not shown" else 1
                else:
                    print('Warning: {} >= {} is past the end {}'.format(
                        frame, num_frames, meta['video']))
        return labels, visibility

    # Returns relevant information about each video (game)
    @property
    def videos(self):
        if (self._dataset == 'soccernet') | (self._dataset == 'soccernetball'):
            return sorted([
                (v['video'], math.ceil(v['num_frames'] / self._stride),
                FPS_SN / self._stride) for v in self._labels])
        return sorted([
            (v['video'], math.ceil(v['num_frames'] / self._stride),
            v['fps'] / self._stride) for v in self._labels])

    # Returns a list containing the labels for all games
    @property
    def labels(self):
        assert self._stride > 0
        if self._stride == 1:
            return self._labels
        else:
            labels = []
            for x in self._labels:
                x_copy = copy.deepcopy(x)
                
                if (self._dataset == 'soccernet') | (self._dataset == 'soccernetball'):
                    x_copy['fps'] = FPS_SN / self._stride
                else:
                    x_copy['fps'] /= self._stride
                x_copy['num_frames'] //= self._stride

                if self._dataset == 'soccernet':
                    labels_file = load_json(os.path.join(LABELS_SN_PATH, "/".join(x_copy['video'].split('/')[:-1]) + '/Labels-v2.json'))['annotations']
                    for e in labels_file:
                        half = int(e['gameTime'][0])
                        if half == int(x_copy['video'][-1]):
                            e['frame'] = int(int(e['position']) / 1000 * FPS_SN) // self._stride
                    x_copy['events'] = labels_file

                elif self._dataset == 'soccernetball':
                    labels_file = load_json(os.path.join(LABELS_SNB_PATH, x_copy['video'] + '/Labels-ball.json'))['annotations']
                    for e in labels_file:
                        e['frame'] = int(int(e['position']) / 1000 * FPS_SN) // self._stride
                    x_copy['events'] = labels_file

                else:
                    for e in x_copy['events']:
                        e['frame'] //= self._stride

                labels.append(x_copy)
            return labels

    def print_info(self):
        num_frames = sum([x['num_frames'] for x in self._labels])
        print('{} : {} videos, {} frames ({} stride)'.format(
            self._src_file, len(self._labels), num_frames, self._stride)
        )
        

class FrameReaderVideo:

    def __init__(self, frame_dir, dataset):
        self._frame_dir = frame_dir
        self._dataset = dataset

    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path) #/ 255 -> modified for ActionSpotVideoDataset (to be compatible with train reading without / 255)
        return img

    def load_frames(self, video_name, start_frame, end_frame, start_index, end_index, pad=False, stride=1, source_info = None):
        ret = []
        n_pad_start = 0
        n_pad_end = 0

        if self._dataset == 'finediving':
            video_name = video_name.replace('__', '/')
            path = os.path.join(self._frame_dir, video_name)
            frame0 = sorted(os.listdir(path))[0]
            ndigits = len(frame0[:-4])
            frame0 = int(frame0[:-4])
        
        if self._dataset == 'tennis':
            frame0 = int(video_name.split('_')[-2])
            video_name = '_'.join(video_name.split('_')[:-2])
            path = os.path.join(self._frame_dir, video_name)

        if self._dataset == 'finegym':
            frame0 = source_info['start_frame'] - source_info['pad'][0]
            video_name = video_name.split('_')[0]  
            path = os.path.join(self._frame_dir, video_name)

        zero_image = None
        num_missing = 0
        for n, frame_num in enumerate(range(start_frame, end_frame, stride)):

            if frame_num < 0:
                n_pad_start += 1
                continue
            
            if self._dataset == 'finediving':
                frame_path = os.path.join(path, str(frame0 + frame_num).zfill(ndigits) + '.jpg')

            elif (self._dataset == 'fs_comp') or (self._dataset == 'fs_perf'):
                frame_path = os.path.join(
                    self._frame_dir, video_name, 'frame' + str(frame_num) + '.jpg'
                )

            elif self._dataset == 'soccernet':
                frame_path = os.path.join(
                    self._frame_dir, video_name, 'frame' + str(frame_num) + '.jpg'
                )

            elif self._dataset == 'soccernetball' or 'soccernetballanticipation':
                frame_path = os.path.join(self._frame_dir, video_name, 'frame' + str(frame_num) + '.jpg')
            
            elif self._dataset == 'tennis':
                frame_path = os.path.join(path, 'frame' + str(frame0 + frame_num) + '.jpg')

            elif self._dataset == 'finegym':
                frame_path = os.path.join(path, 'frame' + str(frame0 + frame_num) + '.jpg')     
            try:
                zero_image = torch.zeros_like(self.read_frame(frame_path)) if zero_image is None else zero_image
                if (n-num_missing >= start_index) and (n-num_missing < end_index):
                    img = self.read_frame(frame_path)
                else:
                    img = zero_image
                ret.append(img)
            except RuntimeError:
                n_pad_end += 1
                num_missing += 1

        if len(ret) == 0:
            return -1 # Return -1 if no frames were loaded

        # In the multicrop case, the shape is (B, T, C, H, W)
        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))

        # Always pad start, but only pad end if requested
        if n_pad_start > 0 or (pad and n_pad_end > 0):
            ret = torch.nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad else 0))
        return ret


def _print_info_helper(src_file, labels):
        num_frames = sum([x['num_frames'] for x in labels])
        print('{} : {} videos, {} frames'.format(
            src_file, len(labels), num_frames))

# Joint dataset for joint training
class ActionSpotDatasetJoint(Dataset):

    def __init__(
            self,
            dataset1,
            dataset2
    ):
        self._dataset1 = dataset1
        self._dataset2 = dataset2
        self._label_pad_idx = dataset1._label_pad_idx
        
    # Randomly returns a clip from one of the two datasets
    def __getitem__(self, unused):

        if random.random() < 0.5:
            data = self._dataset1.__getitem__(unused)
            data['dataset'] = torch.tensor(1)
            return data
        else:
            data = self._dataset2.__getitem__(unused)
            data['dataset'] = torch.tensor(2)
            return data

    def __len__(self):
        return self._dataset1.__len__() + self._dataset2.__len__()
    
    def my_collate(self, batch):
        '''custom collate function, gets inputs as a batch, output : batch'''
        '''Makes sure the output comes out in a proper format instead of a dictionary'''

        b_features = [item['frames'] for item in batch]
        b_past_label = [item['past_label'] for item in batch]
        b_trans_future_off = [item['future_offset'] for item in batch]
        b_trans_future_target = [item['future_target'] for item in batch]
        b_actionness = [item['actionness'] for item in batch]
        b_dataset = [item['dataset'] for item in batch]

        batch_size = len(batch)

        b_features = torch.nn.utils.rnn.pad_sequence(b_features, batch_first=True, padding_value=0) #[B, S, C]
        b_past_label = torch.nn.utils.rnn.pad_sequence(b_past_label, batch_first=True,
                                                         padding_value=self._label_pad_idx)
        b_trans_future_off = torch.nn.utils.rnn.pad_sequence(b_trans_future_off, batch_first=True,
                                                        padding_value=self._label_pad_idx)
        b_trans_future_target = torch.nn.utils.rnn.pad_sequence(b_trans_future_target, batch_first=True, padding_value=self._label_pad_idx)
        b_actionness = torch.nn.utils.rnn.pad_sequence(b_actionness, batch_first=True, padding_value=self._label_pad_idx)

        batch = [b_features, b_past_label, b_trans_future_off, b_trans_future_target, b_actionness, b_dataset]

        return batch
