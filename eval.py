import os
import numpy as np
import torch
import copy
from tqdm import tqdm
from math import ceil
from torch.utils.data import DataLoader
from collections import defaultdict
from tabulate import tabulate
from dataset.datasets import STRIDE_SNB
from dataset.frame import ActionSpotVideoDataset
from SoccerNet.Evaluation.ActionSpotting import average_mAP

#Constants
INFERENCE_BATCH_SIZE = 4

class ErrorStat:

    def __init__(self):
        self._total = 0
        self._err = 0

    def update(self, true, pred):
        self._err += np.sum(true != pred)
        self._total += true.shape[0]

    def get(self):
        return self._err / self._total

    def get_acc(self):
        return 1. - self._get()
    
class ForegroundF1:

    def __init__(self):
        self._tp = defaultdict(int)
        self._fp = defaultdict(int)
        self._fn = defaultdict(int)

    def update(self, true, pred):
        if pred != 0:
            if true != 0:
                self._tp[None] += 1
            else:
                self._fp[None] += 1

            if pred == true:
                self._tp[pred] += 1
            else:
                self._fp[pred] += 1
                if true != 0:
                    self._fn[true] += 1
        elif true != 0:
            self._fn[None] += 1
            self._fn[true] += 1

    def get(self, k):
        return self._f1(k)

    def tp_fp_fn(self, k):
        return self._tp[k], self._fp[k], self._fn[k]

    def _f1(self, k):
        denom = self._tp[k] + 0.5 * self._fp[k] + 0.5 * self._fn[k]
        if denom == 0:
            assert self._tp[k] == 0
            denom = 1
        return self._tp[k] / denom

# Calculates f1 and error and divides support
def process_frame_predictions(pred_dict, target_labels, pad_idx):


    err = ErrorStat()
    f1 = ForegroundF1() # Ignores EOS

    pred_scores = {}
    for video, (scores, support) in (sorted(pred_dict.items())):
        label = target_labels[video]
        if np.min(support) == 0:
            support[support == 0] = 1
        assert np.min(support) > 0, (video, support.tolist())
        scores /= support[:, None]
        indices_to_pad = scores.sum(axis=1) == 0
        pred = np.argmax(scores, axis=1)
        pred[indices_to_pad] = pad_idx
        err.update(label, pred)

        pred_scores[video] = scores#.tolist()
        for i in range(pred.shape[0]):
            f1.update(label[i], pred[i])
    
    return err, f1, pred_scores

def evaluate(split, model, n_class, classes_dict, pad_index, args, overlap=0.5, test=False, use_actionness=False, use_anchors=False):
    model.eval()
    split_path = os.path.join('data', args.dataset, f'{split}.json')
    if os.path.exists(split_path):
        EOS_index = 0
        obs_len = int(args.clip_len*args.test_obs_perc)
        pred_len = int(args.clip_len*args.pred_perc)
        pred_dict = {}
        actual_labels = {}
        actual_visibility = {}
        target_labels = {}
        target_visibility = {}
        clip_indices = {}
        jointtrain_exists = args.jointtrain is not None
        model_head1_sizes = [n_class, n_class - 1*args.actionness]      # Used in case model was trained with dual dataset

        # Get evaluation dataset
        split_data = ActionSpotVideoDataset(
                classes_dict, split_path, args.frame_dir, args.clip_len,
                int(args.clip_len*args.cheating_range[0]) if args.cheating_dataset else 0,
                int(args.clip_len*args.cheating_range[1]) if args.cheating_dataset else obs_len,
                overlap_len = int(args.clip_len*overlap),    # overlap_len is number of frames to overlap
                stride = STRIDE_SNB, dataset = args.dataset)
        if not split_data._dataset == 'soccernetball':
            raise NotImplementedError(f'Evaluation for {split_data._dataset} is not implemented yet.')
        
        # Get video and label information
        for video, video_len, _ in split_data.videos:
            pred_dict[video] = (
                np.zeros((video_len, n_class), np.float32),
                np.zeros(video_len, np.int32))
            actual_labels[video],  actual_visibility[video]= split_data.get_labels(video)   # Has a number representing class for each frame. 0 is background
            # Create extra dictionaries for post-processed GTs
            target_labels[video] = np.ones_like(actual_labels[video]) * pad_index
            target_visibility[video] = np.zeros_like(actual_visibility[video])
            clip_indices[video] = []


        with torch.no_grad():
            for clip in tqdm(DataLoader(
                    split_data, num_workers=args.num_workers, pin_memory=args.num_workers > 0,
                    batch_size=INFERENCE_BATCH_SIZE, shuffle=False
            )):
                # Get predictions from model
                outputs = model(clip['frame'][:,int(args.clip_len*args.cheating_range[0]):int(args.clip_len*args.cheating_range[1])], mode="test") if args.cheating_dataset else model(clip['frame'][:,:obs_len], mode="test")
                # Remove join training classes if joint training is used
                if jointtrain_exists:
                    batch_pred_scores = outputs['action'][...,:model_head1_sizes[1]].softmax(dim=2).detach().cpu().numpy()
                else:
                    batch_pred_scores = outputs['action'].softmax(dim=2).detach().cpu().numpy()
                # Add a background/EOS class for compatibility with the evaluation script
                # Since with actionness the model does not predict the background class
                # Mainly because the background index is removed in aux_evaluate, and we do not want to remove the first class instead.
                if use_actionness:
                    # Get actionness scores
                    if jointtrain_exists:
                        batch_actionness = outputs['actionness'][...,:args.n_query].sigmoid().detach().cpu().numpy()
                    else:
                        batch_actionness = outputs['actionness'].sigmoid().detach().cpu().numpy()
                    # Add a background/EOS class
                    temp = np.zeros((batch_pred_scores.shape[0], batch_pred_scores.shape[1], 1), batch_pred_scores.dtype)
                    batch_pred_scores = np.concatenate((temp, batch_pred_scores), axis=2)
                    # If BCE was used, then actionness is not needed
                    if args.loss_func == "BCE":
                        batch_pred_scores = batch_pred_scores
                    else:
                        # Multiply actionness score with class probabilities
                        for i in range(batch_pred_scores.shape[2]):
                            batch_pred_scores[...,i] *= batch_actionness
                # Get offsets
                if jointtrain_exists:
                    batch_pred_offsets = outputs['offset'][...,:args.n_query].detach().cpu().numpy()
                else:
                    batch_pred_offsets = outputs['offset'].detach().cpu().numpy()
                # Transforming from a sequence to frame segmentation
                batch_seg_scores = np.zeros((batch_pred_scores.shape[0], pred_len, batch_pred_scores.shape[2]), batch_pred_scores.dtype)
                if use_anchors:
                    max_offset = ceil(pred_len / args.n_query)
                    # Assigns the scores to the correct frame (According to the offset and index)
                    for i in range(batch_pred_scores.shape[0]):         # Loop through batch
                        for j in range(batch_pred_scores.shape[1]):     # Loop through anticipated actions in clip
                            # Skip if offset is out of bound from the anchor
                            if batch_pred_offsets[i, j] < 0 or batch_pred_offsets[i, j] >= max_offset:
                                continue
                            # Skip if the anchor is a background prediction.
                            elif np.argmax(batch_pred_scores[i, j]) == EOS_index:
                                if args.anticipate_background:
                                    continue
                            elif batch_seg_scores[i, int(j*max_offset + batch_pred_offsets[i, j])].sum() > 0: # There is already a prediction there
                                batch_seg_scores[i, int(j*max_offset + batch_pred_offsets[i, j])] += batch_pred_scores[i, j]       # Will add prediction over each other if they share offset
                                batch_seg_scores[i, int(j*max_offset + batch_pred_offsets[i, j])] /= 2                             # This is wrong, but it's such a low chance that it makes problems that I cannot be bothered to fix it
                            else:
                                batch_seg_scores[i, int(j*max_offset + batch_pred_offsets[i, j])] = batch_pred_scores[i, j]
                else:
                    # Assigns the scores to the correct frame (According to the offset)
                    # Once it reaches the EOS token it stops (Does not assign the EOS score)
                    for i in range(batch_pred_scores.shape[0]):         # Loop through batch
                        for j in range(batch_pred_scores.shape[1]):     # Loop through anticipated actions in clip
                            # Skip if offset is out of bound
                            if batch_pred_offsets[i, j] < 0 or batch_pred_offsets[i, j] >= pred_len:
                                continue
                            # Ignore all predictions after an EOS prediction. Only if using EOS instead of backgrounds in anticipation
                            elif np.argmax(batch_pred_scores[i, j]) == EOS_index:# and not args.anticipate_background:
                                if args.anticipate_background:
                                    continue
                                else:
                                    break
                            elif batch_seg_scores[i, int(batch_pred_offsets[i, j])].sum() > 0: # There is already a prediction there
                                batch_seg_scores[i, int(batch_pred_offsets[i, j])] += batch_pred_scores[i, j]       # Will add prediction over each other if they share offset
                                batch_seg_scores[i, int(batch_pred_offsets[i, j])] /= 2                             # This is wrong, but it's such a low chance that it makes problems that I cannot be bothered to fix it
                            else:
                                batch_seg_scores[i, int(batch_pred_offsets[i, j])] = batch_pred_scores[i, j]

                # Iterate through the batch
                for i in range(clip['frame'].shape[0]):
                    video = clip['video'][i]
                    scores, support = pred_dict[video]
                    pred_scores = batch_seg_scores[i]
                    # Adjust for padding in dataset
                    start = clip['start'][i].item() + obs_len
                    if start < 0:
                        pred_scores = pred_scores[-start:, :]
                        start = 0
                    end = start + pred_scores.shape[0]
                    if end >= scores.shape[0]:
                        end = scores.shape[0]
                        pred_scores = pred_scores[:end - start, :]

                    # Assign scores to their corresponding frames in the temporary full video predictions list
                    scores[start:end, :] += pred_scores
                    support[start:end] += (pred_scores.sum(axis=1) != 0) * 1
                    # Keep track of the frames that the clip starts and ends at. This is used to properly split clips later when evaluating mAP
                    clip_indices[video].append([start, end])
                    label_index = start
                    recorded_labels = 0
                    # Remove some unneeded target labels
                    # Only record labels that the model can anticipate
                    # Remove labels that exceed n_query
                    while label_index < end:
                        # Forced ignoring background class
                        if (actual_labels[video][label_index] != pad_index) and (actual_labels[video][label_index] != 0) and (actual_labels[video][label_index] not in args.excluded_classes):
                            target_labels[video][label_index] = actual_labels[video][label_index]
                            target_visibility[video][label_index] = actual_visibility[video][label_index]
                            recorded_labels += 1
                        label_index += 1
                    # Note: First obs_len frames that are skipped will be assigned the pad_index

        # Get error rate and f1 score
        err, f1, pred_scores = process_frame_predictions(pred_dict, target_labels, pad_index)
        # Evaluate mAP
        if not test:
            if split_data._dataset == 'soccernetball':
                return evaluate_SNB(target_labels, target_visibility, pred_scores, clip_indices, ((pred_len*2)//(25/STRIDE_SNB))+1, split = split)
            else:
                raise NotImplementedError(f'Evaluation for {split_data._dataset} is not implemented yet.')
        else:
            # TODO: Extract submittable frames when working on challenge set
            if split != 'challenge':
                print('=== Results on {} (w/o NMS) ==='.format(split))
                print('Error (frame-level): {:0.2f}\n'.format(err.get() * 100))

                def get_f1_tab_row(str_k):
                    k = classes_dict[str_k] if str_k != 'any' else None
                    return [str_k, f1.get(k) * 100, *f1.tp_fp_fn(k)]
                rows = [get_f1_tab_row('any')]
                for c in sorted(classes_dict):
                    rows.append(get_f1_tab_row(c))

                print(tabulate(rows, headers=['Exact frame', 'F1', 'TP', 'FP', 'FN'],
                                floatfmt='0.2f'))
                print()
                if split_data._dataset == 'soccernetball':
                    return evaluate_SNB(target_labels, target_visibility, pred_scores, clip_indices, ((pred_len*2)//(25/STRIDE_SNB))+1, split = split)
                else:
                    raise NotImplementedError(f'Evaluation for {split_data._dataset} is not implemented yet.')

def evaluate_SNB(target_labels, target_visibility, pred_scores, clip_indices, max_mAP, split = 'test'):
    games = {
        'train': ["england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich",
            "england_efl/2019-2020/2019-10-01 - Hull City - Sheffield Wednesday",
            "england_efl/2019-2020/2019-10-01 - Brentford - Bristol City",
            "england_efl/2019-2020/2019-10-01 - Blackburn Rovers - Nottingham Forest"],
        'val' : ["england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End"],
        'test': ["england_efl/2019-2020/2019-10-01 - Stoke City - Huddersfield Town",
            "england_efl/2019-2020/2019-10-01 - Reading - Fulham"],
        'challenge': ["england_efl/2019-2020/2019-10-02 - Cardiff City - Queens Park Rangers",
            "england_efl/2019-2020/2019-10-01 - Wigan Athletic - Birmingham City"]
        }

    return multi_aux_evaluate(target_labels, target_visibility, pred_scores, clip_indices, max_mAP, list_games = games[split], version = 2, framerate=25/STRIDE_SNB)

# Evaluates multiple mAP metrics and produces a stable mAP metric without free kick and goal classes (Assumes free kick and goal classes are last in the label list)
def multi_aux_evaluate(target_labels, target_visibility, pred_scores, clip_indices, max_mAP, list_games, version=2, framerate=25):
    res = {}
    # Evaluate mAP values that are in the list
    for metric in ["at1", "at2", "at3", "at4", "at5", ["atInfty", [max_mAP]], "tight", ["tightV2", [1, 5, max_mAP]]]:
        if isinstance(metric, str):
            res[metric], targets = aux_evaluate(target_labels, target_visibility, pred_scores, clip_indices, list_games, version, framerate, metric)
        else:
            res[metric[0]], targets = aux_evaluate(target_labels, target_visibility, pred_scores, clip_indices, list_games, version, framerate, metric[1])
    # Add a stable evaluation metric that ignores free kick and goal classes. Hardcodes class indices
    for k, v in res.items():
        sum = 0
        index_to_eval = [0,1,2,3,4,5,6,7,8,9]
        for i in index_to_eval:
            sum += v["a_mAP_per_class"][i]
        v["a_mAP_stable"] = sum/len(index_to_eval)
    return res, pred_scores, targets

# Split clips and convert them into a format that the SoccerNet mAP evaluation script can use
def aux_evaluate(target_labels, target_visibility, pred_scores, clip_indices, list_games, version=2, framerate=25, metric="loose"):

    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()
        
    targets_save = {}

    for game in tqdm(list_games):

        predictions_game = pred_scores[game]-1e-6       # Remove 1e-06 because of line 343 in mAP calculation script. As script accepts thresholds of 0
        targets_game = np.zeros_like(predictions_game)
        for i in range(len(targets_game)):
            if target_labels[game][i] < targets_game.shape[1]:
                targets_game[i, target_labels[game][i]] = target_visibility[game][i]    # This will make sure pad values are all 0 because visibility for them is 0

        # Used to save targets for debugging if needed
        targets_save[game] = targets_game

        closest_numpy = np.zeros(targets_game.shape) - 1
        # Get the closest action index
        for c in np.arange(targets_game.shape[-1]):
            indexes = np.where(targets_game[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest_numpy.shape[0])
            for i in np.arange(len(indexes) - 2) + 1:
                start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                closest_numpy[start:stop, c] = targets_game[indexes[i], c]

        # Split the clips into seperate arrays such that they are handled seperately without overlap
        for start, stop in clip_indices[game]:
            # A matrix of frames, where for each frame there is a confidence vector.
            # It's 0 in all values except the label where it is 1 if visible and -1 if not.
            # Frames with no labels are all 0
            targets_numpy.append(targets_game[start:stop,1:])        # Remove background class, because otherwise it gets a 0 mAP and weighs the average mAP down
            # A matrix of frames, where for each frame there is a confidence vector.
            # It's 0 in all values except the label where it is confidence of the prediction.
            # Frames with no predictions are all 0
            detections_numpy.append(predictions_game[start:stop,1:]) # Remove background class, because otherwise it gets a 0 mAP and weighs the average mAP down
            # A matrix of frames, where for each frame there is a confidence vector.
            # The vector of each frame mirrors the vector of the closest frame with a label.
            # No frames have all 0 vectors.
            closests_numpy.append(closest_numpy[start:stop,1:])      # Remove background class, because otherwise it gets a 0 mAP and weighs the average mAP down


    if metric == "loose":
        deltas = np.arange(12) * 5 + 5
    elif metric == "tight":
        deltas = np.arange(5) * 1 + 1
    elif metric == "at1":
        deltas = np.array([1])
    elif metric == "at2":
        deltas = np.array([2])
    elif metric == "at3":
        deltas = np.array([3])
    elif metric == "at4":
        deltas = np.array([4])
    elif metric == "at5":
        deltas = np.array([5])
    elif isinstance(metric, list):
        deltas = np.array(metric)
    # Compute the performances
    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = (
        average_mAP(targets_numpy, detections_numpy, closests_numpy, framerate, deltas=deltas)
    )

    results = {
        "a_mAP": a_mAP,
        "a_mAP_per_class": a_mAP_per_class,
        "a_mAP_visible": a_mAP_visible if version == 2 else None,
        "a_mAP_per_class_visible": a_mAP_per_class_visible if version == 2 else None,
        "a_mAP_unshown": a_mAP_unshown if version == 2 else None,
        "a_mAP_per_class_unshown": a_mAP_per_class_unshown if version == 2 else None,
    }
    return results, targets_save