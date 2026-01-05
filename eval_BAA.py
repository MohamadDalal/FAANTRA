import os
import numpy as np
import torch
import copy
from tqdm import tqdm
from math import ceil
from torch.utils.data import DataLoader
from collections import defaultdict
from tabulate import tabulate
from dataset.datasets import STRIDE_SNBA
from dataset.frame import ActionAnticipationVideoDataset, FPS_SN
from util.io import store_json_snba
from SoccerNet.Evaluation.ActionSpotting import average_mAP

#Constants
# Use batch_size=1 to avoid collation issues with inconsistent data types
INFERENCE_BATCH_SIZE = 1

class ErrorStat:

    def __init__(self):
        self._total = 0
        self._err = 0

    def update(self, true, pred):
        self._err += np.sum(true != pred)
        self._total += true.shape[0]*true.shape[1]

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
        scores /= support[..., None]
        indices_to_pad = scores.sum(axis=-1) == 0
        pred = np.argmax(scores, axis=-1)
        pred[indices_to_pad] = pad_idx
        err.update(label, pred)

        pred_scores[video] = scores#.tolist()
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                f1.update(label[i,j], pred[i,j])
    
    return err, f1, pred_scores

def evaluate_BAA(split, model, n_class, classes_dict, pad_index, args, test=False, use_actionness=False, use_anchors=False, save_pred = None):
    model.eval()
    split_path = os.path.join('data', args.dataset, f'{split}.json')
    if os.path.exists(split_path):
        EOS_index = 0
        obs_len = int(args.clip_len*args.test_obs_perc)
        pred_len = ceil(5*FPS_SN/STRIDE_SNBA)   # Hard coded to 5 seconds
        pred_dict = {}
        actual_labels = {}
        actual_visibility = {}
        target_labels = {}
        target_visibility = {}
        jointtrain_exists = args.jointtrain is not None
        model_head1_sizes = [n_class, n_class - 1*args.actionness]      # Used in case model was trained with dual dataset

        # Get evaluation dataset
        split_data = ActionAnticipationVideoDataset(classes_dict, split_path, args.frame_dir, obs_len, stride = STRIDE_SNBA, dataset = args.dataset)
        if split_data._dataset == 'soccernetball':
            raise NotImplementedError(f'To evaluate on the soccernetball dataset use the eval.py file.')
        elif 'anticipation' not in split_data._dataset:
            raise NotImplementedError(f'Evaluation for {split_data._dataset} is not implemented yet.')
        
        # Get video and label information
        for video, _, num_clip, _ in split_data.videos:
            pred_dict[video] = (
                np.zeros((num_clip, pred_len, n_class), np.float32),
                np.zeros((num_clip, pred_len), np.int32))
            actual_labels[video] = np.zeros((num_clip, pred_len), np.int32)   # Has a number representing class for each frame. 0 is background
            actual_visibility[video] = np.zeros((num_clip, pred_len), np.int32)
            # Create extra dictionaries for post-processed GTs
            target_labels[video] = np.ones_like(actual_labels[video]) * pad_index
            target_visibility[video] = np.zeros_like(actual_visibility[video])


        skipped_clips = 0
        with torch.no_grad():
            for clip in tqdm(DataLoader(
                    split_data, num_workers=args.num_workers, pin_memory=args.num_workers > 0,
                    batch_size=INFERENCE_BATCH_SIZE, shuffle=False
            )):
                # Skip clips with malformed data (missing frames)
                try:
                    frame_tensor = clip['frame']
                    if frame_tensor.dim() < 4:  # Expected: [batch, frames, C, H, W]
                        skipped_clips += 1
                        continue
                except Exception as e:
                    skipped_clips += 1
                    continue

                # Get predictions from model
                try:
                    outputs = model(clip['frame'][:,:obs_len], mode="test")
                except (IndexError, RuntimeError) as e:
                    skipped_clips += 1
                    continue
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
                    clip_num = clip['clip_num'][i]
                    # Populate the value of ground truth labels and visibility from clip data
                    actual_labels[video][clip_num, ...] = clip['label'][i]
                    actual_visibility[video][clip_num, ...] = clip['visibility'][i]
                    scores, support = pred_dict[video]
                    pred_scores = batch_seg_scores[i]

                    # Assign scores to their corresponding frames in the temporary full video predictions list
                    scores[clip_num, ...] += pred_scores
                    support[clip_num, :] += (pred_scores.sum(axis=1) != 0) * 1
                    # Remove some unneeded target labels
                    # Only record labels that the model can anticipate
                    # Remove labels that exceed n_query
                    for label_index in range(pred_scores.shape[0]):
                        # Forced ignoring background class
                        if (actual_labels[video][clip_num][label_index] != pad_index) and (actual_labels[video][clip_num][label_index] != 0) and (actual_labels[video][clip_num][label_index] not in args.excluded_classes):
                            target_labels[video][clip_num][label_index] = actual_labels[video][clip_num][label_index]
                            target_visibility[video][clip_num][label_index] = actual_visibility[video][clip_num][label_index]

        if skipped_clips > 0:
            print(f"\nWarning: Skipped {skipped_clips} clips due to missing/malformed frames")

        # Get error rate and f1 score
        err, f1, pred_scores = process_frame_predictions(pred_dict, target_labels, pad_index)
        # Evaluate mAP
        if not test:
            if 'anticipation' in split_data._dataset:
                return evaluate_SNBA(target_labels, target_visibility, pred_scores, ((pred_len*2)//(FPS_SN/STRIDE_SNBA))+1)
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
                if save_pred is not None:
                    print(f"Storing test predictions somewhere under {os.path.join('/'.join(save_pred.split('/')[:-1]) + '/preds')}")
                    store_json_snba(save_pred, pred_scores, pad_index, classes_dict, STRIDE_SNBA)
                if 'anticipation' in split_data._dataset:
                    return evaluate_SNBA(target_labels, target_visibility, pred_scores, ((pred_len*2)//(FPS_SN/STRIDE_SNBA))+1)
                else:
                    raise NotImplementedError(f'Evaluation for {split_data._dataset} is not implemented yet.')
            else:
                if save_pred is not None:
                    print(f"Storing challenge predictions somewhere under {os.path.join('/'.join(save_pred.split('/')[:-1]) + '/preds')}")
                    store_json_snba(save_pred, pred_scores, pad_index, classes_dict, STRIDE_SNBA)
                else:
                    print("No path for storing has been given. Will not store challenge predictions")




def evaluate_SNBA(target_labels, target_visibility, pred_scores, max_mAP):
    return multi_aux_evaluate(target_labels, target_visibility, pred_scores, max_mAP, version = 2, framerate=FPS_SN/STRIDE_SNBA)

# Evaluates multiple mAP metrics
def multi_aux_evaluate(target_labels, target_visibility, pred_scores, max_mAP, version=2, framerate=FPS_SN):
    res = {}
    # Evaluate mAP values that are in the list
    for metric in ["at1", "at2", "at3", "at4", "at5", ["atInfty", [max_mAP]], "tight", ["tightV2", [1, 5, max_mAP]]]:
        if isinstance(metric, str):
            res[metric], targets = aux_evaluate(target_labels, target_visibility, pred_scores, version, framerate, metric)
        else:
            res[metric[0]], targets = aux_evaluate(target_labels, target_visibility, pred_scores, version, framerate, metric[1])
    # For backward compatibility with BAS evaluator
    for k, v in res.items():
        v["a_mAP_stable"] = v["a_mAP"]
    return res, pred_scores, targets

# Split clips and convert them into a format that the SoccerNet mAP evaluation script can use
def aux_evaluate(target_labels, target_visibility, pred_scores, version=2, framerate=FPS_SN, metric="loose"):

    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()
        
    targets_save = {}

    for video in tqdm(list(pred_scores.keys())):

        predictions_game = pred_scores[video]-1e-6       # Remove 1e-06 because of line 343 in mAP calculation script. As script accepts thresholds of 0
        targets_game = np.zeros_like(predictions_game)
        for c in range(targets_game.shape[0]):
            for i in range(targets_game.shape[1]):
                if target_labels[video][c][i] < targets_game.shape[1]:
                    targets_game[c, i, target_labels[video][c][i]] = target_visibility[video][c][i]    # This will make sure pad values are all 0 because visibility for them is 0

        # Used to save targets for debugging if needed
        targets_save[video] = targets_game

        closest_numpy = np.zeros(targets_game.shape) - 1
        # Get the closest action index
        for clip in np.arange(targets_game.shape[0]):
            for c in np.arange(targets_game.shape[-1]):
                indexes = np.where(targets_game[clip][:, c] != 0)[0].tolist()
                if len(indexes) == 0:
                    continue
                indexes.insert(0, -indexes[0])
                indexes.append(2 * closest_numpy.shape[0])
                for i in np.arange(len(indexes) - 2) + 1:
                    start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                    stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                    closest_numpy[clip][start:stop, c] = targets_game[clip][indexes[i], c]

        # Convert numpy array to a list of arrays so that it works with the average_mAP function
        for clip in range(targets_game.shape[0]):
            # A matrix of frames, where for each frame there is a confidence vector.
            # It's 0 in all values except the label where it is 1 if visible and -1 if not.
            # Frames with no labels are all 0
            targets_numpy.append(targets_game[clip,:,1:])        # Remove background class, because otherwise it gets a 0 mAP and weighs the average mAP down
            # A matrix of frames, where for each frame there is a confidence vector.
            # It's 0 in all values except the label where it is confidence of the prediction.
            # Frames with no predictions are all 0
            detections_numpy.append(predictions_game[clip,:,1:]) # Remove background class, because otherwise it gets a 0 mAP and weighs the average mAP down
            # A matrix of frames, where for each frame there is a confidence vector.
            # The vector of each frame mirrors the vector of the closest frame with a label.
            # No frames have all 0 vectors.
            closests_numpy.append(closest_numpy[clip,:,1:])      # Remove background class, because otherwise it gets a 0 mAP and weighs the average mAP down


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