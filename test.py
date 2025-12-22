import torch
import json
import os
import argparse
import numpy as np
from opts import update_args
from util.dataset import load_classes
from torch import nn
from eval import evaluate
from eval_BAA import evaluate_BAA
from model.futr import FUTR

def main():
    args = argparse.ArgumentParser()
    args.add_argument("config", type=str, help="Path to config file")
    args.add_argument('checkpoint', type=str, help='Path to checkpoint')
    args.add_argument('model', type=str, help='Model name')
    args.add_argument('-s', '--split', type=str, default="test", choices=["train", "val", "test", "challenge"],
                        help='Split to test on.')
    args.add_argument('-o', '--overlap', type=float, default=0.5,
                        help='Overlap between clips (0.5 is no overlap between observations). Only applicable to BAS dataset')
    # From FUTR
    args.add_argument("--cpu", action='store_true', help='run in cpu')
    args = args.parse_args()
    # Load config
    config_path = args.config
    with open(config_path, "r") as f:
        config = json.load(f)
    args = update_args(args, config)
    pad_idx = 255
    actions_dict = load_classes(os.path.join('data', args.dataset, 'class.txt'))
    n_class = len(actions_dict) - len(args.excluded_classes)
    if args.cpu:
        device = torch.device('cpu')
        print('using cpu')
    else:
        device = torch.device('cuda')
        print('using gpu')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print("Config path:", config_path)
    print("Checkpoint path:", args.checkpoint)
    print("Overlap:", args.overlap)
    # Model specification
    src_attn_mask = None
    tgt_attn_mask = None
    if args.mask_attn:
        max_obs_len = int(args.clip_len*max(args.obs_perc))
        src_attn_mask = torch.full((max_obs_len, max_obs_len), True).to(device)
        tgt_attn_mask = torch.full((args.n_query, args.n_query), True).to(device)
        for i in range(max_obs_len):
            start = max(0, i - (args.mask_attn_window_src//2))
            end = min(max_obs_len, i + (args.mask_attn_window_src//2) + 1)
            src_attn_mask[i, start:end] = False
        for i in range(args.n_query):
            start = max(0, i - (args.mask_attn_window_tgt//2))
            end = min(args.n_query, i + (args.mask_attn_window_tgt//2) + 1)
            tgt_attn_mask[i, start:end] = False
    model = FUTR(n_class, args.hidden_dim, device=device, args=args, src_pad_idx=pad_idx,
                            n_query=args.n_query, n_head=args.n_head,
                            num_encoder_layers=args.n_encoder_layer, num_decoder_layers=args.n_decoder_layer,
                            src_attn_mask=src_attn_mask, tgt_attn_mask=tgt_attn_mask).to(device)
    model = nn.DataParallel(model).to(device)
    # Check if training checkpoint or model checkpoint
    if "model_state_dict" in checkpoint.keys():
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    # Evaluate and print results
    if args.split == "challenge":
        if args.dataset == 'soccernetballanticipation':
            evaluate_BAA(args.split, model, n_class, actions_dict, pad_idx, args, True, args.actionness, args.use_anchors, args.checkpoint)
        else:
            evaluate(args.split, model, n_class, actions_dict, pad_idx, args, args.overlap, True, args.actionness, args.use_anchors)
    else:
        if args.dataset == 'soccernetballanticipation':
            results, predictions, targets = evaluate_BAA(args.split, model, n_class, actions_dict, pad_idx, args, True, args.actionness, args.use_anchors, args.checkpoint)
        else:
            results, predictions, targets = evaluate(args.split, model, n_class, actions_dict, pad_idx, args, args.overlap, True, args.actionness, args.use_anchors)
        print(results)
        # Save target and predictions for debugging
        os.makedirs(args.checkpoint[:-5]+"-results/", exist_ok=True)
        for game in predictions.keys():
            np.save(args.checkpoint[:-5]+f"-results/predictions-{game.split('/')[-1]}.npy", predictions[game])
        for game in targets.keys():
            np.save(args.checkpoint[:-5]+f"-results/targets-{game.split('/')[-1]}.npy", targets[game])


if __name__ == "__main__":
    main()