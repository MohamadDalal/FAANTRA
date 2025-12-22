import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import wandb
import sys
from opts import get_args, update_args
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
from model.futr import FUTR
from train import train
from train_dual import train_dual
from eval import evaluate
from eval_BAA import evaluate_BAA
from util.io import load_json
from util.dataset import load_classes
from utils import seed_everything
from dataset.datasets import get_datasets

device = torch.device('cuda')

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])


def main():
    # Load arguments and config and apply seed
    args = get_args()
    config_path = args.config
    config = load_json(config_path)
    args = update_args(args, config)

    if args.cpu:
        device = torch.device('cpu')
        print('using cpu')
    else:
        device = torch.device('cuda')
        print('using gpu')
    seed_everything(args.seed)


    # Load checkpoint
    checkpoint = {}
    if not args.checkpoint_path is None:
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)

    # initialize wandb
    wandb.login()
    if checkpoint.get("wandb_run_id", None) is None or args.wandb_mode != "online" or args.wandb_new_id:
        wandb.init(config = args,
                dir = "wandb_logs/",
                project = 'SoccerNet_Anticipate_FUTR',
                name = args.model + '-' + str(args.seed),
                mode = args.wandb_mode)
    else:
        wandb.init(config = args,
                dir = "wandb_logs/",
                project = 'SoccerNet_Anticipate_FUTR',
                name = args.model + '-' + str(args.seed),
                id = checkpoint["wandb_run_id"],
                resume = 'must',
                mode = args.wandb_mode)

    print('save directory : ', args.save_dir)
    print('model type : ', args.model)
    print('Epoch : ', args.epochs)
    print("batch size : ", args.batch_size)

    # Load classes listed in class.txt and remove any excluded classes
    # Can only remove classes at the end of the list
    actions_dict = load_classes(os.path.join('data', args.dataset, 'class.txt'))

    n_class = len(actions_dict) - len(args.excluded_classes)
    print("n_class:",n_class)
    pad_idx = 255 # Pad index. Set to 255 to not collide with offset values. Increase if you need more than 254 frames

    # Create attention masks if needed
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

    # Model specification
    model = FUTR(n_class, args.hidden_dim, device=device, args=args, src_pad_idx=pad_idx,
                            n_query=args.n_query, n_head=args.n_head,
                            num_encoder_layers=args.n_encoder_layer, num_decoder_layers=args.n_decoder_layer,
                            src_attn_mask=src_attn_mask, tgt_attn_mask=tgt_attn_mask).to(device)

    model_save_path = os.path.join(args.save_dir, 'transformer')
    results_save_path = os.path.join(args.save_dir, 'results', 'transformer')
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)


    model = nn.DataParallel(model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss(reduction = 'none')  # Criterion is used for offset loss

    # Training
    # Get datasets train, validation
    _, _, trainset, valset, _ = get_datasets(args, pad_idx, n_class)
    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Stop training here and rerun.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')
    loader_batch_size = args.batch_size

    # Dataloaders
    # Dataset is shuffled by default since it gives a random clip
    prefetch = 2 if args.num_workers > 0 else None
    train_loader = DataLoader(
        trainset, shuffle=False, batch_size=loader_batch_size,
        pin_memory=args.num_workers > 0, num_workers=args.num_workers,
        prefetch_factor=prefetch, collate_fn=trainset.my_collate)

    val_loader = DataLoader(
        valset, shuffle=False, batch_size=loader_batch_size,
        pin_memory=args.num_workers > 0, num_workers=args.num_workers,
        prefetch_factor=prefetch, collate_fn=valset.my_collate)
    
    # Create scheduler
    num_steps_per_epoch = len(train_loader)
    scheduler = get_lr_scheduler(args, optimizer, num_steps_per_epoch)
    
    # Load parameters from checkpoint
    start_epoch = 0
    best_mAP = 0
    best_model_path = ""
    if not args.checkpoint_path is None:
        if not "model_state_dict" in checkpoint.keys():
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_mAP = checkpoint["best_mAP"]
            best_model_path = checkpoint["best_model_path"]

    # Start training
    if start_epoch < args.epochs:
        if args.jointtrain is not None:
            model, best_model_path = train_dual(args, model, train_loader, val_loader, optimizer, scheduler, criterion,
                                        model_save_path, pad_idx, device, int(args.pred_perc*args.clip_len),
                                        n_class, actions_dict, args.n_query, start_epoch=start_epoch,
                                        offset_loss_weight=args.offset_loss_weight, use_actionness=args.actionness,
                                        use_anchors=args.use_anchors, loss_func=args.loss_func ,best_mAP=best_mAP,
                                        best_model_path=best_model_path)
        else:
            model, best_model_path = train(args, model, train_loader, val_loader, optimizer, scheduler, criterion,
                                        model_save_path, pad_idx, device, int(args.pred_perc*args.clip_len),
                                        n_class, actions_dict, args.n_query, start_epoch=start_epoch,
                                        offset_loss_weight=args.offset_loss_weight, use_actionness=args.actionness,
                                        use_anchors=args.use_anchors, loss_func=args.loss_func ,best_mAP=best_mAP,
                                        best_model_path=best_model_path)
    
    # Load best checkpoint and evaluate it on test dataset
    if not best_model_path == "":
        best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint)
    if args.dataset == 'soccernetballanticipation':
        eval_results, _, _ = evaluate_BAA("test", model, n_class, actions_dict, pad_idx, args, True, use_actionness=args.actionness, use_anchors=args.use_anchors)
    else:
        eval_results, _, _ = evaluate("test", model, n_class, actions_dict, pad_idx, args, 0.9 if args.n_query == 1 else 1-args.pred_perc, True, use_actionness=args.actionness, use_anchors=args.use_anchors)
    print("Results from the best model:")
    print(eval_results)
    wandb.log(eval_results)
    wandb.finish(quiet=True)

if __name__ == '__main__':
    main()
