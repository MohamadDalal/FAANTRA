import torch
import os
import wandb
from dataset.datasets import STRIDE_SNB
from dataset.frame import FPS_SN
from tqdm import tqdm
from utils import cal_performance, normalize_offset, cal_actionness_performance, CALF_matching, CALF_matching2
from eval import evaluate
from eval_BAA import evaluate_BAA

# Segmentation loss is forced to can only use CE as loss function
def train(args, model, train_loader, val_loader, optimizer, scheduler, criterion,
          model_save_path, pad_idx, device, num_pred_frames, n_class, class_dict, n_query,
          start_epoch=0, offset_loss_weight=1.0, use_actionness=False, use_anchors=False,
          loss_func="CE", best_mAP = 0, best_model_path=""):
    torch.autograd.set_detect_anomaly(True)     # Detects if NaNs are present in backpropagation. Disable for faster training.
    inv_class_dict = {v: k for k, v in class_dict.items()}
    num_pred_frames = int(num_pred_frames // n_query) if use_anchors else num_pred_frames
    BCE_with_actionness = loss_func == "BCE" and use_actionness
    if BCE_with_actionness: use_actionness = False               # Force actionness off, because BCE is made to replace it
    model.to(device)
    model.train()
    best_mAP = best_mAP
    best_model_path = best_model_path
    print("Training Start")
    for epoch in range(start_epoch, args.epochs):

        ########################################
        # Training
        ########################################

        epoch_loss = 0
        epoch_loss_class = 0
        epoch_loss_off = 0
        epoch_loss_seg = 0
        epoch_class_stats = None
        epoch_class_stats_seg = None
        if use_actionness:
            epoch_loss_actionness = 0
            epoch_actionness_stats = None
        total_class = 0
        total_class_correct = 0
        total_off_correct = 0
        total_seg = 0
        total_seg_correct = 0
        train_loop = tqdm(train_loader)
        for i, data in enumerate(train_loop):
            step_log_dict = {"train/step": epoch*len(train_loader) + i+1}
            postfix_kwargs = {"loss": 0}
            optimizer.zero_grad()
            features, past_label, trans_off_future, trans_future_target, target_actionness = data
            features = features.to(device) #[B, S, C]
            past_label = past_label.to(device) #[B, S]
            trans_off_future = trans_off_future.to(device)
            trans_future_target = trans_future_target.to(device)
            trans_off_future_mask = (trans_off_future != pad_idx).long().to(device) # Mask off padding in ground truth predictions. Not relevant when using background
            target_actionness = target_actionness.to(device)

            target_off = trans_off_future*trans_off_future_mask # Mask off padding in ground truth offsets. Not relevant when using background
            target = trans_future_target
            inputs = (features, past_label)

            outputs = model(inputs)
            losses = 0

            ########################################
            # Past segmentation (Auxilary task)
            ########################################
            if args.seg :
                output_seg = outputs['seg']
                B, T, C = output_seg.size()
                output_seg = output_seg.view(-1, C).to(device)
                target_past_label = past_label.view(-1)
                class_weights = torch.tensor(args.class_weights, device=device)
                # Calculate loss and accuracy
                loss_seg, n_seg_correct, n_seg_total, seg_class_stats = cal_performance(output_seg, target_past_label, pad_idx, class_weights=class_weights)
                losses += loss_seg
                total_seg += n_seg_total
                total_seg_correct += n_seg_correct
                epoch_loss_seg += loss_seg.item()
                step_log_dict["train/seg_loss"] = loss_seg.item()
                postfix_kwargs["loss_seg"] = loss_seg.item()
                # Record loss, accuracy and class-wise statistics for auxilary task
                if n_seg_total == 0:
                    step_log_dict["train/seg_acc"] = 1
                    postfix_kwargs["acc_seg"] = 1
                else:
                    step_log_dict["train/seg_acc"] = n_seg_correct/n_seg_total
                    postfix_kwargs["acc_seg"] = n_seg_correct/n_seg_total
                if epoch_class_stats_seg is None:
                    epoch_class_stats_seg = seg_class_stats
                else:
                    for class_key in seg_class_stats.keys():
                        for class_stat in seg_class_stats[class_key].keys():
                            epoch_class_stats_seg[class_key][class_stat] += seg_class_stats[class_key][class_stat]
            
            ########################################
            # Future anticipation (Main task)
            ########################################
            if args.anticipate :
                ########################################
                # Prediction processing
                ########################################
                output = outputs['action']
                B, T, C = output.size()
                # Perform CALF matching of predictions and ground truth if needed
                if args.CALF_matching:
                    if args.CALF_probability_matching:
                        output, output_off, output_actionness = CALF_matching2(output, target, outputs['offset'], target_off, pad_idx, use_actionness=use_actionness,
                                                                               output_actionness=outputs['actionness'] if use_actionness else None,
                                                                               target_actionness=target_actionness if use_actionness else None)
                    else:
                        output, output_off, output_actionness = CALF_matching(output, target, outputs['offset'], target_off, pad_idx, use_actionness=use_actionness,
                                                                              output_actionness=outputs['actionness'] if use_actionness else None,
                                                                              target_actionness=target_actionness if use_actionness else None)
                output = output.view(-1, C).to(device)
                target = target.contiguous().view(-1)
                # Shift target by 1 if using actionness, so that it matches the predictions that do not have background/eos class
                # Does not shift padding predictions
                target = torch.where(target == pad_idx, target, target - 1) if use_actionness or BCE_with_actionness else target       # If using Actionness, then it moves target values back by one to compensate for lack of EOS
                class_weights = torch.tensor(args.class_weights, device=device)
                class_weights[0] = args.eos_weight  # Replace background weight with EOS weight
                # Calculate prediction loss and accuracy
                if use_actionness or BCE_with_actionness:
                    loss, n_correct, n_total, class_stats = cal_performance(output, target, pad_idx, loss_func=loss_func, class_weights=class_weights[1:], actionness=True)
                else:
                    loss, n_correct, n_total, class_stats = cal_performance(output, target, pad_idx, loss_func=loss_func, class_weights=class_weights, actionness=use_actionness)
                acc = 1 if n_total == 0 else n_correct / n_total
                # Actionness gets NaN CE_loss at the start of an epoch and I do not know why.
                # So this is how I chose to sweep the issue under the rug
                loss = torch.nan_to_num(loss)
                losses += loss
                total_class += n_total
                total_class_correct += n_correct
                epoch_loss_class += loss.item()

                ########################################
                # Offset processing
                ########################################
                output_off = output_off if args.CALF_matching else outputs['offset']
                # Normalize offset to be between 0-1 instead of 0-num_frames
                transformed_output_off = normalize_offset(output_off, trans_off_future_mask, num_pred_frames)
                transformed_target_off = normalize_offset(target_off, trans_off_future_mask, num_pred_frames)
                # Calculate offset loss and 1s accuracy
                if torch.sum(trans_off_future_mask) == 0:
                    loss_off = torch.sum(criterion(transformed_output_off, transformed_target_off))     # Due to trans_off_future_mask, this will be 0, as both vectors are fully 0
                else:
                    loss_off = torch.sum(criterion(transformed_output_off, transformed_target_off)) / \
                    torch.sum(trans_off_future_mask)
                loss_off *= offset_loss_weight
                losses += loss_off
                epoch_loss_off += loss_off.item()
                # Get offset accuracy in 1 second intervals
                unrolled_output_off = output_off.view(-1)
                unrolled_target_off = target_off.view(-1)
                unrolled_off_mask = trans_off_future_mask.view(-1)
                off_correct = 0
                for d in range(len(unrolled_target_off)):
                    if unrolled_off_mask[d]:
                        if (unrolled_output_off[d] - unrolled_target_off[d]).abs() <= FPS_SN/STRIDE_SNB:
                            off_correct += 1
                
                ########################################
                # Actionness processing
                ########################################
                if use_actionness:
                    # Calculate actionness loss
                    output_actionness = output_actionness if args.CALF_matching else outputs['actionness']
                    output_actionness = output_actionness.view(-1).to(device)
                    target_actionness = target_actionness.contiguous().view(-1)
                    actionness_loss, actionness_stats = cal_actionness_performance(output_actionness, target_actionness, threshold=0.5)
                    losses += actionness_loss
                    epoch_loss_actionness += actionness_loss.item()

                # Record and display main task training statistics
                total_off_correct += off_correct
                step_log_dict["train/CE_acc"] = acc
                step_log_dict["train/CE_loss"] = loss.item()
                step_log_dict["train/offset_loss"] = loss_off.item()
                step_log_dict["train/offset_acc@1s"] = 1 if n_total == 0 else off_correct/n_total
                if use_actionness: step_log_dict["train/actionness_acc"] = (actionness_stats["TP"] + actionness_stats["TN"]) / (actionness_stats["TP"] + actionness_stats["FN"] + actionness_stats["FP"] + actionness_stats["TN"])
                postfix_kwargs["loss_CE"] = loss.item()
                postfix_kwargs["loss_offset"] = loss_off.item()
                postfix_kwargs["acc_ant"] = acc
                postfix_kwargs["acc_offset"] = 1 if n_total == 0 else off_correct/n_total
                if use_actionness: postfix_kwargs["acc_actionness"] = (actionness_stats["TP"] + actionness_stats["TN"]) / (actionness_stats["TP"] + actionness_stats["FN"] + actionness_stats["FP"] + actionness_stats["TN"])
                if epoch_class_stats is None:
                    epoch_class_stats = class_stats
                else:
                    for class_key in class_stats.keys():
                        for class_stat in class_stats[class_key].keys():
                            epoch_class_stats[class_key][class_stat] += class_stats[class_key][class_stat]
                if use_actionness:
                    if epoch_actionness_stats is None:
                        epoch_actionness_stats = actionness_stats
                    else:
                        for stat_key in actionness_stats.keys():
                            epoch_actionness_stats[stat_key] += actionness_stats[stat_key]

            # Back propagate loss
            epoch_loss += losses.item()
            losses.backward()
            optimizer.step()
            # Record and display full training statistics for the iteration
            step_log_dict["train/full_loss"] = losses.item()
            postfix_kwargs["loss"] = losses.item()
            train_loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
            train_loop.set_postfix(**postfix_kwargs)
            step_log_dict["train/lr"] = optimizer.param_groups[0]['lr']
            if args.seg:
                inv_class_dict[0] = "BACKGROUND"
                step_log_dict = log_class_metrics(step_log_dict, seg_class_stats, "train/seg", inv_class_dict)
            if args.anticipate:
                inv_class_dict[0] = "EOS"
                step_log_dict = log_class_metrics(step_log_dict, class_stats, "train/anticipate", inv_class_dict)
            wandb.log(step_log_dict)
            scheduler.step()


        ########################################
        # Validation
        ########################################

        val_epoch_loss = 0
        val_epoch_loss_class = 0
        val_epoch_loss_off = 0
        val_epoch_loss_seg = 0
        val_epoch_class_stats = None
        val_epoch_class_stats_seg = None
        if use_actionness:
            val_epoch_loss_actionness = 0
            val_epoch_actionness_stats = None
        val_total_class = 0
        val_total_class_correct = 0
        val_total_off_correct = 0
        val_total_seg = 0
        val_total_seg_correct = 0
        val_loop = tqdm(val_loader)
        with torch.no_grad():
            for j, data in enumerate(val_loop):
                #print(j)
                step_log_dict = {"val/step": epoch*len(val_loader) + j+1}
                postfix_kwargs = {"loss": 0}
                features, past_label, trans_off_future, trans_future_target, target_actionness = data
                features = features.to(device) #[B, S, C]
                past_label = past_label.to(device) #[B, S]
                trans_off_future = trans_off_future.to(device)
                trans_future_target = trans_future_target.to(device)
                trans_off_future_mask = (trans_off_future != pad_idx).long().to(device) # Mask off padding in ground truth predictions. Not relevant when using background
                target_actionness = target_actionness.to(device)

                target_off = trans_off_future*trans_off_future_mask # Mask off padding in ground truth offsets. Not relevant when using background
                target = trans_future_target
                inputs = features

                outputs = model(inputs, mode="validation")
                losses = 0
                
                ########################################
                # Past segmentation (Auxilary task)
                ########################################
                if args.seg :
                    output_seg = outputs['seg']
                    B, T, C = output_seg.size()
                    output_seg = output_seg.view(-1, C).to(device)
                    target_past_label = past_label.view(-1)
                    class_weights = torch.tensor(args.class_weights, device=device)
                    # Calculate loss and accuracy
                    loss_seg, n_seg_correct, n_seg_total, seg_class_stats = cal_performance(output_seg, target_past_label, pad_idx, class_weights=class_weights)
                    losses += loss_seg
                    val_total_seg += n_seg_total
                    val_total_seg_correct += n_seg_correct
                    val_epoch_loss_seg += loss_seg.item()
                    step_log_dict["val/seg_loss"] = loss_seg.item()
                    postfix_kwargs["loss_seg"] = loss_seg.item()
                    # Record loss, accuracy and class-wise statistics for auxilary task
                    if n_seg_total == 0:
                        step_log_dict["val/seg_acc"] = 1
                        postfix_kwargs["acc_seg"] = 1
                    else:
                        step_log_dict["val/seg_acc"] = n_seg_correct/n_seg_total
                        postfix_kwargs["acc_seg"] = n_seg_correct/n_seg_total
                    if val_epoch_class_stats_seg is None:
                        val_epoch_class_stats_seg = seg_class_stats
                    else:
                        for class_key in seg_class_stats.keys():
                            for class_stat in seg_class_stats[class_key].keys():
                                val_epoch_class_stats_seg[class_key][class_stat] += seg_class_stats[class_key][class_stat]
                
                ########################################
                # Future anticipation (Main task)
                ########################################
                if args.anticipate:
                    ########################################
                    # Prediction processing
                    ########################################
                    output = outputs['action']
                    # Perform CALF matching of predictions and ground truth if needed
                    if args.CALF_matching:
                        if args.CALF_probability_matching:
                            output, output_off, output_actionness = CALF_matching2(output, target, outputs['offset'], target_off, pad_idx, use_actionness=use_actionness,
                                                                                   output_actionness=outputs['actionness'] if use_actionness else None,
                                                                                   target_actionness=target_actionness if use_actionness else None)
                        else:
                            output, output_off, output_actionness = CALF_matching(output, target, outputs['offset'], target_off, pad_idx, use_actionness=use_actionness,
                                                                                  output_actionness=outputs['actionness'] if use_actionness else None,
                                                                                  target_actionness=target_actionness if use_actionness else None)
                    B, T, C = output.size()
                    output = output.view(-1, C).to(device)
                    target = target.contiguous().view(-1)
                    # Shift target by 1 if using actionness, so that it matches the predictions that do not have background/eos class
                    # Does not shift padding predictions
                    target = torch.where(target == pad_idx, target, target - 1) if use_actionness or BCE_with_actionness else target
                    class_weights = torch.tensor(args.class_weights, device=device)
                    class_weights[0] = args.eos_weight  # Replace background weight with EOS weight
                    # Calculate prediction loss and accuracy
                    if use_actionness or BCE_with_actionness:
                        loss, n_correct, n_total, class_stats = cal_performance(output, target, pad_idx, loss_func=loss_func, class_weights=class_weights[1:], actionness=True)
                    else:
                        loss, n_correct, n_total, class_stats = cal_performance(output, target, pad_idx, loss_func=loss_func, class_weights=class_weights, actionness=use_actionness)
                    acc = 1 if n_total == 0 else n_correct / n_total
                    # Actionness gets NaN CE_loss at the start of an epoch and I do not know why.
                    # So this is how I chose to sweep the issue under the rug
                    loss = torch.nan_to_num(loss)
                    losses += loss
                    val_total_class += n_total
                    val_total_class_correct += n_correct
                    val_epoch_loss_class += loss.item()

                    ########################################
                    # Offset processing
                    ########################################
                    output_off = output_off if args.CALF_matching else outputs['offset']
                    # Normalize offset to be between 0-1 instead of 0-num_frames
                    transformed_output_off = normalize_offset(output_off, trans_off_future_mask, num_pred_frames)
                    transformed_target_off = normalize_offset(target_off, trans_off_future_mask, num_pred_frames)
                    # Calculate offset loss and 1s accuracy
                    if torch.sum(trans_off_future_mask) == 0:
                        loss_off = torch.sum(criterion(transformed_output_off, transformed_target_off))
                    else:
                        loss_off = torch.sum(criterion(transformed_output_off, transformed_target_off)) / \
                        torch.sum(trans_off_future_mask)
                    loss_off *= offset_loss_weight
                    losses += loss_off
                    val_epoch_loss_off += loss_off.item()
                    # Get offset accuracy in 1 second intervals
                    unrolled_output_off = output_off.view(-1)
                    unrolled_target_off = target_off.view(-1)
                    unrolled_off_mask = trans_off_future_mask.view(-1)
                    off_correct = 0
                    for d in range(len(unrolled_target_off)):
                        if unrolled_off_mask[d]:
                            if (unrolled_output_off[d] - unrolled_target_off[d]).abs() <= 25/STRIDE_SNB:
                                off_correct += 1
                    
                    ########################################
                    # Actionness processing
                    ########################################
                    if use_actionness:
                        # Calculate actionness loss
                        output_actionness = output_actionness if args.CALF_matching else outputs['actionness']
                        output_actionness = output_actionness.view(-1).to(device)
                        target_actionness = target_actionness.contiguous().view(-1)
                        actionness_loss, actionness_stats = cal_actionness_performance(output_actionness, target_actionness, threshold=0.5)
                        losses += actionness_loss
                        val_epoch_loss_actionness += actionness_loss.item()

                    # Record and display main task training statistics
                    val_total_off_correct += off_correct
                    step_log_dict["val/CE_acc"] = acc
                    step_log_dict["val/CE_loss"] = loss.item()
                    step_log_dict["val/offset_loss"] = loss_off.item()
                    step_log_dict["val/offset_acc@1s"] = 1 if n_total == 0 else off_correct/n_total
                    if use_actionness: step_log_dict["val/actionness_acc"] = (actionness_stats["TP"] + actionness_stats["TN"]) / (actionness_stats["TP"] + actionness_stats["FN"] + actionness_stats["FP"] + actionness_stats["TN"])
                    postfix_kwargs["loss_CE"] = loss.item()
                    postfix_kwargs["loss_offset"] = loss_off.item()
                    postfix_kwargs["acc_ant"] = acc
                    postfix_kwargs["acc_offset"] = 1 if n_total == 0 else off_correct/n_total
                    if use_actionness: postfix_kwargs["acc_actionness"] = (actionness_stats["TP"] + actionness_stats["TN"]) / (actionness_stats["TP"] + actionness_stats["FN"] + actionness_stats["FP"] + actionness_stats["TN"])
                    if val_epoch_class_stats is None:
                        val_epoch_class_stats = class_stats
                    else:
                        for class_key in class_stats.keys():
                            for class_stat in class_stats[class_key].keys():
                                val_epoch_class_stats[class_key][class_stat] += class_stats[class_key][class_stat]
                    if use_actionness:
                        if val_epoch_actionness_stats is None:
                            val_epoch_actionness_stats = actionness_stats
                        else:
                            for stat_key in actionness_stats.keys():
                                val_epoch_actionness_stats[stat_key] += actionness_stats[stat_key]


                # Record and display full training statistics for the iteration
                val_epoch_loss += losses.item()
                step_log_dict["val/full_loss"] = losses.item()
                postfix_kwargs["loss"] = losses.item()
                val_loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
                val_loop.set_postfix(**postfix_kwargs)
                if args.seg:
                    inv_class_dict[0] = "BACKGROUND"
                    step_log_dict = log_class_metrics(step_log_dict, seg_class_stats, "val/seg", inv_class_dict)
                if args.anticipate:
                    inv_class_dict[0] = "EOS"
                    step_log_dict = log_class_metrics(step_log_dict, class_stats, "val/anticipate", inv_class_dict)
                wandb.log(step_log_dict)

        ########################################
        # Full epoch statistics
        ########################################
        epoch_loss = epoch_loss / (i+1)
        val_epoch_loss = val_epoch_loss / (j+1)
        print("Epoch [", (epoch+1), '/', args.epochs, '] Loss : %.3f'%epoch_loss, 'Val Loss: %.3f'%val_epoch_loss)
        epoch_log_dict = {"epoch": epoch+1, "epoch/full_loss": epoch_loss, "epoch/val_full_loss": val_epoch_loss}
        if args.anticipate :
            inv_class_dict[0] = "EOS"
            accuracy = total_class_correct/total_class
            epoch_loss_class = epoch_loss_class / (i+1)
            val_accuracy = val_total_class_correct/val_total_class
            val_epoch_loss_class = val_epoch_loss_class / (j+1)
            print('Training Acc :%.3f'%accuracy, 'CE loss :%.3f'%epoch_loss_class )
            print('Val Acc :%.3f'%val_accuracy, 'CE loss :%.3f'%val_epoch_loss_class )
            epoch_log_dict["epoch/CE_acc"] = accuracy
            epoch_log_dict["epoch/CE_loss"] = epoch_loss_class
            epoch_log_dict["epoch/val_CE_acc"] = val_accuracy
            epoch_log_dict["epoch/val_CE_loss"] = val_epoch_loss_class
            epoch_loss_off = epoch_loss_off / (i+1)
            offset_acc = total_off_correct / total_class
            val_epoch_loss_off = val_epoch_loss_off / (j+1)
            val_offset_acc = val_total_off_correct / val_total_class
            print('offset acc:%.5f'%offset_acc,'offset loss: %.5f'%epoch_loss_off)
            print('val offset acc:%.5f'%val_offset_acc,'val offset loss: %.5f'%val_epoch_loss_off)
            epoch_log_dict["epoch/offset_loss"] = epoch_loss_off
            epoch_log_dict["epoch/offset_acc"] = offset_acc
            epoch_log_dict["epoch/val_offset_loss"] = val_epoch_loss_off
            epoch_log_dict["epoch/val_offset_acc"] = val_offset_acc
            epoch_log_dict = log_class_metrics(epoch_log_dict, epoch_class_stats, "epoch/train_anticipate", inv_class_dict)
            epoch_log_dict = log_class_metrics(epoch_log_dict, val_epoch_class_stats, "epoch/val_anticipate", inv_class_dict)
            if use_actionness:
                epoch_loss_actionness = epoch_loss_actionness / (i+1)
                val_epoch_loss_actionness = val_epoch_loss_actionness / (j+1)
                epoch_log_dict["epoch/actionness_loss"] = epoch_loss_actionness
                epoch_log_dict["epoch/val_actionness_loss"] = val_epoch_loss_actionness
                epoch_log_dict = log_confusion_matrix(epoch_log_dict, epoch_actionness_stats, "epoch/train_actionness")
                epoch_log_dict = log_confusion_matrix(epoch_log_dict, val_epoch_actionness_stats, "epoch/val_actionness")

        if args.seg :
            inv_class_dict[0] = "BACKGROUND"
            acc_seg = total_seg_correct / total_seg
            val_acc_seg = val_total_seg_correct / val_total_seg
            epoch_loss_seg = epoch_loss_seg / (i+1)
            val_epoch_loss_seg = val_epoch_loss_seg / (j+1)
            print('seg loss :%.3f'%epoch_loss_seg, ', seg acc : %.5f'%acc_seg)
            print('val_seg loss :%.3f'%val_epoch_loss_seg, ', val_seg acc : %.5f'%val_acc_seg)
            epoch_log_dict["epoch/seg_loss"] = epoch_loss_seg
            epoch_log_dict["epoch/seg_acc"] = acc_seg
            epoch_log_dict["epoch/val_seg_loss"] = val_epoch_loss_seg
            epoch_log_dict["epoch/val_seg_acc"] = val_acc_seg
            epoch_log_dict = log_class_metrics(epoch_log_dict, epoch_class_stats_seg, "epoch/train_seg", inv_class_dict)
            epoch_log_dict = log_class_metrics(epoch_log_dict, val_epoch_class_stats_seg, "epoch/val_seg", inv_class_dict)

        epoch_log_dict["epoch/lr"] = optimizer.param_groups[0]['lr']

        ########################################
        # Checkpoint saving (before evaluation to ensure progress is saved)
        ########################################
        save_path = os.path.join(model_save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Save a checkpoint with all necessary information to resume training
        checkpoint_dir = os.path.join(save_path, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "arguments": args,
            "wandb_run_id": wandb.run.id,
            "best_mAP": best_mAP,
            "best_model_path": best_model_path
        }, os.path.join(checkpoint_dir, "checkpoint.ckpt"))

        ########################################
        # Evaluation
        ########################################
        # Evaluate model only after start_map_epoch
        if epoch >= args.start_map_epoch:
            try:
                if args.anticipate:
                    if args.dataset == 'soccernetballanticipation':
                        maps, _, _ = evaluate_BAA("val", model, n_class, class_dict, pad_idx, args, False, use_actionness or BCE_with_actionness, use_anchors)
                    else:
                        maps, _, _ = evaluate("val", model, n_class, class_dict, pad_idx, args, 0.9 if args.n_query == 1 else 1-args.pred_perc, False, use_actionness or BCE_with_actionness, use_anchors)
                    for key in maps.keys():
                        print(key)
                        print(maps[key])
                        epoch_log_dict[f"epoch/map_{key}"] = maps[key]
                # Save a checkpoint for safety
                torch.save(model.state_dict(), os.path.join(save_path, f'checkpoint{epoch+1}.ckpt'))
                # Save the best checkpoint with highest stable tightV2 mAP (Average of 1s, 5s and infinity mAP)
                if maps["tightV2"]["a_mAP_stable"] >= best_mAP:
                    print(f"\nSaving new best model at epoch {epoch+1} with mAP {maps['tightV2']['a_mAP_stable']} (+{maps['tightV2']['a_mAP_stable']-best_mAP})\n")
                    best_model_path = os.path.join(save_path, 'best_checkpoint.ckpt')
                    torch.save(model.state_dict(), best_model_path)
                    wandb.save(best_model_path) # Save best checkpoint on wandb. No other checkpoints are uploaded
                    best_mAP = maps["tightV2"]["a_mAP_stable"]
            except Exception as e:
                print(f"\nWarning: Evaluation failed at epoch {epoch+1}: {e}")
                print("Continuing training without evaluation metrics...\n")
        # Log statistics on wandb
        wandb.log(epoch_log_dict)
        
    return model, best_model_path

# Logs the class-wise statistics into the wandb dictionary
def log_class_metrics(log_dict, class_stats, log_prefix, inv_class_dict):
    for class_key in class_stats.keys():
        for class_stat in class_stats[class_key].keys():
            log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/{class_stat}"] = class_stats[class_key][class_stat]
        if (class_stats[class_key]["TP"] + class_stats[class_key]["FP"] + class_stats[class_key]["TN"] + class_stats[class_key]["FN"]) == 0:
            # If no samples of this class were present, we can't calculate any metrics
            continue
        log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/acc"] = (class_stats[class_key]["TP"] + class_stats[class_key]["TN"]) / (class_stats[class_key]["TP"] + class_stats[class_key]["FP"] + class_stats[class_key]["TN"] + class_stats[class_key]["FN"])
        log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/Prevalence"] = (class_stats[class_key]["TP"]) / (class_stats[class_key]["TP"] + class_stats[class_key]["FP"] + class_stats[class_key]["TN"] + class_stats[class_key]["FN"])
        do_TPR = (class_stats[class_key]["TP"] + class_stats[class_key]["FN"]) > 0
        do_TNR = (class_stats[class_key]["TN"] + class_stats[class_key]["FP"]) > 0
        if do_TPR: log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/TPR"] = class_stats[class_key]["TP"] / (class_stats[class_key]["TP"] + class_stats[class_key]["FN"])
        if do_TNR: log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/TNR"] = class_stats[class_key]["TN"] / (class_stats[class_key]["TN"] + class_stats[class_key]["FP"])
        if do_TPR and do_TNR:
            log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/Balanced_Acc"] = (log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/TPR"] + log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/TNR"]) / 2
        do_precision = (class_stats[class_key]["TP"] + class_stats[class_key]["FP"]) > 0
        do_recall = (class_stats[class_key]["TP"] + class_stats[class_key]["FN"]) > 0
        if do_precision: log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/precision"] = (class_stats[class_key]["TP"]) / (class_stats[class_key]["TP"] + class_stats[class_key]["FP"])
        if do_recall: log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/recall"] = (class_stats[class_key]["TP"]) / (class_stats[class_key]["TP"] + class_stats[class_key]["FN"])
        if do_precision and do_recall:
            log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/f1"] = 0 if class_stats[class_key]["TP"] == 0 else 2 * (log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/precision"] * log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/recall"]) / (log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/precision"] + log_dict[f"{log_prefix}/{inv_class_dict[class_key]}/recall"])
    return log_dict

# Logs the confusion matrix into the wandb dictionary
def log_confusion_matrix(log_dict, confusion_matrix, log_prefix):
    for metric in confusion_matrix.keys():
        log_dict[f"{log_prefix}/{metric}"] = confusion_matrix[metric]
    if (confusion_matrix["TP"] + confusion_matrix["FP"] + confusion_matrix["TN"] + confusion_matrix["FN"]) == 0:
        # If no samples of this class were present, we can't calculate any metrics
        return log_dict
    log_dict[f"{log_prefix}/acc"] = (confusion_matrix["TP"] + confusion_matrix["TN"]) / (confusion_matrix["TP"] + confusion_matrix["FP"] + confusion_matrix["TN"] + confusion_matrix["FN"])
    log_dict[f"{log_prefix}/Prevalence"] = (confusion_matrix["TP"]) / (confusion_matrix["TP"] + confusion_matrix["FP"] + confusion_matrix["TN"] + confusion_matrix["FN"])
    do_TPR = (confusion_matrix["TP"] + confusion_matrix["FN"]) > 0
    do_TNR = (confusion_matrix["TN"] + confusion_matrix["FP"]) > 0
    if do_TPR: log_dict[f"{log_prefix}/TPR"] = confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FN"])
    if do_TNR: log_dict[f"{log_prefix}/TNR"] = confusion_matrix["TN"] / (confusion_matrix["TN"] + confusion_matrix["FP"])
    if do_TPR and do_TNR:
        log_dict[f"{log_prefix}/Balanced_Acc"] = (log_dict[f"{log_prefix}/TPR"] + log_dict[f"{log_prefix}/TNR"]) / 2
    do_precision = (confusion_matrix["TP"] + confusion_matrix["FP"]) > 0
    do_recall = (confusion_matrix["TP"] + confusion_matrix["FN"]) > 0
    if do_precision: log_dict[f"{log_prefix}/precision"] = (confusion_matrix["TP"]) / (confusion_matrix["TP"] + confusion_matrix["FP"])
    if do_recall: log_dict[f"{log_prefix}/recall"] = (confusion_matrix["TP"]) / (confusion_matrix["TP"] + confusion_matrix["FN"])
    if do_precision and do_recall:
        log_dict[f"{log_prefix}/f1"] = 0 if confusion_matrix["TP"] == 0 else 2 * (log_dict[f"{log_prefix}/precision"] * log_dict[f"{log_prefix}/recall"]) / (log_dict[f"{log_prefix}/precision"] + log_dict[f"{log_prefix}/recall"])
    return log_dict
