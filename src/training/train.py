import json
import logging
import math
import os
import time
from contextlib import suppress

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from open_clip_code import ClipLoss, loss_factory
from .distributed import is_master
from .zero_shot import zero_shot_eval

activation_layer_0 = []
grad_layer_0 = []
activation_layer_5 = []
grad_layer_5 = []
activation_layer_11 = []
grad_layer_11 = []

def get_activation_hook_layer_0(module, input, output):
    activation_layer_0.append(output)

def get_activation_hook_layer_5(module, input, output):
    activation_layer_5.append(output)

def get_activation_hook_layer_11(module, input, output):
    activation_layer_11.append(output)

def get_grad_hook_layer_0(module, grad_input, grad_output):
    grad_layer_0.append(grad_output)

def get_grad_hook_layer_5(module, grad_input, grad_output):
    grad_layer_5.append(grad_output)

def get_grad_hook_layer_11(module, grad_input, grad_output):
    grad_layer_11.append(grad_output)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.amp.autocast if args.precision == 'amp' else suppress

    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        images, new_images, texts, new_texts, hard_captions, new_hard = batch

        images = images.to(device=device, non_blocking=True)
        new_images = new_images.to(device=device, non_blocking=True)

        texts = texts.to(device=device, non_blocking=True)
        new_texts = new_texts.to(device=device, non_blocking=True)

        hard_captions = hard_captions.to(device=device, non_blocking=True)
        new_hard = new_hard.to(device=device, non_blocking=True)

        images = torch.cat([images, new_images])

        texts = torch.cat([texts, new_texts])
        texts = torch.cat([texts, hard_captions])
        texts = torch.cat([texts, new_hard])

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            total_loss = loss(image_features, text_features, logit_scale)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.norm_gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.norm_gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} - V4"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

def train_one_epoch_clip(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.amp.autocast if args.precision == 'amp' else suppress

    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        
        images, new_images, texts, new_texts, hard_captions, new_hard = batch

        images = images.to(device=device, non_blocking=True)
        new_images = new_images.to(device=device, non_blocking=True)

        texts = texts.to(device=device, non_blocking=True)
        new_texts = new_texts.to(device=device, non_blocking=True)

        hard_captions = hard_captions.to(device=device, non_blocking=True)
        new_hard = new_hard.to(device=device, non_blocking=True)

        if args.add_neg_sample:
            images = torch.cat([images, new_images])

            texts = torch.cat([texts, new_texts])
            texts = torch.cat([texts, hard_captions])
            texts = torch.cat([texts, new_hard])

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            total_loss = loss_factory[args.loss_fn](args=args)(image_features, text_features, logit_scale)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.norm_gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.norm_gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} - V4"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def train_one_epoch_macco_clip(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    # with gradient accumulation
    device = torch.device(args.device)
    autocast = torch.amp.autocast if args.precision == 'amp' else suppress

    # Set gradient accumulation steps. The default of 1 preserves the original behavior.
    accumulation_steps = args.gradient_accumulation_steps if hasattr(args, 'gradient_accumulation_steps') else 1

    model.train()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_fn = loss_factory[args.loss_fn](args=args)
    loss_name_list = loss_fn.get_loss_keys()
    loss_name_list.remove("total_loss")
    
    total_loss_m = AverageMeter()
    
    meter_dict = {}
    batch_loss_dict = {}
    for loss_name in loss_name_list:
        meter_dict[loss_name] = AverageMeter()

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # Reset gradients only at the start of an accumulation window.
    optimizer.zero_grad()
    
    for i, batch in enumerate(dataloader):
        # step = num_batches_per_epoch * epoch + i
        # scheduler(step)
        # Compute the optimizer step index with accumulation accounted for.
        real_step = num_batches_per_epoch * epoch + i // accumulation_steps
                
        if args.mask_text_image:
            images, texts, hard_captions, comp_concept_token_flag, object_token_flag, image_token_mask_flag = batch

            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            if args.add_neg_sample: # Whether to use negative samples.
                hard_captions = hard_captions.to(device=device, non_blocking=True)
                texts = torch.cat([texts, hard_captions])
            
            data_time_m.update(time.time() - end)

            with autocast('cuda'):
                output_dict = model(images, texts, image_token_mask_flag, comp_concept_token_flag)
                logit_scale = output_dict["logit_scale"]
                loss = loss_factory[args.loss_fn](args=args)(output_dict)
                total_loss = loss["total_loss"] / accumulation_steps
                for loss_name in loss_name_list:
                    batch_loss_dict[loss_name] = loss[loss_name] / accumulation_steps

        elif args.mask_image:
            images, texts, hard_captions, image_token_mask_flag = batch
            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            if args.add_neg_sample: # Whether to use negative samples.
                hard_captions = hard_captions.to(device=device, non_blocking=True)
                texts = torch.cat([texts, hard_captions])
            
            data_time_m.update(time.time() - end)

            with autocast('cuda'):
                output_dict = model(images, texts, image_token_mask_flag)
                logit_scale = output_dict["logit_scale"]
                loss = loss_factory[args.loss_fn](args=args)(output_dict)
                total_loss = loss["total_loss"] / accumulation_steps
                for loss_name in loss_name_list:
                    batch_loss_dict[loss_name] = loss[loss_name] / accumulation_steps
        else:
            images, texts, hard_captions, comp_concept_token_flag, object_token_flag = batch

            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            if args.add_neg_sample: # Whether to use negative samples.
                hard_captions = hard_captions.to(device=device, non_blocking=True)
                texts = torch.cat([texts, hard_captions])
                
            data_time_m.update(time.time() - end)

            with autocast('cuda'):
                if args.mask_token_type == "object":
                    output_dict = model(images, texts, object_token_flag)
                elif args.mask_token_type == "relation_and_attribute":
                    output_dict = model(images, texts, comp_concept_token_flag)
                
                logit_scale = output_dict["logit_scale"]
                loss = loss_factory[args.loss_fn](args=args)(output_dict)
                total_loss = loss["total_loss"] / accumulation_steps
                for loss_name in loss_name_list:
                    batch_loss_dict[loss_name] = loss[loss_name] / accumulation_steps
        
        if scaler is not None:
            # import ipdb; ipdb.set_trace()
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.norm_gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                if (i + 1) % accumulation_steps == 0 or (i + 1) == num_batches_per_epoch: # Gradient accumulation.
                    scheduler(real_step)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            else:
                if args.norm_gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                if (i + 1) % accumulation_steps == 0 or (i + 1) == num_batches_per_epoch: # Gradient accumulation.
                    scheduler(real_step)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
        else:
            total_loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            if (i + 1) % accumulation_steps == 0 or (i + 1) == num_batches_per_epoch: # Gradient accumulation.
                scheduler(real_step)
                optimizer.step()
                optimizer.zero_grad()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == num_batches_per_epoch: # Gradient accumulation.
            # Note: we clamp to 4.6052 = ln(100), as in the original paper.
            with torch.no_grad():
                unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        # EMA update
        if args.EMA:
            model.update_ema()

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            total_loss_m.update(total_loss.item() * accumulation_steps, batch_size)
            for loss_name in loss_name_list:
                meter_dict[loss_name].update(batch_loss_dict[loss_name].item() * accumulation_steps, batch_size)    
            
            logit_scale_scalar = logit_scale.item()
            loss_log = ""
            for loss_name in loss_name_list:
                loss_log = loss_log + f"{loss_name}: {meter_dict[loss_name].val:#.5g} ({meter_dict[loss_name].avg:#.4g}) "
            
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Total Loss: {total_loss_m.val:#.5g} ({total_loss_m.avg:#.4g}) "
                + loss_log + 
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR_CLIP: {optimizer.param_groups[0]['lr']:.10f} "
                f"LR_predictor: {optimizer.param_groups[2]['lr']:.10f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "total_loss": total_loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr_clip": optimizer.param_groups[0]["lr"],
                "lr_predictor": optimizer.param_groups[2]["lr"],
            }
            for loss_name in loss_name_list:
                log_data[f"{loss_name}"] = meter_dict[loss_name].val
            
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, real_step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': real_step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = torch.amp.autocast if args.precision == 'amp' else suppress
    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, hard_images, texts, texts_hard_images, hard_captions, hard_captions_of_hard_images = batch

                images = images.to(device=device, non_blocking=True)
                hard_images = hard_images.to(device=device, non_blocking=True)

                texts = texts.to(device=device, non_blocking=True)
                texts_hard_images = texts_hard_images.to(device=device, non_blocking=True)

                hard_captions = hard_captions.to(device=device, non_blocking=True)
                hard_captions_of_hard_images = hard_captions_of_hard_images.to(device=device, non_blocking=True)

                images = torch.cat([images, hard_images])

                texts = torch.cat([texts, texts_hard_images])
                texts = torch.cat([texts, hard_captions])
                texts = torch.cat([texts, hard_captions_of_hard_images])

                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic

                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features[:len(logits_per_image)].cpu())

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text[:len(logits_per_image)], labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text[:len(logits_per_image)]}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
