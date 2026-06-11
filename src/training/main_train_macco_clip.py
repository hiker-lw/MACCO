import logging
import os
import random
import pytz
from datetime import datetime
from pathlib import Path

import pdb
import numpy as np
import torch
from torch import optim
from torch.amp import GradScaler

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from open_clip_code import create_model_and_transforms, trace_model, MACCO_CLIP_factory
from training.data_macco import get_data
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, cosine_lr_two_stage
from training.train import train_one_epoch_macco_clip, evaluate
from training.evaluate_compositional_benchmark import evaluate_compositional_benchmark
from training.utils import create_transform_with_tracking

import yaml

# solve the ssl certificate error
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

def load_config(CONFIG_PATH, config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

def merge_config_to_args(args, config):
    for k, v in config.items():
        setattr(args, k, v)

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    # add another items
    torch.cuda.manual_seed_all(seed+rank)

def _freeze_params(module):
    if hasattr(module, "parameters"):
        for param in module.parameters():
            param.requires_grad = False
    else:
        module.requires_grad = False

def _fire_params(module):
    if hasattr(module, "parameters"):
        for param in module.parameters():
            param.requires_grad = True
    else:
        module.requires_grad = True

def main(config):
    torch.autograd.set_detect_anomaly(False)
    args = parse_args()

    # merge yaml config file to args
    merge_config_to_args(args, config)

    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    args.model = args.model.replace('/', '-')

    # get the name of the experiments
    if args.mask_text_image:
        args.name = '-'.join([
            args.name,
            f"{args.macco_clip_version}",
            datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y_%m_%d-%H_%M"),
            f"model_{args.model}",
            f"lr_clip_{args.lr_clip}",
            f"lr_predictor_{args.lr_predictor}",
            f"b_{args.batch_size}",
            f"loss_{args.loss_fn}",
            f"mt_loss_w_{args.mask_text_loss_w}",
            f"mi_loss_w_{args.mask_image_loss_w}",
            f"IMC_loss_w_{args.intra_modal_contrast_loss_w}"

        ])
    elif args.mask_image:
        args.name = '-'.join([
            args.name,
            f"{args.macco_clip_version}",
            datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y_%m_%d-%H_%M"),
            f"model_{args.model}",
            f"lr_clip_{args.lr_clip}",
            f"lr_predictor_{args.lr_predictor}",
            f"b_{args.batch_size}",
            f"loss_{args.loss_fn}",
            f"mask_image_loss_w_{args.mask_image_loss_w}"
        ])
    else:
        args.name = '-'.join([
            args.name,
            f"{args.macco_clip_version}",
            datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y_%m_%d-%H_%M"),
            f"model_{args.model}",
            f"lr_clip_{args.lr_clip}",
            f"lr_predictor_{args.lr_predictor}",
            f"b_{args.batch_size}",
            f"loss_{args.loss_fn}",
            f"mask_text_loss_w_{args.mask_text_loss_w}"
        ])
    # discover initial world args early so we can log properly
    # args.distributed = True
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path):
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = init_distributed_device(args)

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if is_master(args):
        args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''

    if args.copy_codebase:
        copy_codebase(args)

    assert args.precision in ['amp', 'fp16', 'fp32']
    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 0)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image        
    )
    if args.mask_image or args.mask_text_image:
        preprocess_train, get_transform_params = create_transform_with_tracking(image_size=model.visual.image_size)
    
    MACCO_CLIP = MACCO_CLIP_factory[args.macco_clip_version]
    logging.info(f'Running with MACCO_CLIP {args.macco_clip_version}')
    
    model = MACCO_CLIP(model, args).to(device)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True, **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None
    if args.train_data:
        assert not args.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        clip = lambda n, p: "CLIP" in n
        predictor = lambda n, p: not clip(n,p)

        named_parameters = list(model.named_parameters())
        # gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        # rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        gain_or_bias_params_clip = [p for n, p in named_parameters if exclude(n, p) and clip(n,p) and p.requires_grad]
        other_params_clip = [p for n, p in named_parameters if include(n, p) and clip(n,p) and p.requires_grad]

        params_predictor = [p for n, p in named_parameters if predictor(n, p) and p.requires_grad]
        
        if hasattr(model, 'module'):
            m = model.module.CLIP
        else:
            m = model.CLIP
        # freeze CLIP
        if args.freeze_clip:
            if is_master(args):
                logging.info("Freeze CLIP: True")
            _freeze_params(m.transformer)
            _freeze_params(m.positional_embedding)
            _freeze_params(m.text_projection)
            _freeze_params(m.token_embedding)
            _freeze_params(m.ln_final)
            _freeze_params(m.visual)
        else:
            if is_master(args):
                logging.info("Freeze CLIP: False")
            
        if args.freeze_image_encoder:
            if is_master(args):
                logging.info("Freeze CLIP image encoder: True")
            _freeze_params(m.visual)
        else:
            if is_master(args):
                logging.info("Freeze CLIP image encoder: False")
        
        if args.freeze_text_encoder:
            if is_master(args):
                logging.info("Freeze CLIP text encoder: True")
            _freeze_params(m.transformer)
            _freeze_params(m.positional_embedding)
            _freeze_params(m.text_projection)
            _freeze_params(m.token_embedding)
            _freeze_params(m.ln_final)
        else:
            if is_master(args):
                logging.info("Freeze CLIP text encoder: False")

        param_groups = [
                {"params": gain_or_bias_params_clip, "weight_decay": 0. , "lr": args.lr_clip},
                {"params": other_params_clip, "weight_decay": args.wd, "lr": args.lr_clip},
                {"params": params_predictor, "weight_decay": 0.01, "lr": args.lr_predictor},
            ]
        optimizer = optim.AdamW(
            param_groups,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_master(args):
        logging.info(f"Number of trainable parameters: {n_parameters}")
    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            if 'epoch' in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                #start_epoch = checkpoint["epoch"]
                start_epoch = 0
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                #if optimizer is not None:
                    #optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    # initialize datasets
    if args.mask_image or args.mask_text_image:
        data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, get_transform_params=get_transform_params)
    else:
        data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch)
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        if args.two_stage:
            steps_first_stage = data["train"].dataloader.num_batches * args.first_stage_epochs
            steps_second_stage = data["train"].dataloader.num_batches * args.second_stage_epochs
            scheduler = cosine_lr_two_stage(optimizer, args.lr_clip, args.lr_predictor, args.warmup, steps_first_stage, steps_second_stage)
        else:
            total_steps = data["train"].dataloader.num_batches * args.epochs
            scheduler = cosine_lr(optimizer, args.lr_clip, args.lr_predictor, args.warmup, total_steps)
    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project="MACCO_CLIP",
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if 'train' not in data:
        evaluate(model, data, start_epoch, args, writer)
        return

    # evaluate before training
    if args.val_before_train:
        evaluate_compositional_benchmark(model, preprocess_val, args, device, completed_epoch=0)
    

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')
        if epoch < args.first_stage_epochs:
            _freeze_params(model.CLIP)
        else:
            _fire_params(model.CLIP)
            # _freeze_params(model.prediction_text)
        #train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        # initialize datasets
        if args.rebuild_dataset_every_epoch:
            if args.mask_image or args.mask_text_image:
                data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, get_transform_params=get_transform_params)
            else:
                data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch)

        train_one_epoch_macco_clip(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        completed_epoch = epoch + 1

        # evaluate
        if not args.no_val:
            evaluate_compositional_benchmark(model, preprocess_val, args, device, completed_epoch)
        else:
            logging.info(f'skipping validation on compositional benchmark!')

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            pass

        # Saving checkpoints.
        if args.save_logs:
            # if args.EMA:
            #     model.apply_ema()  # Ensure EMA weights are applied before saving
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model.CLIP.state_dict(),
                # "state_dict": model.state_dict(),
                # "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.save_most_recent:
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
                )

    if args.wandb and is_master(args):
        wandb.finish()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    import wandb
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
    config_name = "MACCO_CLIP.yaml"
    config = load_config(CONFIG_PATH, config_name) 
    main(config)
