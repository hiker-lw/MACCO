import numpy as np


def assign_learning_rate(optimizer, clip_new_lr, predictor_new_lr):
    
    optimizer.param_groups[0]["lr"] = clip_new_lr
    optimizer.param_groups[1]["lr"] = clip_new_lr
    optimizer.param_groups[2]["lr"] = predictor_new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, clip_base_lr, predictor_base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr_clip = _warmup_lr(clip_base_lr, warmup_length, step)
            lr_predictor = _warmup_lr(predictor_base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr_clip = 0.5 * (1 + np.cos(np.pi * e / es)) * clip_base_lr
            lr_predictor = 0.5 * (1 + np.cos(np.pi * e / es)) * predictor_base_lr
        assign_learning_rate(optimizer, lr_clip, lr_predictor)
        return lr_clip, lr_predictor
    return _lr_adjuster

def cosine_lr_two_stage(optimizer, clip_base_lr, predictor_base_lr, warmup_length, steps_first_stage, steps_second_stage):
    def _lr_adjuster(step):
        if step < steps_first_stage:
            # first stage: training predictor
            lr_clip = 0
            if step < warmup_length:
                lr_predictor = _warmup_lr(predictor_base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps_first_stage + steps_second_stage - warmup_length
                lr_predictor = 0.5 * (1 + np.cos(np.pi * e / es)) * predictor_base_lr
        else:
            # second stage: training CLIP
            e = step - warmup_length
            es = steps_first_stage + steps_second_stage - warmup_length
            lr_predictor = 0.5 * (1 + np.cos(np.pi * e / es)) * predictor_base_lr
            
            if step - steps_first_stage < warmup_length:
                lr_clip = _warmup_lr(clip_base_lr, warmup_length, step - steps_first_stage)
            else:
                e = step - steps_first_stage - warmup_length
                es = steps_second_stage - warmup_length
                lr_clip = 0.5 * (1 + np.cos(np.pi * e / es)) * clip_base_lr

        assign_learning_rate(optimizer, lr_clip, lr_predictor)
        return lr_clip, lr_predictor
    return _lr_adjuster