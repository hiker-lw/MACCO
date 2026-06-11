from .MACCO_CLIP_loss_text_image_raw_space import macco_clip_loss_text_image

loss_factory = {
    "loss_text_image": macco_clip_loss_text_image,
}