import torch
import torch.nn.functional as F

def clip_loss_with_negatives(model, processor, caption, pos_imgs, neg_imgs, device):
    import torch
    import torch.nn.functional as F

    all_imgs = pos_imgs + neg_imgs
    if len(all_imgs) == 0:
        return None

    # Encode
    inputs = processor(
        text=[caption] * len(all_imgs),
        images=all_imgs,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    outputs = model(**inputs)

    img_embeds = outputs.image_embeds
    txt_embeds = outputs.text_embeds

    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
    txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)

    sims = (txt_embeds * img_embeds).sum(dim=-1) * model.logit_scale.exp()
    probs = torch.sigmoid(sims)

    # TRUE LABELS: positive = 1, negative = 0
    labels = torch.zeros_like(probs)
    labels[:len(pos_imgs)] = 1.0

    loss = F.binary_cross_entropy(probs, labels)

    return loss