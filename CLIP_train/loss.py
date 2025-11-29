import torch
import torch.nn.functional as F

def clip_loss_with_negatives(model, processor, caption, pos_imgs, neg_imgs, device, loss_setting):
    # ==============================================
    # LOSS SETTING 1: POS + NEG (Sigmoid / BCE)     |
    # ==============================================
    if loss_setting == 1:
        all_imgs = pos_imgs + neg_imgs
        if len(all_imgs) == 0:
            return None

        inputs = processor(
            text=[caption] * len(all_imgs),
            images=all_imgs,
            return_tensors="pt",
            padding=True, truncation=True
        ).to(device)

        outputs = model(**inputs)

        img_embeds = outputs.image_embeds
        txt_embeds = outputs.text_embeds

        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)

        sims = (txt_embeds * img_embeds).sum(dim=-1) * model.logit_scale.exp()
        probs = torch.sigmoid(sims)

        labels = torch.zeros_like(probs)
        labels[:len(pos_imgs)] = 1.0  # pos = 1, neg = 0

        return F.binary_cross_entropy(probs, labels)

    # ==============================================
    # LOSS SETTING 2: CHỈ POS (Sigmoid / BCE)       |
    # ==============================================
    elif loss_setting == 2:
        if len(pos_imgs) == 0:
            return None

        inputs = processor(
            text=[caption] * len(pos_imgs),
            images=pos_imgs,
            return_tensors="pt",
            padding=True, truncation=True
        ).to(device)

        outputs = model(**inputs)

        img_embeds = outputs.image_embeds
        txt_embeds = outputs.text_embeds

        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)

        sims = (txt_embeds * img_embeds).sum(dim=-1) * model.logit_scale.exp()
        probs = torch.sigmoid(sims)

        labels = torch.ones_like(probs)  # tất cả đều POS

        return F.binary_cross_entropy(probs, labels)

    # ==============================================
    # LOSS SETTING 3: CHUẨN CLIP CONTRASTIVE LOSS   |
    # ==============================================
    elif loss_setting == 3:
        if len(pos_imgs) == 0:
            return None

        inputs = processor(
            text=[caption] * len(pos_imgs),
            images=pos_imgs,
            return_tensors="pt",
            padding=True, truncation=True
        ).to(device)

        outputs = model(**inputs)

        img_embeds = outputs.image_embeds        # shape: [N, D]
        txt_embeds = outputs.text_embeds        # shape: [N, D]

        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)

        # similarity matrix: [N, N]
        logits_per_text = txt_embeds @ img_embeds.t() * model.logit_scale.exp()
        logits_per_image = img_embeds @ txt_embeds.t() * model.logit_scale.exp()

        labels = torch.arange(len(pos_imgs), dtype=torch.long, device=device)

        loss_t = F.cross_entropy(logits_per_text, labels)
        loss_i = F.cross_entropy(logits_per_image, labels)

        return (loss_t + loss_i) / 2

    # fallback
    return None
