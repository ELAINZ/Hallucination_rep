import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import OrderedDict
from unets import UNet
from analyse import classify_simple_shapes_cv

def load_unet(model_ckpt, device):
    model = UNet(image_size=64, in_channels=1, out_channels=1)
    ckpt = torch.load(model_ckpt, map_location=device)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    elif "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if any(k.startswith("module.") for k in ckpt.keys()):
        ckpt = OrderedDict((k.replace("module.", ""), v) for k, v in ckpt.items())
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded UNet from {model_ckpt}")
    return model

def extract_latent_features(model, X, device):
    feats = []
    buffer = {}

    def hook_fn(_, __, output):
        buffer["feat"] = output.detach()

    layer = list(model.output_blocks)[-1][0].out_layers[-1]
    handle = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for i in range(0, len(X), 64):
            batch = torch.tensor(X[i:i+64]).float().to(device)
            if batch.ndim == 4 and batch.shape[-1] == 1:
                batch = batch.squeeze(-1)
            if batch.ndim == 3:
                batch = batch.unsqueeze(1)
            batch = batch / 255.

            t = torch.zeros(len(batch), dtype=torch.long, device=device)
            buffer["feat"] = None
            _ = model(batch, t)
            if buffer["feat"] is None:
                print(f"Hook failed at batch {i}")
                continue
            feat = buffer["feat"].cpu().numpy().reshape(len(batch), -1)
            feats.append(feat)

    handle.remove()
    feats = np.concatenate(feats, axis=0)
    print(f"Extracted features: {feats.shape}")
    return feats

def plot_tsne_region(emb_all, labels, save_path="tsne_square_region_split.png"):
    labels = np.array(labels)

    mask_1 = np.array(["1_square" in lbl for lbl in labels])
    mask_2 = np.array(["2_square" in lbl for lbl in labels])
    mask_other = ~(mask_1 | mask_2)

    emb_all[:, 0] = -emb_all[:, 0]

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_facecolor("#f0f0f0")                  
    ax.grid(True, color="white", linewidth=1, zorder=0) 

    plt.scatter(emb_all[mask_other, 0], emb_all[mask_other, 1],
                s=12, c="lightgray", alpha=0.4, marker='D', label="others", zorder=2)
    plt.scatter(emb_all[mask_1, 0], emb_all[mask_1, 1],
                s=20, alpha=0.9, c="#0000FF", marker='D', label="1_square", zorder=3)
    plt.scatter(emb_all[mask_2, 0], emb_all[mask_2, 1],
                s=20, alpha=0.9, c="#FF8C00", marker='D', label="2_square", zorder=3)

    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.title("t-SNE of UNet latent features (1_square & 2_square region)", fontsize=13)

    plt.tight_layout()
    plt.legend(fontsize=8, frameon=False)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved t-SNE figure to {save_path}")

def main():
    npz_path = "trained_models/simple-shapes_ddpm-md_T_1000_bs_32_random_700_850_seed_1234_test-single/gen_9_generated_data_epoch_50-timesteps_1000_sampling-steps_100.npz"
    model_ckpt = "trained_models/simple-shapes_ddpm-md_T_1000_bs_32_random_700_850_seed_1234_test-single/gen-9_UNet_simple-shapes-epoch_50-timesteps_1000-class_condn_False.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(npz_path)
    X = data["X"]
    if X.ndim == 4 and X.shape[-1] == 1:
        X = X.squeeze(-1)
    print(f"Loaded {len(X)} images")

    labels_dict = classify_simple_shapes_cv(X, gen=9, args=type("obj",(object,),{"log_results":False,"local_rank":0})())
    labels = [labels_dict[i] for i in range(len(X))]
    print("Example labels:", labels[:15])

    model = load_unet(model_ckpt, device)
    feats_all = extract_latent_features(model, X, device)

    perplexity = max(5, min(30, len(X)//2))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    emb_all = tsne.fit_transform(feats_all)
    print(f"t-SNE embedding done: {emb_all.shape}")

    plot_tsne_region(emb_all, labels, save_path="tsne_square_region.png")

if __name__ == "__main__":
    main()
