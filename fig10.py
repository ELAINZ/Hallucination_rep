import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import OrderedDict

# -----------------------------
# 1. 加载样本
# -----------------------------
def load_filtered_data(npz_path):
    data = np.load(npz_path)
    X = data["X"]
    y = data["y"] if "y" in data else None
    if X.ndim == 4 and X.shape[-1] == 1:
        X = X.squeeze(-1)
    print(f"Loaded {len(X)} images with shape {X.shape[1:]} from {npz_path}")
    return X, y


# -----------------------------
# 2. 加载模型（自动去除 module.）
# -----------------------------
def load_unet(model_ckpt, device):
    from unets import UNet
    model = UNet(image_size=64, in_channels=1, out_channels=1)
    ckpt = torch.load(model_ckpt, map_location=device)
    if 'model' in ckpt:
        ckpt = ckpt['model']
    elif 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    if any(k.startswith('module.') for k in ckpt.keys()):
        ckpt = OrderedDict((k.replace('module.', ''), v) for k, v in ckpt.items())
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded UNet checkpoint from {model_ckpt}")
    return model

# -----------------------------
# 3. 提取中间层特征
# -----------------------------
def extract_latent_features(model, X, device="cuda"):
    feats = []
    handles = []
    buffer = {}

    def hook_fn(_, __, output):
        buffer['feat'] = output.detach()

    # 注册 hook：抓取某一层输出（比如最后一层卷积前）
    layer = list(model.output_blocks)[-1][0].out_layers[-1]
    handle = layer.register_forward_hook(hook_fn)
    handles.append(handle)

    with torch.no_grad():
        for i in range(0, len(X), 64):
            batch = torch.tensor(X[i:i+64]).float().unsqueeze(1).to(device) / 255.
            t = torch.zeros(len(batch), dtype=torch.long, device=device)
            _ = model(batch, t)  # forward 一次
            feat = buffer['feat'].cpu().numpy().reshape(len(batch), -1)
            feats.append(feat)

    for h in handles:
        h.remove()
    feats = np.concatenate(feats, axis=0)
    print(f"Extracted features: {feats.shape}")
    return feats

# -----------------------------
# 4. t-SNE 可视化
# -----------------------------
def tsne_plot(features, labels=None, save_path="tsne_plot_squares_only.png"):
    tsne = TSNE(n_components=2, random_state=0, perplexity=30)
    emb = tsne.fit_transform(features)

    plt.figure(figsize=(6, 6))
    if labels is None:
        plt.scatter(emb[:, 0], emb[:, 1], s=5, alpha=0.6, c="gray")
    else:
        labels = np.array(labels)
        classes = np.unique(labels)
        cmap = plt.get_cmap("tab10" if len(classes) <= 10 else "tab20")

        for i, cls in enumerate(classes):
            mask = labels == cls
            plt.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=8,
                alpha=0.7,
                color=cmap(i),
                label=str(cls),
            )

        plt.legend(
            title="Digit",
            loc="best",
            fontsize=8,
            title_fontsize=9,
            frameon=True,
            edgecolor="black",
        )

    plt.title("t-SNE of UNet latent features (Gen 0, Epoch 50)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved t-SNE with legend to {save_path}")
    return emb


# -----------------------------
# 5. 主函数
# -----------------------------
def main():
    npz_path = "trained_models/simple-shapes_ddpm-md_T_1000_bs_32_random_700_850_seed_1234_test-single/gen_9_generated_data_epoch_50-timesteps_1000_sampling-steps_100.npz"
    model_ckpt = "trained_models/simple-shapes_ddpm-md_T_1000_bs_32_random_700_850_seed_1234_test-single/gen-9_UNet_simple-shapes-epoch_50-timesteps_1000-class_condn_False.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y = load_filtered_data(npz_path)
    model = load_unet(model_ckpt, device)
    feats = extract_latent_features(model, X, device)
    tsne_plot(feats, labels=y)

if __name__ == "__main__":
    main()
