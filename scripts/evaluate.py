import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
    roc_curve,
    auc,
    RocCurveDisplay
)
import wandb
from datasets.dvisionlab.dvisionlab import get_dvisionlab_dataloader
from datasets.inbreast.inbreast import get_inbreast_dataloader
from datasets.cbis.cbis import get_cbis_dataloader
from models.mvswintransformer import SwinTransformer_singleview, MVSwinTransformer

# -------------------------------
# ARGPARSE
# -------------------------------
parser = argparse.ArgumentParser(description="Evaluation script for Breast Density")
parser.add_argument('--dataset', type=str, choices=['dvisionlab', 'inbreast', 'cbis'], required=True)
parser.add_argument('--view_mode', type=str, choices=['single', 'multi'], default='multi')
parser.add_argument('--csv_path', type=str, required=True)
parser.add_argument('--images_root', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--img_size', type=int, default=384)
parser.add_argument('--window_size', type=int, default=4)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--embed_dim', type=int, default=96)
parser.add_argument('--num_classes', type=int, required=True)
parser.add_argument('--view_type', type=str, choices=['CC','MLO'], default=None)


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# WANDB INIT
# -------------------------------
wandb.init(project=f"breast_density_evaluation_metrics", name=f"{args.dataset}_{args.view_mode}_{args.img_size}_eval")

# -------------------------------
# DATALOADER
# -------------------------------
if args.dataset == 'dvisionlab':
    get_loader = get_dvisionlab_dataloader
elif args.dataset == 'inbreast':
    get_loader = get_inbreast_dataloader
elif args.dataset == 'cbis':
    get_loader = get_cbis_dataloader

_, val_loader, _, val_ds = get_loader(
    csv_path=args.csv_path,
    images_root=args.images_root,
    out_dir='./metrics',
    view_mode=args.view_mode,
    batch_size=args.batch_size,
    img_size=args.img_size,
    num_workers=4,
    keep_incomplete=False,
    use_weighted_sampler=False,
    augment=False,
    view_type=args.view_type
)

# -------------------------------
# MODEL
# -------------------------------
if args.view_mode == 'single':
    model = SwinTransformer_singleview(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=1,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        window_size=args.window_size
    )
else:
    model = MVSwinTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=1,
        num_classes=args.num_classes, 
        embed_dim=args.embed_dim,
        window_size=args.window_size
    )

model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state_dict"])
model = model.to(device)
model.eval()

# -------------------------------
# EVALUATION
# -------------------------------
all_preds = []
all_labels = []
all_probs = []
all_metas = []

with torch.no_grad():
    for batch in val_loader:
        if args.view_mode == 'single':
            inputs, labels, metas = batch
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            outputs = model(inputs)
        else:
            inputs_cc, inputs_mlo, labels, _ = batch
            inputs_cc = inputs_cc.float().to(device)
            inputs_mlo = inputs_mlo.float().to(device)
            labels = labels.long().to(device)
            outputs = model(inputs_cc, inputs_mlo)


        
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        if args.view_mode == 'single':
            all_metas.extend(metas)
        

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()
all_probs = torch.cat(all_probs).numpy()

# -------------------------------
# ACCURACY
# -------------------------------
overall_acc = accuracy_score(all_labels, all_preds)
print(f"Overall Accuracy: {overall_acc:.4f}")
wandb.log({"overall_accuracy": overall_acc})

per_class_acc = {}
for c in range(args.num_classes):
    mask = all_labels == c
    acc_c = accuracy_score(all_labels[mask], all_preds[mask])
    per_class_acc[f"class_{c}_accuracy"] = acc_c
    print(f"Accuracy class {c}: {acc_c:.4f}")
wandb.log(per_class_acc)


# -------------------------------
# F1-SCORE
# -------------------------------

f1_macro = f1_score(all_labels, all_preds, average='macro')
f1_per_class = f1_score(all_labels, all_preds, average=None)
print(f"F1 macro: {f1_macro:.4f}")
for c, f1_c in enumerate(f1_per_class):
    print(f"F1 class {c}: {f1_c:.4f}")
wandb.log({"f1_macro": f1_macro, **{f"f1_class_{i}": f for i,f in enumerate(f1_per_class)}})

# -------------------------------
# CONFUSION MATRIX
# -------------------------------

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, cmap="Blues", values_format='d')
plt.title("Confusion Matrix")
plt.tight_layout()
wandb.log({"confusion_matrix": wandb.Image(fig)})
plt.close(fig)


# -------------------------------
# ROC & AUC per class (improved)
# -------------------------------

# Fixed color palette 
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blue, orange, green, red
class_names = [f"Class {i}" for i in range(args.num_classes)]

fig, ax = plt.subplots(figsize=(8, 6))

for c in range(args.num_classes):
    fpr, tpr, _ = roc_curve(all_labels == c, all_probs[:, c])
    auc_c = auc(fpr, tpr)

    ax.plot(
        fpr, tpr,
        label=f"{class_names[c]} (AUC={auc_c:.3f})",
        color=colors[c],
        linewidth=2
    )

    wandb.log({f"roc_auc_class_{c}": auc_c})

# Diagonal line
ax.plot(
    [0, 1], [0, 1],
    linestyle="--",
    color="black",
    linewidth=1.5
)

# Aesthetics
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves per Class", fontsize=14)

# Legend bottom right
ax.legend(loc="lower right", fontsize=10, frameon=True)

# Better layout
plt.tight_layout()

wandb.log({"roc_curve": wandb.Image(fig)})
plt.close(fig)


# ---------------------------------------
# METRICHE SEPARATE CC vs MLO (single-view)
# ---------------------------------------

if args.view_mode == 'single':
    views = [m['view'] for m in all_metas]

    cc_idx = [i for i, v in enumerate(views) if v == 'CC']
    mlo_idx = [i for i, v in enumerate(views) if v == 'MLO']

    def metrics_subset(name, idx):
        if len(idx) == 0:
            print(f"No samples for {name}")
            return

        y_true = all_labels[idx]
        y_pred = all_preds[idx]
        y_prob = all_probs[idx]

        # -----------------------
        # ACCURACY
        # -----------------------
        acc = accuracy_score(y_true, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")

        # -----------------------
        # F1
        # -----------------------
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_per_class = f1_score(y_true, y_pred, average=None)
        print(f"{name} F1-macro: {f1_macro:.4f}")
        for c, f1_c in enumerate(f1_per_class):
            print(f"{name} F1 class {c}: {f1_c:.4f}")

        wandb.log({
            f"{name}_accuracy": acc,
            f"{name}_f1_macro": f1_macro,
            **{f"{name}_f1_class_{i}": f1 for i, f1 in enumerate(f1_per_class)}
        })

        # -----------------------
        # CONFUSION MATRIX
        # -----------------------
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap="Blues", values_format='d')
        plt.title(f"Confusion Matrix {name}")
        plt.tight_layout()
        wandb.log({f"cm_{name}": wandb.Image(fig)})
        plt.close(fig)

        # -----------------------
        # ROC & AUC per class
        # -----------------------
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for c in range(args.num_classes):
            fpr, tpr, _ = roc_curve(y_true == c, y_prob[:, c])
            auc_c = auc(fpr, tpr)
            ax.plot(
                fpr, tpr,
                label=f"Class {c} (AUC={auc_c:.3f})",
                color=colors[c],
                linewidth=2
            )
            wandb.log({f"{name}_roc_auc_class_{c}": auc_c})

        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves per Class — {name}")
        ax.legend(loc="lower right")
        plt.tight_layout()
        wandb.log({f"roc_curve_{name}": wandb.Image(fig)})
        plt.close(fig)

    # Compute metrics
    metrics_subset("CC", cc_idx)
    metrics_subset("MLO", mlo_idx)