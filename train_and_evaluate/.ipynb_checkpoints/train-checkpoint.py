import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from models.mvswintransformer import SwinTransformer_singleview, MVSwinTransformer
# Import unified dataloaders
from datasets.dvisionlab.dvisionlab import get_dvisionlab_dataloader
from datasets.inbreast.inbreast import get_inbreast_dataloader
from datasets.cbis.cbis import get_cbis_dataloader
from utils import EarlyStopper, load_checkpoint

# -------------------------------
# ARGPARSE: parse command line arguments
# -------------------------------
parser = argparse.ArgumentParser(description="Unified Breast Density Training")
parser.add_argument('--dataset', type=str, choices=['dvisionlab', 'inbreast', 'cbis'], required=True)
parser.add_argument('--view_mode', type=str, choices=['single', 'multi'], default='multi')
parser.add_argument('--csv_path', type=str, required=True)
parser.add_argument('--images_root', type=str, required=True)
parser.add_argument('--out_dir', type=str, default='./datasets')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--patience', type=int, default= 10)
parser.add_argument('--patiences', type=int, default=100)
parser.add_argument('--min_delta', type=float, default=0.001)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--window_size', type=int, default=7)
parser.add_argument('--num_classes', type=int, required=True)
parser.add_argument('--drop_rate', type=float, default=0.1)
parser.add_argument('--attn_drop_rate', type=float, default=0.1)
parser.add_argument('--drop_path_rate', type=float, default=0.1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--embed_dim', type=int, default=96)
parser.add_argument("--weighted_ce", action="store_true",help="Use Weighted Cross Entropy")
parser.add_argument('--keep_incomplete', action='store_true')
parser.add_argument('--use_weighted_sampler', action='store_true')
parser.add_argument('--resume', action='store_true', help="Resume training from latest checkpoint if available")
parser.add_argument('--pretrained_checkpoint', type=str, help="Load model weights from another dataset (only model_state_dict)")
parser.add_argument('--view_type', type=str, choices=['CC','MLO'], default=None)
args = parser.parse_args()

# -------------------------------
# SEED
# -------------------------------
def set_seed(seed=5042):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(5042)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# WANDB: initialize experiment tracking
# -------------------------------
wandb.init(
    project=f"breast_density_{args.dataset}",
    name=f"{args.dataset}_{args.view_mode}_train",
    id=None,
    config=vars(args)
)

# -------------------------------
#  DATALOADER: select loader based on dataset
# -------------------------------
if args.dataset == 'dvisionlab':
    get_loader = get_dvisionlab_dataloader
elif args.dataset == 'inbreast':
    get_loader = get_inbreast_dataloader
elif args.dataset == 'cbis':
    get_loader = get_cbis_dataloader
else:
    raise ValueError("Invalid dataset")

train_loader, val_loader, train_ds, val_ds = get_loader(
    csv_path=args.csv_path,
    images_root=args.images_root,
    out_dir=args.out_dir,
    view_mode=args.view_mode,
    batch_size=args.batch_size,
    img_size=args.img_size,
    num_workers=args.num_workers,
    keep_incomplete=args.keep_incomplete,
    use_weighted_sampler=args.use_weighted_sampler,
    view_type=args.view_type,
    augment=True
)

# -------------------------------
# MODEL: initialize single or multi-view Swin Transformer
# -------------------------------
if args.view_mode == 'single':
    model = SwinTransformer_singleview(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=1,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        
    )
else:
    model = MVSwinTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=1,
        num_classes=args.num_classes,
        embed_dim=96,
        window_size=args.window_size,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
    )

model = model.to(device)
wandb.watch(model, log="all")


weights = torch.tensor([0.8, 1.5, 0.9, 1.0])

if args.weighted_ce:
    print('using weighted_ce')
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
else:   
    criterion = nn.CrossEntropyLoss()

# -------------------------------
# OPTIMIZER , SCHEDULER,  LOSS
# -------------------------------
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience, factor=0.5, threshold=0.001, min_lr=1e-7, verbose=True)
criterion = nn.CrossEntropyLoss()
early_stopper = EarlyStopper(patience=args.patiences, min_delta=args.min_delta)

# -------------------------------
# CHECKPOINT DIRS
# -------------------------------
checkpoint_dir = os.path.join("./checkpoints", f"{args.dataset}_{args.view_mode}_{args.img_size}_{args.window_size}_augm_latest")
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")

checkpoint_best_dir = os.path.join("./checkpoints", f"{args.dataset}_{args.view_mode}_{args.img_size}_{args.window_size}_augm_best")
os.makedirs(checkpoint_best_dir, exist_ok=True)
checkpoint_best_path = os.path.join(checkpoint_best_dir, "best.pth.tar")

start_epoch = 1
curr_best_val_acc = 0.0

# -------------------------------
# Helper: move optimizer state to device (useful when loading)
# -------------------------------
def optimizer_to_device(optim, device):
    for state in optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

# -------------------------------
# Confusion matrix logging 
# -------------------------------
def log_confusion_matrix(true_labels, pred_labels, title):
    """
    true_labels, pred_labels: lists or numpy arrays
    title: string used as wandb key and figure title
    """
    if len(true_labels) == 0:
        print(f"[WARN] Empty labels for {title}, skipping confusion matrix.")
        return
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", values_format='d')
    ax.set_title(title)
    plt.tight_layout()
    try:
        wandb.log({title: wandb.Image(fig)})
    except Exception as e:
        print(f"[WARN] Failed to log confusion matrix to wandb: {e}")
    plt.close(fig)

# -------------------------------
# OPTIONAL RESUME FROM CHECKPOINT
# -------------------------------
if args.resume and os.path.exists(checkpoint_best_path):
    try:
        print(f"Found checkpoint at {checkpoint_best_path}, loading...")
        ckpt = torch.load(checkpoint_best_path, map_location=device)

        # model
        model.load_state_dict(ckpt["model_state_dict"])
        # optimizer
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                optimizer_to_device(optimizer, device)
            except Exception as e:
                print(f"[WARN] Could not fully load optimizer state: {e}")
        

        # epoch and best val acc
        start_epoch = ckpt.get("epoch", 0) + 1
        curr_best_val_acc = ckpt.get("val_acc", 0.0)
        print(f"✔ Resuming from epoch {start_epoch}, best val acc = {curr_best_val_acc:.2f}%")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}. Starting from scratch.")
else:
    if args.resume:
        print("Resume flag set but no checkpoint found -> starting from scratch.")
    else:
        print("Starting from scratch (resume not requested).")

# -------------------------------
#  TRAIN LOOP
# -------------------------------
for epoch in range(start_epoch, args.epochs + 1):
    since = time.time()
    print(f"\n{'-'*20}\nEPOCH {epoch}")
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds_train, all_labels_train = [], []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit='batch'):
        if args.view_mode == 'single':
            inputs, labels, _ = batch
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            preds = model(inputs)
        else:
            inputs_cc, inputs_mlo, labels, _ = batch
            inputs_cc = inputs_cc.float().to(device, non_blocking=True)
            inputs_mlo = inputs_mlo.float().to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
            preds = model(inputs_cc, inputs_mlo)

        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds_class = preds.argmax(dim=1)
        running_loss += loss.item()
        total += labels.size(0)
        correct += (preds_class == labels).sum().item()
        all_preds_train.extend(preds_class.cpu().numpy())
        all_labels_train.extend(labels.cpu().numpy())

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    

    # ---------------------------------------------------------------------------------------------------------
    # === VALIDATION
    # --------------------------------------------------------------------------------------------------------- 

    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds_val, all_labels_val = [], []

    with torch.no_grad():
        for batch in val_loader:
            if args.view_mode == 'single':
                inputs, labels, _ = batch
                inputs, labels = inputs.float().to(device), labels.long().to(device)
                preds = model(inputs)
            else:
                inputs_cc, inputs_mlo, labels, _ = batch
                inputs_cc = inputs_cc.float().to(device)
                inputs_mlo = inputs_mlo.float().to(device)
                labels = labels.long().to(device)

                preds = model(inputs_cc, inputs_mlo)

            loss = criterion(preds, labels)

            preds_class = preds.argmax(dim=1)
            val_loss += loss.item()
            total += labels.size(0)
            correct += (preds_class == labels).sum().item()
            all_preds_val.extend(preds_class.cpu().numpy())
            all_labels_val.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = 100 * correct / total
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")



    # === LOGGING
    wandb.log({
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "epoch": epoch,
        "lr": optimizer.param_groups[0]['lr']
    })

    # === CHECKPOINTS
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        "val_acc": val_acc,
        # try to save early_stopper state if possible
        #"early_stopper_state": getattr(early_stopper, "__dict__", None)
    }
    # latest always overwritten
    torch.save(checkpoint, checkpoint_latest_path)



    # if best -> save and log confusion matrix

    if val_acc > curr_best_val_acc:
        curr_best_val_acc = val_acc
        torch.save(checkpoint, checkpoint_best_path)
        print(f"New best val acc: {val_acc:.2f}%")
        # Log confusion matrix for this best epoch
        try:
            log_confusion_matrix(all_labels_val, all_preds_val, f"Best_ConfMatrix_epoch{epoch}")
        except Exception as e:
            print(f"[WARN] Could not log best confusion matrix: {e}")



    # === CONFUSION MATRIX EVERY 10 EPOCHS
    if epoch % 10 == 0:
        try:
            log_confusion_matrix(all_labels_val, all_preds_val, f"ConfusionMatrix_epoch{epoch}")
        except Exception as e:
            print(f"[WARN] Could not log periodic confusion matrix: {e}")

    # === EARLY STOP
    if early_stopper.early_stop(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break


    # step scheduler (ReduceLROnPlateau wants val loss)
    scheduler.step(val_loss) 
