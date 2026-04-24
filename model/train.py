import torch
import time
import math
from tqdm import tqdm

def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def evaluate(model, dataloader, device):
    model = model.to(device)
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for inp, lbl in dataloader:
        inp, lbl = inp.to(device), lbl.to(device)
        _, loss  = model(inp, labels=lbl)
        total_loss   += loss.item() * lbl.numel()
        total_tokens += lbl.numel()
    avg_loss   = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity


def cosine_schedule(optimizer, epoch, total_epochs, lr_max, lr_min=1e-5, warmup=3):
    if epoch < warmup:
        lr = lr_max * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def train_one_epoch(model, dl, optimizer, scaler, device, grad_clip=1.0):
    model.train()
    total_loss, steps = 0.0, 0
    for inp, lbl in tqdm(dl, desc='Training'):
        inp, lbl = inp.to(device), lbl.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == 'cuda'):
            _, loss = model(inp, labels=lbl)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        steps      += 1
    return total_loss / steps


def train(model, train_dl, val_dl, device, label, EPOCHS, LR):
    model = model.to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    train_hist, ppl_hist, time_hist = [], [], []

    print(f"\n  ── Training [{label}] ──")
    print(f"  {'Ep':>3} | {'TrainLoss':>10} | {'ValPPL':>8} | {'Time':>6}")
    print(f"  {'-'*38}")

    for epoch in range(1, EPOCHS + 1):
        cosine_schedule(opt, epoch - 1, EPOCHS, LR)
        t0 = time.time()
        tl = train_one_epoch(model, train_dl, opt, scaler, device)
        _, ppl = evaluate(model, val_dl, device)
        elapsed = time.time() - t0

        train_hist.append(tl)
        ppl_hist.append(ppl)
        time_hist.append(elapsed)
        print(f"  {epoch:>3} | {tl:>10.4f} | {ppl:>8.2f} | {elapsed:>5.1f}s")

    return train_hist, ppl_hist, time_hist
