import torch
from model.with_attention import WithAttention
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
from model.train import count_params, train, evaluate
from model.inference import build_generation_case, generate_sample

SEQ_LEN    = 128
BATCH      = 16
EPOCHS     = 10
LR         = 3e-4
DROPOUT    = 0.1
MAX_SEQ    = SEQ_LEN
MAX_VOCAB  = 10000
FF_MULT    = 8 / 3

DIM          = 512
N_LAYERS     = 6
N_ATTN_HEADS = 4
N_WAVE_HEADS = 4
N_SCALES     = 4  
SIGMA_SCALES = [1.0, 4.0, 16.0, 64.0]


class SimpleWordTokenizer:
    def __init__(self, texts, max_vocab=10000):
        counter = Counter()
        for t in texts:
            counter.update(t.split())
        self.vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        for word, _ in counter.most_common(max_vocab - 4):
            self.vocab[word] = len(self.vocab)
        self.inv = {v: k for k, v in self.vocab.items()}
        print(f"  Vocab size: {len(self.vocab):,}")

    def encode(self, text):
        return [self.vocab.get(w, 1) for w in text.split()]

    def decode(self, ids):
        return ' '.join(self.inv.get(i, '<unk>') for i in ids)

    def __len__(self):
        return len(self.vocab)


class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len=128):
        flat = []
        for ids in token_ids:
            flat.extend(ids)
        self.chunks = []
        for i in range(0, len(flat) - seq_len, seq_len):
            chunk = flat[i: i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                inp = torch.tensor(chunk[:-1], dtype=torch.long)
                lbl = torch.tensor(chunk[1:],  dtype=torch.long)
                self.chunks.append((inp, lbl))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds = load_dataset("wikitext", "wikitext-2-raw-v1")

train_texts = [t for t in ds['train']['text']      if len(t.strip()) > 20]
val_texts   = [t for t in ds['validation']['text'] if len(t.strip()) > 20]
test_texts  = [t for t in ds['test']['text']       if len(t.strip()) > 20]
print(f"  Sentences: train={len(train_texts):,}  val={len(val_texts):,}  test={len(test_texts):,}")


tokenizer  = SimpleWordTokenizer(train_texts, MAX_VOCAB)
vocab_size = len(tokenizer)

train_ids = [tokenizer.encode(t) for t in train_texts]
val_ids   = [tokenizer.encode(t) for t in val_texts]
test_ids  = [tokenizer.encode(t) for t in test_texts]

train_ds = TextDataset(train_ids, SEQ_LEN)
val_ds   = TextDataset(val_ids,   SEQ_LEN)
test_ds  = TextDataset(test_ids,  SEQ_LEN)

pin      = device.type == 'cuda'
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=pin)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=pin)
test_dl  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=pin)
print(f"  Batches : train={len(train_dl):,}  val={len(val_dl):,}  test={len(test_dl):,}")


model = WithAttention(
    vocab_size=vocab_size, dim=DIM, n_layers=N_LAYERS,
    n_attn_heads=N_ATTN_HEADS, n_wave_heads=N_WAVE_HEADS,
    n_scales=N_SCALES, sigma_scales=SIGMA_SCALES,
    ff_mult=FF_MULT, dropout=DROPOUT, max_seq=MAX_SEQ
).to(device)

pc, _ = count_params(model)
print(f"  {'CrystalWave + Attention':<25} {pc:>12,}   {'attn=4, wave=4':>18}")

tc, ppl_c, tim_c = train(model, train_dl, val_dl, device, "CrystalWave + Attention (4 heads)", EPOCHS, LR)

print(f"\n  Evaluating test set ...")
avg_tc = sum(tim_c) / EPOCHS
_, test_ppl_c = evaluate(model, test_dl, device)
print(f"  {'CrystalWave + Attention':<25} {test_ppl_c:>9.2f} {tc[-1]:>13.4f} {avg_tc:>10.2f}s {pc:>10,}")

sample_prompt, sample_target = build_generation_case(test_texts)
sample_crystal = generate_sample(model, tokenizer, device, sample_prompt, MAX_SEQ=MAX_SEQ)

print(f"\n  Pair Hasil Generation")
print(f"  {'-'*72}")
print(f"  Prompt    : {sample_prompt}")
if sample_target:
    print(f"  Referensi : {sample_target}")
print(f"  {'-'*72}")
print(f"  [CrystalWave]\n  {sample_crystal}\n")

# ============================================================
# INSPEKSI: Cari penyebab PPL terlalu rendah (1.27)
# ============================================================
import math

print("\n" + "="*60)
print("  INSPEKSI DIAGNOSTIK")
print("="*60)

# ----------------------------------------------------------
# 1. Cek overlap train vs val vs test
# ----------------------------------------------------------
print("\n[1] Cek Data Leakage (overlap antar split)")
train_set = set(train_texts)
val_set   = set(val_texts)
test_set  = set(test_texts)

overlap_tv = train_set & val_set
overlap_tt = train_set & test_set
overlap_vt = val_set   & test_set

print(f"    Train ∩ Val  : {len(overlap_tv):,} kalimat {'<-- LEAKAGE!' if overlap_tv else '(ok)'}")
print(f"    Train ∩ Test : {len(overlap_tt):,} kalimat {'<-- LEAKAGE!' if overlap_tt else '(ok)'}")
print(f"    Val ∩ Test   : {len(overlap_vt):,} kalimat {'<-- LEAKAGE!' if overlap_vt else '(ok)'}")

# ----------------------------------------------------------
# 2. Cek input vs label alignment di dataset
# ----------------------------------------------------------
print("\n[2] Cek Input vs Label Alignment (harusnya geser 1 token)")
for split_name, split_ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
    inp, lbl = split_ds[0]
    is_shifted = torch.equal(inp[1:], lbl[:-1])
    match_exact = torch.equal(inp, lbl)
    print(f"    [{split_name}]")
    print(f"      input[:5]  : {inp[:5].tolist()}")
    print(f"      label[:5]  : {lbl[:5].tolist()}")
    print(f"      shifted +1 : {'YES (benar)' if is_shifted else 'NO <-- BUG! input == label atau misaligned'}")
    print(f"      inp==lbl   : {'SAMA PERSIS <-- BUG BESAR!' if match_exact else 'berbeda (ok)'}")

# ----------------------------------------------------------
# 3. Cek apakah chunk antar split saling overlap
# ----------------------------------------------------------
print("\n[3] Cek Chunk Overlap antar TextDataset")
def chunk_set(ds):
    return set(tuple(inp.tolist()) for inp, _ in ds)

train_chunks = chunk_set(train_ds)
val_chunks   = chunk_set(val_ds)
test_chunks  = chunk_set(test_ds)

ov_tv = train_chunks & val_chunks
ov_tt = train_chunks & test_chunks
print(f"    Train ∩ Val  chunks: {len(ov_tv):,} {'<-- LEAKAGE!' if ov_tv else '(ok)'}")
print(f"    Train ∩ Test chunks: {len(ov_tt):,} {'<-- LEAKAGE!' if ov_tt else '(ok)'}")

# ----------------------------------------------------------
# 4. Re-evaluate loss manual untuk sanity check PPL
# ----------------------------------------------------------
print("\n[4] Sanity Check PPL Manual")
import torch.nn.functional as F

model.eval()
total_loss = 0
total_tokens = 0
with torch.no_grad():
    for batch_idx, (x, y) in enumerate(val_dl):
        if batch_idx >= 20:  # sample 20 batch saja
            break
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        # logits: (B, T, V) -> flatten
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=0
        )
        total_loss += loss.item()
        total_tokens += (y != 0).sum().item()

manual_avg_loss = total_loss / 20
manual_ppl = math.exp(manual_avg_loss)
print(f"    Manual avg loss (20 batch val) : {manual_avg_loss:.4f}")
print(f"    Manual PPL                     : {manual_ppl:.2f}")
print(f"    PPL dari training log          : 1.28")
print(f"    Selisih                        : {'MENCURIGAKAN!' if abs(manual_ppl - 1.28) > 1.0 else 'konsisten'}")

# ----------------------------------------------------------
# 5. Cek apakah model bisa predict token berikutnya dengan benar
# ----------------------------------------------------------
print("\n[5] Cek Prediksi Token — Apakah Model Terlalu 'Hafal'?")
model.eval()
sample_inp, sample_lbl = val_ds[0]
x_in = sample_inp.unsqueeze(0).to(device)
with torch.no_grad():
    logits, _ = model(x_in)  # (1, T, V)

probs = torch.softmax(logits[0], dim=-1)
top_probs, top_ids = probs.topk(1, dim=-1)  # (T, 1)

correct = (top_ids.squeeze(-1).cpu() == sample_lbl).float().mean().item()
avg_top1_prob = top_probs.mean().item()

print(f"    Top-1 accuracy pada 1 sample val : {correct*100:.1f}%")
print(f"    Rata-rata probabilitas top-1     : {avg_top1_prob:.4f}")
if avg_top1_prob > 0.8:
    print("    --> Model terlalu confident! Kemungkinan hafal data / target bocor.")
elif avg_top1_prob > 0.5:
    print("    --> Agak tinggi, patut dicurigai.")
else:
    print("    --> Probabilitas wajar.")

print("\n" + "="*60)
print("  SELESAI INSPEKSI")
print("="*60 + "\n")

# ============================================================
# INSPEKSI ARSITEKTUR: Cek apakah ada future leakage
# ============================================================
print("\n[6] Cek Kausal Masking di Arsitektur")

model.eval()
with torch.no_grad():
    # Buat input dummy: token semua sama
    dummy = torch.zeros(1, SEQ_LEN, dtype=torch.long).to(device)
    dummy[0, 0] = 100  # hanya posisi 0 yang berbeda

    out_A, _ = model(dummy)
    out_A = out_A.cpu()  # (1, T, V)

    dummy2 = dummy.clone()
    dummy2[0, 5] = 999  # ubah posisi 5

    out_B, _ = model(dummy2)
    out_B = out_B.cpu()

# Jika causal: output posisi 0-4 TIDAK BOLEH berubah
diff = (out_A[0] - out_B[0]).abs()  # (T, V)
diff_per_pos = diff.max(dim=-1).values  # (T,)

print("    Pengaruh perubahan token posisi 5 terhadap output tiap posisi:")
print("    (harusnya posisi 0-4 = 0.0000, posisi 5+ boleh berubah)\n")
for i in range(min(12, SEQ_LEN)):
    bar = "█" * min(40, int(diff_per_pos[i].item() * 100))
    flag = " <-- BOCOR!" if i < 5 and diff_per_pos[i].item() > 1e-4 else ""
    print(f"    pos {i:>3} : {diff_per_pos[i].item():.6f}  {bar}{flag}")

print()
any_leak = any(diff_per_pos[i].item() > 1e-4 for i in range(5))
if any_leak:
    print("    KESIMPULAN: Causal masking BOCOR — model bisa lihat masa depan!")
    print("    Cek implementasi FFT/Conv di no_attention.py")
    print("    --> FFT global tidak causal by default")
    print("    --> Conv perlu padding kiri, bukan padding simetris")
else:
    print("    KESIMPULAN: Causal masking aman.")
