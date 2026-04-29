import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from datasets import DownloadConfig, DownloadMode, load_dataset
from datasets.utils.logging import enable_progress_bar, set_verbosity_info
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.config import load_config
from model.inference import build_generation_case, generate_sample
from model.model import CrystalWaveModel
from model.train import count_params, train, evaluate
from utils.cleaning import iter_clean_texts
from utils.dataset import TextDataset, encode_texts_to_memmap, load_memmap_metadata
from utils.tokenizer import CrystalWaveTokenizer

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

enable_progress_bar()
set_verbosity_info()


def _format_seconds(elapsed):
    return f"{elapsed:.2f}s" if elapsed < 60 else f"{elapsed / 60:.2f}m"


@contextmanager
def _stage_timer(label):
    print(f"\n[{label}] mulai...")
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"[{label}] selesai dalam {_format_seconds(time.perf_counter() - start)}")

def _iter_filtered_texts(texts, min_text_length, desc=None):
    total = len(texts) if hasattr(texts, "__len__") else None
    iterator = tqdm(
        texts,
        desc=desc,
        total=total,
        unit="text",
        dynamic_ncols=True,
    ) if desc else texts
    for text in iterator:
        if isinstance(text, str) and len(text.strip()) > min_text_length:
            yield text


def _iter_clean_filtered_texts(texts, min_text_length, chunk_size, desc=None):
    for cleaned in iter_clean_texts(
        _iter_filtered_texts(texts, min_text_length, desc=desc),
        chunk_size=chunk_size,
    ):
        if cleaned:
            yield cleaned


def _count_filtered_texts(texts, min_text_length, desc=None):
    return sum(1 for _ in _iter_filtered_texts(texts, min_text_length, desc=desc))


def _write_clean_text_file(texts, path, min_text_length, chunk_size, desc=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for cleaned in _iter_clean_filtered_texts(texts, min_text_length, chunk_size, desc=desc):
            f.write(cleaned)
            f.write("\n")
            count += 1
    return count


def _write_clean_train_val_split(texts, train_path, val_path, split_idx, min_text_length, chunk_size, desc=None):
    train_path = Path(train_path)
    val_path = Path(val_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    train_count = 0
    val_count = 0
    seen = 0
    with train_path.open("w", encoding="utf-8") as train_f, val_path.open("w", encoding="utf-8") as val_f:
        for cleaned in _iter_clean_filtered_texts(texts, min_text_length, chunk_size, desc=desc):
            if seen < split_idx:
                train_f.write(cleaned)
                train_f.write("\n")
                train_count += 1
            else:
                val_f.write(cleaned)
                val_f.write("\n")
                val_count += 1
            seen += 1
    return train_count, val_count


def _read_text_samples(path, limit):
    samples = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n")
            if text:
                samples.append(text)
            if len(samples) >= limit:
                break
    return samples


def _prepare_text_splits(dataset_dict, min_text_length, validation_split_ratio, cache_dir, cleaning_chunk_size):
    text_dir = Path(cache_dir) / "clean_text"
    text_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = text_dir / "metadata.json"
    train_path = text_dir / "train.txt"
    val_path = text_dir / "val.txt"
    test_path = text_dir / "test.txt"
    test_split = "test" if "test" in dataset_dict else "validation" if "validation" in dataset_dict else "train"
    has_validation = "validation" in dataset_dict

    if metadata_path.exists() and train_path.exists() and val_path.exists() and test_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cache_matches = (
            metadata.get("min_text_length") == min_text_length
            and metadata.get("validation_split_ratio") == validation_split_ratio
            and metadata.get("cleaning_chunk_size") == cleaning_chunk_size
            and metadata.get("test_split") == test_split
            and metadata.get("has_validation") == has_validation
        )
        if cache_matches:
            print(f"  Reusing cleaned text cache from {text_dir}")
            return metadata

    metadata = {
        "min_text_length": min_text_length,
        "validation_split_ratio": validation_split_ratio,
        "cleaning_chunk_size": cleaning_chunk_size,
        "has_validation": has_validation,
        "test_split": test_split,
        "paths": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }

    if has_validation:
        train_count = _write_clean_text_file(
            dataset_dict["train"]["text"],
            train_path,
            min_text_length,
            cleaning_chunk_size,
            desc="Clean train",
        )
        val_count = _write_clean_text_file(
            dataset_dict["validation"]["text"],
            val_path,
            min_text_length,
            cleaning_chunk_size,
            desc="Clean val",
        )
    else:
        valid_train_count = _count_filtered_texts(
            dataset_dict["train"]["text"],
            min_text_length,
            desc="Count train",
        )
        if valid_train_count < 2:
            raise ValueError(
                "train split harus memiliki minimal 2 text setelah filtering "
                "agar train/validation split bisa dibuat"
            )
        split_idx = max(1, int(valid_train_count * (1.0 - validation_split_ratio)))
        split_idx = min(split_idx, valid_train_count - 1)
        train_count, val_count = _write_clean_train_val_split(
            dataset_dict["train"]["text"],
            train_path,
            val_path,
            split_idx,
            min_text_length,
            cleaning_chunk_size,
            desc="Clean train/val",
        )

    test_count = _write_clean_text_file(
        dataset_dict[test_split]["text"],
        test_path,
        min_text_length,
        cleaning_chunk_size,
        desc="Clean test",
    )
    metadata["counts"] = {
        "train": train_count,
        "val": val_count,
        "test": test_count,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _build_generation_cases(texts, sample_count, prompt_words, target_words):
    eligible = []
    min_words = prompt_words + 4
    for text in texts:
        words = text.split()
        if len(words) > min_words:
            eligible.append(text)

    if not eligible:
        prompt, target = build_generation_case(
            texts,
            prompt_words=prompt_words,
            target_words=target_words,
        )
        return [(prompt, target)]

    sample_count = min(sample_count, len(eligible))
    if sample_count == len(eligible):
        selected = eligible
    else:
        step = max(1, len(eligible) // sample_count)
        selected = eligible[::step][:sample_count]

    cases = []
    for text in selected:
        prompt, target = build_generation_case(
            [text],
            prompt_words=prompt_words,
            target_words=target_words,
        )
        cases.append((prompt, target))
    return cases


def _save_training_plots(history, training_config, output_directory):
    if plt is None:
        print("  Plot skip: matplotlib belum terpasang")
        return

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_steps = [
        training_config.log_every_steps * idx
        for idx in range(1, len(history["train_loss"]) + 1)
    ]
    val_steps = history["steps"]
    grad_steps = [
        training_config.log_every_steps * idx
        for idx in range(1, len(history["grad_norm"]) + 1)
    ]

    loss_plot_path = output_dir / "loss_curve.png"
    grad_plot_path = output_dir / "grad_norm_curve.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    if train_steps and history["train_loss"]:
        ax.plot(train_steps, history["train_loss"], label="Train Loss", linewidth=2)
    if val_steps and history["val_loss"]:
        ax.plot(val_steps, history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_plot_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    if grad_steps and history["grad_norm"]:
        ax.plot(grad_steps, history["grad_norm"], label="Gradient Norm", linewidth=2, color="tab:red")
    ax.set_title("Gradient Norm")
    ax.set_xlabel("Step")
    ax.set_ylabel("Grad Norm")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(grad_plot_path, dpi=180)
    plt.close(fig)

    print(f"  Plot saved: {loss_plot_path}")
    print(f"  Plot saved: {grad_plot_path}")


def _load_dataset_with_cache(dataset_name, subset, cache_directory):
    cache_dir = Path(cache_directory)
    cache_dir.mkdir(parents=True, exist_ok=True)
    load_kwargs = {
        "path": dataset_name,
        "name": subset,
        "cache_dir": str(cache_dir),
        "download_mode": DownloadMode.REUSE_DATASET_IF_EXISTS,
    }

    if any(cache_dir.iterdir()):
        try:
            print(f"  Loading dataset from local cache: {cache_dir}")
            return load_dataset(
                download_config=DownloadConfig(local_files_only=True),
                **load_kwargs,
            )
        except Exception as exc:
            print(f"  Local cache belum lengkap, fallback ke download normal ({exc})")

    print(f"  Loading dataset with cache directory: {cache_dir}")
    return load_dataset(**load_kwargs)


def _tokenizer_ready(tokenizer_dir):
    return (
        (tokenizer_dir / "tokenizer_config.json").exists()
        and (tokenizer_dir / "tokenizer.json").exists()
    )

def _prepare_split(cfg, cache_dir, tokenizer_dir, tokenizer, vocab_size, split_name, text_path, text_count):
    token_path = cache_dir / f"{split_name}.bin"
    meta_path = cache_dir / f"{split_name}.json"
    rewrite_cache = cfg.tokenizer.retrain or cfg.tokenizer.rewrite_cache
    metadata = None

    if not rewrite_cache and token_path.exists() and meta_path.exists():
        metadata = load_memmap_metadata(meta_path)
        cache_matches = (
            metadata.get("text_count") == text_count
            and metadata.get("add_bos") == cfg.tokenizer.add_bos
            and metadata.get("add_eos") == cfg.tokenizer.add_eos
            and metadata.get("vocab_size") == vocab_size
            and metadata.get("tokenizer_backend", cfg.tokenizer.backend) == cfg.tokenizer.backend
        )
        if not cache_matches:
            metadata = None

    if metadata is None:
        print(f"  Encoding {split_name} split to memmap ...")
        _, metadata = encode_texts_to_memmap(
            tokenizer,
            text_path,
            token_path,
            meta_path,
            vocab_size=vocab_size,
            add_bos=cfg.tokenizer.add_bos,
            add_eos=cfg.tokenizer.add_eos,
            desc=f"Encode {split_name}",
            text_count=text_count,
            chunk_size=cfg.dataset.tokenize_chunk_size,
            extra_metadata={
                "tokenizer_backend": cfg.tokenizer.backend,
                "tokenizer_directory": str(tokenizer_dir),
                "text_path": str(text_path),
            },
        )
    else:
        print(f"  Reusing cached {split_name} memmap ...")

    dataset = TextDataset(
        token_path,
        seq_len=cfg.dataset.seq_len,
        dtype=metadata["dtype"],
    )
    return dataset, metadata["total_tokens"]

def _parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate SUARA / CrystalWave models.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path ke file YAML config eksternal. Jika tidak diisi, pakai data/config.yaml bawaan.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training dari checkpoint last.pt jika tersedia.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Path checkpoint spesifik untuk resume training.",
    )
    return parser.parse_args()


def _resolve_resume_checkpoint(args, checkpoint_config, device):
    if checkpoint_config is None or not checkpoint_config.enabled:
        return None

    requested_path = args.resume_path or checkpoint_config.resume_path
    should_resume = args.resume or checkpoint_config.resume_if_available or requested_path is not None
    if not should_resume:
        return None

    checkpoint_path = Path(requested_path) if requested_path is not None else Path(checkpoint_config.output_directory) / "last.pt"
    if not checkpoint_path.exists():
        print(f"  Resume requested, tapi checkpoint tidak ditemukan: {checkpoint_path}")
        return None

    print(f"  Resuming from checkpoint: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)

def main():
    args = _parse_args()
    cfg = load_config(args.config)
    if args.config is not None:
        print(f"  Using config: {Path(args.config).resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_checkpoint = _resolve_resume_checkpoint(args, cfg.checkpoint, device)
    with _stage_timer("Dataset"):
        ds = _load_dataset_with_cache(
            cfg.dataset.name,
            cfg.dataset.subset,
            cfg.dataset.cache_directory,
        )

    with _stage_timer("Split Resolve"):
        text_split_meta = _prepare_text_splits(
            ds,
            cfg.dataset.min_text_length,
            cfg.dataset.validation_split_ratio,
            cfg.tokenizer.cache_directory,
            cfg.dataset.cleaning_chunk_size,
        )
    train_text_path = Path(text_split_meta["paths"]["train"])
    val_text_path = Path(text_split_meta["paths"]["val"])
    test_text_path = Path(text_split_meta["paths"]["test"])
    train_text_count = text_split_meta["counts"]["train"]
    val_text_count = text_split_meta["counts"]["val"]
    test_text_count = text_split_meta["counts"]["test"]
    print(f"  Sentences: train={train_text_count:,}  val={val_text_count:,}  test={test_text_count:,}")

    tokenizer_dir = Path(cfg.tokenizer.save_directory)
    if cfg.tokenizer.backend != "bpe":
        raise ValueError(f"unsupported tokenizer backend in config: {cfg.tokenizer.backend}")

    with _stage_timer("Tokenizer"):
        if cfg.tokenizer.retrain:
            print("  retrain=true, tokenizer BPE akan dilatih ulang")
            tokenizer = CrystalWaveTokenizer.train_bpe(
                files=[train_text_path],
                vocab_size=cfg.tokenizer.vocab_size,
                min_frequency=cfg.tokenizer.min_frequency,
                lowercase=cfg.tokenizer.lowercase,
                save_directory=tokenizer_dir,
            )
        elif _tokenizer_ready(tokenizer_dir):
            print(f"  Reusing existing BPE tokenizer from {tokenizer_dir}")
            tokenizer = CrystalWaveTokenizer.from_pretrained(tokenizer_dir)
        else:
            print("  Existing BPE tokenizer belum ada, training sekali untuk membuat cache tokenizer ...")
            tokenizer = CrystalWaveTokenizer.train_bpe(
                files=[train_text_path],
                vocab_size=cfg.tokenizer.vocab_size,
                min_frequency=cfg.tokenizer.min_frequency,
                lowercase=cfg.tokenizer.lowercase,
                save_directory=tokenizer_dir,
            )

    vocab_size = len(tokenizer)
    print(f"  Tokenizer: backend={cfg.tokenizer.backend}  vocab={vocab_size:,}  path={tokenizer_dir}")

    cache_dir = Path(cfg.tokenizer.cache_directory)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with _stage_timer("Token Cache"):
        print("  Preparing token cache ...")
        train_ds, train_tokens = _prepare_split(
            cfg, cache_dir, tokenizer_dir, tokenizer, vocab_size, "train", train_text_path, train_text_count
        )
        val_ds, val_tokens = _prepare_split(
            cfg, cache_dir, tokenizer_dir, tokenizer, vocab_size, "val", val_text_path, val_text_count
        )
        test_ds, test_tokens = _prepare_split(
            cfg, cache_dir, tokenizer_dir, tokenizer, vocab_size, "test", test_text_path, test_text_count
        )
    print(f"  Tokens  : train={train_tokens:,}  val={val_tokens:,}  test={test_tokens:,}")

    pin = device.type == "cuda"
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.train_shuffle,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=pin,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.eval_shuffle,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=pin,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.eval_shuffle,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=pin,
    )
    print(f"  Batches : train={len(train_dl):,}  val={len(val_dl):,}  test={len(test_dl):,}")

    model = CrystalWaveModel(
        vocab_size=vocab_size,
        dropout=cfg.training.dropout,
        **cfg.model.model_kwargs,
    ).to(device)

    pc, _ = count_params(model)
    print(f"  {cfg.model.architecture_label:<25} {pc:>12,}   {cfg.model.architecture_detail:>18}")

    val_preview_texts = _read_text_samples(val_text_path, limit=2_048)
    sample_prompt, sample_target = build_generation_case(
        val_preview_texts,
        prompt_words=cfg.generation.prompt_words,
        target_words=cfg.generation.target_words,
    )

    history = train(
        model,
        train_dl,
        val_dl,
        device,
        cfg.model.architecture_label,
        cfg.training,
        generation_config=cfg.generation,
        tokenizer=tokenizer,
        sample_prompt=sample_prompt,
        wandb_config=cfg.wandb,
        model_config=cfg.model.model_kwargs,
        checkpoint_config=cfg.checkpoint,
        resume_checkpoint=resume_checkpoint,
    )

    print(f"\n  Evaluating test set ...")
    avg_tc = (
        sum(history["elapsed_times"]) / len(history["elapsed_times"])
        if history["elapsed_times"]
        else 0.0
    )
    print(f"  Running test evaluation on {len(test_dl):,} batches")
    _, test_ppl_c = evaluate(
        model,
        test_dl,
        device,
        show_progress=True,
        desc="TestEval",
    )
    last_train_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
    print(f"  {cfg.model.architecture_label:<25} {test_ppl_c:>9.2f} {last_train_loss:>13.4f} {avg_tc:>10.2f}s {pc:>10,}")

    if cfg.plots.enabled:
        _save_training_plots(
            history,
            cfg.training,
            cfg.plots.output_directory,
        )

    sample_cases = _build_generation_cases(
        _read_text_samples(test_text_path, limit=4_096),
        sample_count=5,
        prompt_words=cfg.generation.prompt_words,
        target_words=cfg.generation.target_words,
    )

    print(f"\n  5 Sample Hasil Generation")
    for idx, (test_prompt, test_target) in enumerate(sample_cases, start=1):
        sample_crystal = generate_sample(
            model,
            tokenizer,
            device,
            test_prompt,
            MAX_SEQ=cfg.model.max_seq,
            max_new_tokens=cfg.generation.max_new_tokens,
            temperature=cfg.generation.temperature,
            top_k=cfg.generation.top_k,
        )

        print(f"  {'-'*72}")
        print(f"  Sample #{idx}")
        print(f"  Prompt    : {test_prompt}")
        if test_target:
            print(f"  Referensi : {test_target}")
        print(f"  {'-'*72}")
        print(f"  [CrystalWave]\n  {sample_crystal}\n")

if __name__ == "__main__":
    main()
