import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

from data.config import load_config
from model.model import CrystalWaveModel
from utils.tokenizer import CrystalWaveTokenizer

def build_generation_case(texts, prompt_words=12, target_words=24):
    for text in texts:
        words = text.split()
        if len(words) > prompt_words + 4:
            prompt = ' '.join(words[:prompt_words])
            target = ' '.join(words[prompt_words:prompt_words + target_words])
            return prompt, target
    fallback_prompt = "the history of"
    return fallback_prompt, ""

@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt, MAX_SEQ=None, max_new_tokens=40,
                    temperature=0.9, top_k=40, ):
    model = model.to(device)
    model.eval()
    if MAX_SEQ is None:
        MAX_SEQ = getattr(model, 'max_seq', 128)
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.bos_token_id]

    generated = list(token_ids)

    for _ in range(max_new_tokens):
        ctx = generated[-MAX_SEQ:]
        inp = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = model(inp)
        next_logits = logits[0, -1] / max(temperature, 1e-5)

        if top_k is not None and top_k > 0:
            k = min(top_k, next_logits.size(-1))
            top_vals, top_idx = torch.topk(next_logits, k)
            probs = F.softmax(top_vals, dim=-1)
            next_token = top_idx[torch.multinomial(probs, 1)].item()
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    device,
    prompt: str,
    MAX_SEQ: Optional[int] = None,
    max_new_tokens: int = 40,
    temperature: float = 0.9,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9, 
    repetition_penalty: float = 1.1,  
    stop_strings: Optional[List[str]] = None, 
) -> str:
    model.eval()
    if MAX_SEQ is None:
        MAX_SEQ = getattr(model, 'max_seq', 128)

    token_ids: List[int] = tokenizer.encode(prompt)
    if not token_ids:
        bos = getattr(tokenizer, 'bos_token_id', None)
        token_ids = [bos] if bos is not None else [0]

    generated = list(token_ids)
    prompt_len = len(generated)

    for _ in range(max_new_tokens):
        ctx = generated[-MAX_SEQ:]
        inp = torch.tensor([ctx], dtype=torch.long, device=device)

        logits, _ = model(inp)
        next_logits = logits[0, -1].clone().float()
        next_logits /= max(temperature, 1e-5)

        if repetition_penalty != 1.0:
            for tok_id in set(ctx):
                if next_logits[tok_id] > 0:
                    next_logits[tok_id] /= repetition_penalty
                else:
                    next_logits[tok_id] *= repetition_penalty

        if top_k is not None and top_k > 0:
            k = min(top_k, next_logits.size(-1))
            kth_val = torch.topk(next_logits, k).values[-1]
            next_logits = next_logits.masked_fill(next_logits < kth_val, float('-inf'))

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_logits[cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p] = float('-inf')
            next_logits = torch.zeros_like(next_logits).scatter_(0, sorted_idx, sorted_logits)

        probs = F.softmax(next_logits, dim=-1)
        next_token: int = torch.multinomial(probs, 1).item()
        generated.append(next_token)
        eos_id = getattr(tokenizer, 'eos_token_id', None)
        if eos_id is not None and next_token == eos_id:
            break

        if stop_strings:
            partial = tokenizer.decode(generated[prompt_len:], skip_special_tokens=True)
            if any(s in partial for s in stop_strings):
                break
    return tokenizer.decode(generated[prompt_len:], skip_special_tokens=True)


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained SUARA checkpoint.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Prompt teks. Jika kosong, akan dibaca dari stdin bila tersedia.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path ke file config YAML. Default pakai data/config.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path checkpoint model. Default: best.pt lalu fallback ke last.pt di artifacts/checkpoints.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument(
        "--stop-string",
        action="append",
        default=None,
        help="String penghenti. Bisa dipakai berkali-kali.",
    )
    return parser.parse_args()


def _resolve_prompt(cli_prompt: Optional[str]) -> str:
    if cli_prompt is not None and cli_prompt.strip():
        return cli_prompt
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    raise ValueError("prompt wajib diisi lewat argumen atau stdin")


def _resolve_checkpoint_path(checkpoint_arg: Optional[str], checkpoint_dir: str) -> Path:
    if checkpoint_arg is not None:
        checkpoint_path = Path(checkpoint_arg)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint tidak ditemukan: {checkpoint_path}")
        return checkpoint_path

    base_dir = Path(checkpoint_dir)
    for candidate in (base_dir / "best.pt", base_dir / "last.pt"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"checkpoint default tidak ditemukan di {base_dir}. "
        "Gunakan --checkpoint untuk menentukan file checkpoint."
    )


def main():
    args = _parse_args()
    prompt = _resolve_prompt(args.prompt)
    cfg = load_config(args.config)
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, cfg.checkpoint.output_directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CrystalWaveTokenizer.from_pretrained(cfg.tokenizer.save_directory)
    model = CrystalWaveModel(
        vocab_size=len(tokenizer),
        dropout=cfg.training.dropout,
        **cfg.model.model_kwargs,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    output = generate(
        model,
        tokenizer,
        device,
        prompt=prompt,
        MAX_SEQ=cfg.model.max_seq,
        max_new_tokens=args.max_new_tokens or cfg.generation.max_new_tokens,
        temperature=args.temperature if args.temperature is not None else cfg.generation.temperature,
        top_k=args.top_k if args.top_k is not None else cfg.generation.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_strings=args.stop_string,
    )
    print(output)


if __name__ == "__main__":
    main()
