import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, token_source, seq_len=128, dtype=None):
        self.seq_len = seq_len
        self._token_source = token_source
        self._dtype = np.dtype(dtype) if dtype is not None else None
        self._tokens = None
        if not isinstance(token_source, (str, Path)):
            self._tokens = np.asarray(token_source)

    def __len__(self):
        total_tokens = len(self._load_tokens())
        return max(0, (total_tokens - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        stop = start + self.seq_len + 1
        window = self._load_tokens()[start:stop]
        inp = torch.as_tensor(np.asarray(window[:-1], dtype=np.int64))
        lbl = torch.as_tensor(np.asarray(window[1:], dtype=np.int64))
        return inp, lbl

    def _load_tokens(self):
        if self._tokens is None:
            if self._dtype is None:
                raise ValueError("dtype is required when loading token data from a memmap file")
            self._tokens = np.memmap(self._token_source, mode="r", dtype=self._dtype)
        return self._tokens


def count_total_tokens(token_ids):
    if isinstance(token_ids, (str, Path)):
        raise TypeError("count_total_tokens expects in-memory tokens, not a memmap path")
    if isinstance(token_ids, np.ndarray):
        return int(token_ids.shape[0])
    return sum(len(ids) for ids in token_ids)


def write_corpus_file(texts, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for text in texts:
            cleaned = text.strip()
            if cleaned:
                f.write(cleaned)
                f.write("\n")
    return path


def encode_texts(tokenizer, texts, add_bos=False, add_eos=False, desc="Encoding"):
    token_ids = []
    for text in tqdm(texts, desc=desc, unit="text", dynamic_ncols=True):
        token_ids.append(
            tokenizer.encode(
                text,
                add_bos=add_bos,
                add_eos=add_eos,
            )
        )
    return token_ids


def token_dtype_for_vocab(vocab_size):
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.uint16
    if vocab_size <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


def load_memmap_metadata(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encode_texts_to_memmap(
    tokenizer,
    texts,
    output_path,
    metadata_path,
    vocab_size,
    add_bos=False,
    add_eos=False,
    desc="Encoding",
    extra_metadata=None,
):
    output_path = Path(output_path)
    metadata_path = Path(metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    token_dtype = token_dtype_for_vocab(vocab_size)
    total_tokens = 0
    for text in tqdm(texts, desc=f"{desc} size", unit="text", dynamic_ncols=True):
        total_tokens += len(
            tokenizer.encode(
                text,
                add_bos=add_bos,
                add_eos=add_eos,
            )
        )

    mmap = np.memmap(output_path, mode="w+", dtype=token_dtype, shape=(total_tokens,))
    cursor = 0
    for text in tqdm(texts, desc=desc, unit="text", dynamic_ncols=True):
        encoded = np.asarray(
            tokenizer.encode(
                text,
                add_bos=add_bos,
                add_eos=add_eos,
            ),
            dtype=token_dtype,
        )
        next_cursor = cursor + len(encoded)
        mmap[cursor:next_cursor] = encoded
        cursor = next_cursor

    mmap.flush()
    metadata = {
        "dtype": np.dtype(token_dtype).name,
        "total_tokens": total_tokens,
        "text_count": len(texts),
        "add_bos": add_bos,
        "add_eos": add_eos,
        "vocab_size": vocab_size,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path, metadata
