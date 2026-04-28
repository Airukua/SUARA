import json
import logging
import string
from pathlib import Path
from typing import Iterable, Optional

import torch

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.decoders import BPEDecoder
    from tokenizers.models import BPE
    from tokenizers.normalizers import NFKC, Lowercase, Sequence as NormalizerSequence, Strip
    from tokenizers.pre_tokenizers import Punctuation, Sequence as PreTokenizerSequence, Whitespace
    from tokenizers.trainers import BpeTrainer
except ImportError:
    HFTokenizer = None
    BPEDecoder = None
    BPE = None
    NormalizerSequence = None
    NFKC = None
    Lowercase = None
    Strip = None
    Punctuation = None
    PreTokenizerSequence = None
    Whitespace = None
    BpeTrainer = None


logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "tiktoken"
DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"
DEFAULT_BPE_SUFFIX = "</w>"
DEFAULT_SPECIAL_TOKENS = {
    "pad_token": "<|pad|>",
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
    "unk_token": "[UNK]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
}


def _require_tiktoken():
    if tiktoken is None:
        raise ImportError(
            "tiktoken is required for the tiktoken backend. Install it with `pip install tiktoken`."
        )


def _require_hf_tokenizers():
    if HFTokenizer is None:
        raise ImportError(
            "huggingface tokenizers is required for the BPE backend. "
            "Install it with `pip install tokenizers`."
        )


class CrystalWaveTokenizer:
    def __init__(
        self,
        encoding_name: str = DEFAULT_TIKTOKEN_ENCODING,
        model_name: Optional[str] = None,
        special_tokens: Optional[dict[str, str]] = None,
        backend: str = DEFAULT_BACKEND,
        tokenizer_object=None,
        lowercase: bool = False,
    ):
        self.backend = backend
        self.encoding_name = encoding_name
        self.model_name = model_name
        self.lowercase = lowercase
        self.special_tokens = dict(DEFAULT_SPECIAL_TOKENS)
        if special_tokens is not None:
            self.special_tokens.update(special_tokens)

        self._backend_tokenizer = None

        if self.backend == "tiktoken":
            self._init_tiktoken_backend()
        elif self.backend == "bpe":
            self._init_bpe_backend(tokenizer_object)
        else:
            raise ValueError(f"unsupported tokenizer backend: {self.backend}")

        self.pad_token = self.special_tokens["pad_token"]
        self.bos_token = self.special_tokens["bos_token"]
        self.eos_token = self.special_tokens["eos_token"]
        self.unk_token = self.special_tokens["unk_token"]
        self.cls_token = self.special_tokens["cls_token"]
        self.sep_token = self.special_tokens["sep_token"]

        self.pad_token_id = self.token_to_id(self.pad_token)
        self.bos_token_id = self.token_to_id(self.bos_token)
        self.eos_token_id = self.token_to_id(self.eos_token)
        self.unk_token_id = self.token_to_id(self.unk_token)
        self.cls_token_id = self.token_to_id(self.cls_token)
        self.sep_token_id = self.token_to_id(self.sep_token)
        self.vocab_size = self._get_vocab_size()

    def __len__(self) -> int:
        return self.vocab_size

    def _init_tiktoken_backend(self):
        _require_tiktoken()
        if not self.encoding_name and not self.model_name:
            raise ValueError("either encoding_name or model_name must be provided")

        base_encoding = self._resolve_base_encoding()
        next_token_id = base_encoding.n_vocab
        special_token_ids = {}
        for token in self.special_tokens.values():
            if token in base_encoding._special_tokens:
                special_token_ids[token] = base_encoding._special_tokens[token]
            else:
                special_token_ids[token] = next_token_id
                next_token_id += 1

        self._backend_tokenizer = tiktoken.Encoding(
            name=f"crystalwave_{base_encoding.name}",
            pat_str=base_encoding._pat_str,
            mergeable_ranks=base_encoding._mergeable_ranks,
            special_tokens={
                **base_encoding._special_tokens,
                **special_token_ids,
            },
        )

    def _init_bpe_backend(self, tokenizer_object):
        _require_hf_tokenizers()
        if tokenizer_object is None:
            tokenizer_object = HFTokenizer(
                BPE(
                    unk_token=self.special_tokens["unk_token"],
                    end_of_word_suffix=DEFAULT_BPE_SUFFIX,
                )
            )
            tokenizer_object.pre_tokenizer = self._build_pre_tokenizer()
            tokenizer_object.normalizer = self._build_normalizer(self.lowercase)
            tokenizer_object.decoder = BPEDecoder(suffix=DEFAULT_BPE_SUFFIX)
        self._backend_tokenizer = tokenizer_object
        self._validate_special_tokens()

    def _resolve_base_encoding(self):
        if self.model_name:
            return tiktoken.encoding_for_model(self.model_name)
        return tiktoken.get_encoding(self.encoding_name)

    def _build_normalizer(self, lowercase: bool):
        _require_hf_tokenizers()
        normalizers = [NFKC(), Strip()]
        if lowercase:
            normalizers.append(Lowercase())
        return NormalizerSequence(normalizers)

    def _build_pre_tokenizer(self):
        _require_hf_tokenizers()
        return PreTokenizerSequence([Whitespace(), Punctuation()])

    def _validate_special_tokens(self):
        missing = [token for token in self.special_tokens.values() if self._backend_tokenizer.token_to_id(token) is None]
        if missing:
            raise ValueError(
                "BPE tokenizer is missing required special tokens: "
                + ", ".join(missing)
            )

    def _get_vocab_size(self) -> int:
        if self.backend == "tiktoken":
            return self._backend_tokenizer.n_vocab
        return self._backend_tokenizer.get_vocab_size()

    @classmethod
    def from_model(cls, model_name: str, special_tokens: Optional[dict[str, str]] = None):
        if not model_name:
            raise ValueError("model_name must be a non-empty string")
        return cls(model_name=model_name, special_tokens=special_tokens, backend="tiktoken")

    @classmethod
    def train_bpe(
        cls,
        files: Iterable[str | Path],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: Optional[dict[str, str]] = None,
        lowercase: bool = False,
        save_directory: Optional[str | Path] = None,
        initial_alphabet: Optional[Iterable[str]] = None,
    ):
        _require_hf_tokenizers()

        file_list = [str(Path(file)) for file in files]
        if not file_list:
            raise ValueError("files cannot be empty")
        missing_files = [file for file in file_list if not Path(file).exists()]
        if missing_files:
            raise FileNotFoundError(f"training files not found: {missing_files}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if min_frequency < 1:
            raise ValueError(f"min_frequency must be at least 1, got {min_frequency}")

        merged_special_tokens = dict(DEFAULT_SPECIAL_TOKENS)
        if special_tokens is not None:
            merged_special_tokens.update(special_tokens)

        tokenizer = HFTokenizer(
            BPE(
                unk_token=merged_special_tokens["unk_token"],
                end_of_word_suffix=DEFAULT_BPE_SUFFIX,
            )
        )
        tokenizer.normalizer = cls._static_normalizer(lowercase)
        tokenizer.pre_tokenizer = cls._static_pre_tokenizer()
        tokenizer.decoder = BPEDecoder(suffix=DEFAULT_BPE_SUFFIX)
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(dict.fromkeys(merged_special_tokens.values())),
            initial_alphabet=list(initial_alphabet or cls._default_initial_alphabet()),
            end_of_word_suffix=DEFAULT_BPE_SUFFIX,
        )

        logger.info(
            "Training BPE tokenizer on %d files with vocab_size=%d",
            len(file_list),
            vocab_size,
        )
        tokenizer.train(files=file_list, trainer=trainer)

        instance = cls(
            backend="bpe",
            tokenizer_object=tokenizer,
            special_tokens=merged_special_tokens,
            lowercase=lowercase,
        )
        if save_directory is not None:
            instance.save_pretrained(save_directory)
        return instance

    @staticmethod
    def _default_initial_alphabet() -> list[str]:
        return list(dict.fromkeys(string.ascii_letters + string.digits + string.punctuation))

    @staticmethod
    def _static_normalizer(lowercase: bool):
        _require_hf_tokenizers()
        normalizers = [NFKC(), Strip()]
        if lowercase:
            normalizers.append(Lowercase())
        return NormalizerSequence(normalizers)

    @staticmethod
    def _static_pre_tokenizer():
        _require_hf_tokenizers()
        return PreTokenizerSequence([Whitespace(), Punctuation()])

    @property
    def encoding(self):
        return self._backend_tokenizer

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        allowed_special: Optional[Iterable[str]] = None,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")

        if self.backend == "tiktoken":
            special = set(allowed_special) if allowed_special is not None else set(self._backend_tokenizer.special_tokens_set)
            token_ids = self._backend_tokenizer.encode(text, allowed_special=special)
        else:
            token_ids = self._backend_tokenizer.encode(text).ids

        if add_bos:
            token_ids = [self.bos_token_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_token_id]
        return token_ids

    def encode_with_tokens(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> dict[str, list]:
        token_ids = self.encode(text, add_bos=add_bos, add_eos=add_eos)
        return {
            "tokens": self.convert_ids_to_tokens(token_ids),
            "ids": token_ids,
        }

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        tokens = self._normalize_token_ids(token_ids)
        if skip_special_tokens:
            special_ids = {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.unk_token_id,
                self.cls_token_id,
                self.sep_token_id,
            }
            tokens = [token_id for token_id in tokens if token_id not in special_ids]

        if self.backend == "tiktoken":
            return self._backend_tokenizer.decode(tokens)
        return self._backend_tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def encode_batch(
        self,
        texts: Iterable[str],
        add_bos: bool = False,
        add_eos: bool = False,
        allowed_special: Optional[Iterable[str]] = None,
    ) -> list[list[int]]:
        if isinstance(texts, str):
            raise TypeError("texts must be an iterable of strings, not a single string")
        return [
            self.encode(
                text,
                add_bos=add_bos,
                add_eos=add_eos,
                allowed_special=allowed_special,
            )
            for text in texts
        ]

    def decode_batch(self, batch_token_ids: Iterable[Iterable[int]], skip_special_tokens: bool = False) -> list[str]:
        if isinstance(batch_token_ids, torch.Tensor) and batch_token_ids.dim() != 2:
            raise ValueError(
                f"batch_token_ids tensor must be 2D, got shape {tuple(batch_token_ids.shape)}"
            )
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in batch_token_ids
        ]

    def pad(
        self,
        batch_token_ids: Iterable[Iterable[int]],
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        token_lists = [self._normalize_token_ids(token_ids) for token_ids in batch_token_ids]
        if not token_lists:
            raise ValueError("batch_token_ids cannot be empty")

        target_length = max(len(token_ids) for token_ids in token_lists) if max_length is None else max_length
        if target_length <= 0:
            raise ValueError(f"max_length must be positive, got {target_length}")

        padded = []
        attention_mask = []
        for token_ids in token_lists:
            if len(token_ids) > target_length:
                token_ids = token_ids[:target_length]
            pad_len = target_length - len(token_ids)
            padded.append(token_ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(token_ids) + [0] * pad_len)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        if return_tensors is None:
            return {"input_ids": padded, "attention_mask": attention_mask}
        raise ValueError(f"unsupported return_tensors value: {return_tensors}")

    def encode_for_causal_lm(
        self,
        text: str,
        seq_len: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        if seq_len <= 1:
            raise ValueError(f"seq_len must be greater than 1, got {seq_len}")

        token_ids = self.encode(text, add_bos=add_bos, add_eos=add_eos)
        if len(token_ids) < 2:
            token_ids = token_ids + [self.eos_token_id]

        window = seq_len + 1
        token_ids = token_ids[:window]
        if len(token_ids) < window:
            token_ids = token_ids + [self.pad_token_id] * (window - len(token_ids))

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)
        labels = labels.masked_fill(labels == self.pad_token_id, -100)
        return input_ids, labels

    def collate_causal_lm(
        self,
        texts: Iterable[str],
        seq_len: int,
        add_bos: bool = True,
        add_eos: bool = True,
        device=None,
    ):
        if isinstance(texts, str):
            raise TypeError("texts must be an iterable of strings, not a single string")
        pairs = [
            self.encode_for_causal_lm(
                text,
                seq_len=seq_len,
                add_bos=add_bos,
                add_eos=add_eos,
            )
            for text in texts
        ]
        if not pairs:
            raise ValueError("texts cannot be empty")

        input_ids = torch.stack([item[0] for item in pairs], dim=0)
        labels = torch.stack([item[1] for item in pairs], dim=0)
        if device is not None:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
        return input_ids, labels

    def token_to_id(self, token: str) -> int:
        if self.backend == "tiktoken":
            token_id = self._backend_tokenizer._special_tokens.get(token)
        else:
            token_id = self._backend_tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(f"unknown token: {token}")
        return token_id

    def id_to_token(self, token_id: int) -> str:
        token_id = int(token_id)
        if self.backend == "tiktoken":
            return self._backend_tokenizer.decode([token_id])
        token = self._backend_tokenizer.id_to_token(token_id)
        if token is None:
            raise KeyError(f"unknown token id: {token_id}")
        return token

    def convert_ids_to_tokens(self, token_ids: Iterable[int]) -> list[str]:
        return [self.id_to_token(token_id) for token_id in token_ids]

    def save_pretrained(self, save_directory: str | Path):
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        config = {
            "backend": self.backend,
            "encoding_name": self.encoding_name,
            "model_name": self.model_name,
            "special_tokens": self.special_tokens,
            "lowercase": self.lowercase,
        }
        (save_path / "tokenizer_config.json").write_text(
            json.dumps(config, indent=2),
            encoding="utf-8",
        )

        if self.backend == "bpe":
            self._backend_tokenizer.save(str(save_path / "tokenizer.json"))

    @classmethod
    def from_pretrained(cls, load_directory: str | Path):
        load_path = Path(load_directory)
        config_path = load_path / "tokenizer_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"tokenizer config not found: {config_path}")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        backend = config.get("backend", DEFAULT_BACKEND)
        if backend == "bpe":
            tokenizer_path = load_path / "tokenizer.json"
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"BPE tokenizer file not found: {tokenizer_path}")
            _require_hf_tokenizers()
            tokenizer_object = HFTokenizer.from_file(str(tokenizer_path))
            return cls(
                backend="bpe",
                tokenizer_object=tokenizer_object,
                special_tokens=config.get("special_tokens"),
                lowercase=config.get("lowercase", False),
            )

        return cls(
            encoding_name=config.get("encoding_name", DEFAULT_TIKTOKEN_ENCODING),
            model_name=config.get("model_name"),
            special_tokens=config.get("special_tokens"),
            backend="tiktoken",
        )

    def _normalize_token_ids(self, token_ids: Iterable[int]) -> list[int]:
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() != 1:
                raise ValueError(f"token_ids tensor must be 1D, got shape {tuple(token_ids.shape)}")
            return [int(token) for token in token_ids.tolist()]
        return [int(token) for token in token_ids]
