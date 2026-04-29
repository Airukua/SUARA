from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


AttentionMode = Literal["self", "nastar", "gla", "disabled"]
FFNMode = Literal["dense", "moe"]
PrecisionMode = Literal["fp32", "fp16", "bf16"]
OptimizerMode = Literal["adamw", "muon"]

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "wikitext"
    subset: str = "wikitext-2-raw-v1"
    cache_directory: str = "artifacts/dataset_cache"
    min_text_length: int = 20
    cleaning_chunk_size: int = 100_000
    tokenize_chunk_size: int = 100_000
    max_vocab: int = 10_000
    seq_len: int = 128
    validation_split_ratio: float = 0.05


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 16
    num_workers: int = 0
    train_shuffle: bool = True
    eval_shuffle: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int = 5_000
    learning_rate: float = 3e-4
    optimizer: OptimizerMode = "muon"
    dropout: float = 0.1
    precision: PrecisionMode = "bf16"
    warmup_steps: int = 200
    decay_start_step: int | None = None
    min_learning_rate: float = 1e-5
    grad_clip: float = 1.0
    eval_every_steps: int = 500
    eval_max_batches: int | None = 100
    sample_every_steps: int = 500
    log_every_steps: int = 50
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    restore_best_model: bool = True
    weight_decay: float = 0.1
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95


@dataclass(frozen=True)
class CheckpointConfig:
    enabled: bool = True
    output_directory: str = "artifacts/checkpoints"
    save_best: bool = True
    save_last: bool = True
    resume_if_available: bool = False
    resume_path: str | None = None


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = "crystalwave"
    name: str | None = None
    entity: str | None = None
    mode: str = "online"
    log_model: bool = False


@dataclass(frozen=True)
class GenerationConfig:
    prompt_words: int = 12
    target_words: int = 24
    max_new_tokens: int = 40
    temperature: float = 0.9
    top_k: int = 40


@dataclass(frozen=True)
class PlotConfig:
    output_directory: str = "artifacts/plots"
    enabled: bool = True


@dataclass(frozen=True)
class TokenizerConfig:
    backend: Literal["bpe"] = "bpe"
    vocab_size: int = 10_000
    min_frequency: int = 2
    lowercase: bool = False
    save_directory: str = "artifacts/tokenizer"
    cache_directory: str = "artifacts/token_cache"
    rewrite_cache: bool = False
    retrain: bool = False
    add_bos: bool = False
    add_eos: bool = True


@dataclass(frozen=True)
class ModelConfig:
    dim: int = 512
    n_layers: int = 6
    max_seq: int = 128
    n_attn_heads: int = 1
    n_wave_heads: int = 4
    n_scales: int = 4
    sigma_scales: list[float] = field(default_factory=lambda: [1.0, 4.0, 16.0, 64.0])
    ff_mult: float = 8 / 3
    attention_mode: AttentionMode = "self"
    ffn_mode: FFNMode = "moe"
    gradient_checkpointing: bool = False
    num_experts: int = 8
    active_experts: int = 2
    aux_loss_coef: float = 0.01

    @property
    def use_attention(self) -> bool:
        return self.attention_mode != "disabled"

    @property
    def model_kwargs(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_attn_heads": self.n_attn_heads,
            "n_wave_heads": self.n_wave_heads,
            "n_scales": self.n_scales,
            "sigma_scales": self.sigma_scales,
            "ff_mult": self.ff_mult,
            "max_seq": self.max_seq,
            "attention_mode": self.attention_mode,
            "ffn_mode": self.ffn_mode,
            "gradient_checkpointing": self.gradient_checkpointing,
            "num_experts": self.num_experts,
            "active_experts": self.active_experts,
            "aux_loss_coef": self.aux_loss_coef,
        }

    @property
    def architecture_label(self) -> str:
        if self.attention_mode == "disabled":
            attn_label = "NoAttention"
        elif self.attention_mode == "nastar":
            attn_label = "NastarAttention"
        elif self.attention_mode == "gla":
            attn_label = "GLAAttention"
        else:
            attn_label = "SelfAttention"
        if self.ffn_mode == "moe":
            ffn_label = f"MoE({self.active_experts}/{self.num_experts})"
        else:
            ffn_label = "DenseFFN"
        return f"CrystalWave + {attn_label} + {ffn_label}"

    @property
    def architecture_detail(self) -> str:
        return (
            f"attn_mode={self.attention_mode}, "
            f"attn_heads={self.n_attn_heads if self.use_attention else 0}, "
            f"wave={self.n_wave_heads}, "
            f"ffn={self.ffn_mode}, "
            f"grad_ckpt={self.gradient_checkpointing}, "
            f"layers={self.n_layers}, "
            f"dim={self.dim}"
        )


@dataclass(frozen=True)
class ExploreConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config YAML harus berbentuk mapping/object, got {type(data).__name__}")
    return data


def _normalize_nullable_value(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {"none", "null", "~", ""}:
        return None
    return value


def _build_section(section_cls, raw: dict[str, Any] | None):
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError(f"section {section_cls.__name__} harus berbentuk mapping/object")
    raw = {key: _normalize_nullable_value(value) for key, value in raw.items()}
    return section_cls(**raw)


def load_config(path: str | Path | None = None) -> ExploreConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = _read_yaml(config_path)
    return ExploreConfig(
        dataset=_build_section(DatasetConfig, raw.get("dataset")),
        dataloader=_build_section(DataLoaderConfig, raw.get("dataloader")),
        training=_build_section(TrainingConfig, raw.get("training")),
        checkpoint=_build_section(CheckpointConfig, raw.get("checkpoint")),
        wandb=_build_section(WandbConfig, raw.get("wandb")),
        generation=_build_section(GenerationConfig, raw.get("generation")),
        plots=_build_section(PlotConfig, raw.get("plots")),
        tokenizer=_build_section(TokenizerConfig, raw.get("tokenizer")),
        model=_build_section(ModelConfig, raw.get("model")),
    )


CONFIG = load_config()

def config_to_dict(config: ExploreConfig) -> dict[str, Any]:
    return asdict(config)
