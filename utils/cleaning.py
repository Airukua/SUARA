import re
import unicodedata


DEFAULT_TEXT_CHUNK_SIZE = 100_000
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")

_CHAR_REPLACEMENTS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
        "\u200b": "",
        "\ufeff": "",
    }
)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.translate(_CHAR_REPLACEMENTS)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL_CHARS_RE.sub("", cleaned)
    cleaned = cleaned.replace("\n", " ")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def clean_texts(texts):
    return [clean_text(text) for text in texts]


def iter_clean_texts(texts, chunk_size=DEFAULT_TEXT_CHUNK_SIZE):
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    batch = []
    for text in texts:
        batch.append(text)
        if len(batch) >= chunk_size:
            yield from clean_texts(batch)
            batch.clear()

    if batch:
        yield from clean_texts(batch)
