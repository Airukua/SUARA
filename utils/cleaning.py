import re
import unicodedata


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
