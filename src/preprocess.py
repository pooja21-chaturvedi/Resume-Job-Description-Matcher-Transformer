"""Light-weight text cleaning.

Transformer models work best on natural text, so we intentionally avoid
aggressive steps such as stop-word removal or stemming.
"""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+")
_NON_TEXT = re.compile(r"[^A-Za-z0-9+#.,;:()/&%\- ]+")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove URLs, e-mails, odd symbols and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _NON_TEXT.sub(" ", text)
    text = _SPACES.sub(" ", text)
    return text.strip()
