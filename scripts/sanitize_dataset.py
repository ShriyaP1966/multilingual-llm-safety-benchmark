"""
sanitize_dataset.py

Converts the master dataset into a clean UTF-8 benchmark dataset.

This script fixes:
- Windows smart quotes
- Unicode normalization
- UTF-8 encoding
- Invisible whitespace

Author: Shriya Patil
"""

from pathlib import Path
import unicodedata


MASTER_DATASET = Path("data/master_dataset.csv")
WORKING_DATASET = Path("data/working_dataset.csv")


def sanitize_text(text: str) -> str:
    """
    Normalize Unicode and replace problematic characters.
    """

    replacements = {

        "\u2018": "'",      # left single quote

        "\u2019": "'",      # right single quote

        "\u201c": '"',      # left double quote

        "\u201d": '"',      # right double quote

        "\u00a0": " ",      # non-breaking space

        "\ufeff": "",       # BOM

    }

    for old, new in replacements.items():

        text = text.replace(old, new)

    text = unicodedata.normalize("NFC", text)

    return text


def main():

    print("=" * 60)
    print("SANITIZING DATASET")
    print("=" * 60)

    raw = MASTER_DATASET.read_text(
        encoding="utf-8",
        errors="replace"
    )

    cleaned = sanitize_text(raw)

    WORKING_DATASET.write_text(
        cleaned,
        encoding="utf-8",
        newline=""
    )

    print("✓ UTF-8 normalization complete")
    print("✓ Smart quotes removed")
    print("✓ Dataset written successfully")
    print()
    print("Output:")
    print(WORKING_DATASET)
    print("=" * 60)


if __name__ == "__main__":
    main()