from __future__ import annotations

import re

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


def normalize_text(raw_text: str) -> str:
    blocks: list[str] = []
    for block in _PARAGRAPH_SPLIT.split(raw_text.strip()):
        cleaned = re.sub(r"\s+", " ", block.strip())
        if cleaned:
            blocks.append(cleaned)
    return "\n\n".join(blocks)


def split_text_into_chunks(raw_text: str, max_chars: int = 240) -> list[str]:
    if max_chars < 10:
        raise ValueError("max_chars must be at least 10.")

    normalized = normalize_text(raw_text)
    if not normalized:
        return []

    chunks: list[str] = []
    for paragraph in normalized.split("\n\n"):
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue

        sentences = [
            segment.strip()
            for segment in _SENTENCE_SPLIT.split(paragraph)
            if segment.strip()
        ]
        if len(sentences) <= 1:
            chunks.extend(_split_long_sentence(paragraph, max_chars))
            continue

        current: list[str] = []
        current_length = 0
        for sentence in sentences:
            if len(sentence) > max_chars:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                    current_length = 0
                chunks.extend(_split_long_sentence(sentence, max_chars))
                continue

            proposed_length = current_length + len(sentence) + (1 if current else 0)
            if proposed_length > max_chars and current:
                chunks.append(" ".join(current))
                current = [sentence]
                current_length = len(sentence)
            else:
                current.append(sentence)
                current_length = proposed_length

        if current:
            chunks.append(" ".join(current))

    return chunks


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    words = sentence.split()
    if not words:
        return []

    if len(words) == 1:
        return [
            sentence[index : index + max_chars].strip()
            for index in range(0, len(sentence), max_chars)
        ]

    pieces: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        if len(word) > max_chars:
            if current:
                pieces.append(" ".join(current))
                current = []
                current_length = 0
            pieces.extend(
                word[index : index + max_chars]
                for index in range(0, len(word), max_chars)
            )
            continue

        proposed_length = current_length + len(word) + (1 if current else 0)
        if proposed_length > max_chars and current:
            pieces.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length = proposed_length

    if current:
        pieces.append(" ".join(current))

    return pieces
