"""Reversible masking of personal data before a prompt leaves the organisation.

Detected: email addresses, phone numbers, IBANs (checksum-validated) and payment card numbers
(Luhn-validated). Each value is replaced by a placeholder such as [EMAIL_1]; the same value
always gets the same placeholder within a request, so the model can still reason about it
("reply to [EMAIL_1]"). Placeholders in the model's answer are restored before it is returned,
so the user sees real values and the external model never does.

Limits, stated plainly: names, addresses and free-text identifiers are not detected by
patterns. Teams handling that kind of data should use the local_only or local_if_pii policy,
or add a named-entity model here.
"""

from __future__ import annotations

import re
from collections import Counter

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<![\w+])(?:\+\d{1,3}[\s.-]?|0)\d(?:[\s.-]?\d{2,4}){2,5}(?![\w])")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?\b")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def iban_valid(candidate: str) -> bool:
    value = candidate.replace(" ", "")
    if not 15 <= len(value) <= 34:
        return False
    rearranged = value[4:] + value[:4]
    number = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(number) % 97 == 1


def luhn_valid(candidate: str) -> bool:
    digits = [int(d) for d in _digits(candidate)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


# Order matters: longer, validated patterns first, so a card number is not taken for a phone.
DETECTORS = (
    ("IBAN", IBAN, iban_valid),
    ("CARD", CARD, luhn_valid),
    ("EMAIL", EMAIL, lambda s: True),
    ("PHONE", PHONE, lambda s: 8 <= len(_digits(s)) <= 15),
)


class Masker:
    """Masks values across all messages of one request, and restores them in the answer."""

    def __init__(self):
        self.placeholders: dict[str, str] = {}  # original value -> placeholder
        self.counts: Counter[str] = Counter()

    def _placeholder(self, kind: str, value: str) -> str:
        if value not in self.placeholders:
            self.counts[kind] += 1
            self.placeholders[value] = f"[{kind}_{self.counts[kind]}]"
        return self.placeholders[value]

    def mask(self, text: str) -> str:
        for kind, pattern, is_valid in DETECTORS:

            def replace(match, kind=kind, is_valid=is_valid):
                value = match.group(0)
                return self._placeholder(kind, value) if is_valid(value) else value

            text = pattern.sub(replace, text)
        return text

    def mask_messages(self, messages: list[dict]) -> list[dict]:
        return [{**m, "content": self.mask(m["content"])} for m in messages]

    def restore(self, text: str) -> str:
        for value, placeholder in self.placeholders.items():
            text = text.replace(placeholder, value)
        return text

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def contains_pii(messages: list[dict]) -> bool:
    masker = Masker()
    masker.mask_messages(messages)
    return masker.total > 0
