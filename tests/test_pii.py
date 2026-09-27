import pytest

from llmgw.pii import Masker, contains_pii, iban_valid, luhn_valid


def test_email_phone_iban_and_card_are_masked_and_restored():
    text = (
        "Contact jane.doe@example.org or +33 6 12 34 56 78. "
        "Pay to GB82 WEST 1234 5698 7654 32 with card 4111 1111 1111 1111."
    )
    masker = Masker()
    masked = masker.mask(text)
    assert "jane.doe@example.org" not in masked and "[EMAIL_1]" in masked
    assert "[PHONE_1]" in masked and "[IBAN_1]" in masked and "[CARD_1]" in masked
    assert masker.total == 4
    assert masker.restore(masked) == text


def test_same_value_gets_the_same_placeholder_across_messages():
    masker = Masker()
    out = masker.mask_messages(
        [
            {"role": "user", "content": "Write to a@b.org."},
            {"role": "assistant", "content": "Sure."},
            {"role": "user", "content": "Also copy a@b.org and c@d.org."},
        ]
    )
    assert out[0]["content"] == "Write to [EMAIL_1]."
    assert out[2]["content"] == "Also copy [EMAIL_1] and [EMAIL_2]."


@pytest.mark.parametrize(
    "text",
    [
        "The budget is 12000 EUR for 2027.",
        "Meeting on 2027-03-14 at 10:30.",
        "Growth of 2.1 percent, 1,200 pages, ratio 0.65.",
        "Order 1234 5678 9012 3456 is not a valid card.",  # fails the Luhn check
        "FR00 1234 5678 9012 3456 7890 123 is not a valid IBAN.",  # fails mod-97
    ],
)
def test_ordinary_numbers_are_left_alone(text):
    assert not contains_pii([{"role": "user", "content": text}])


def test_validators():
    assert iban_valid("GB82 WEST 1234 5698 7654 32")
    assert not iban_valid("GB83 WEST 1234 5698 7654 32")
    assert luhn_valid("4111 1111 1111 1111")
    assert not luhn_valid("4111 1111 1111 1112")


def test_french_phone_format():
    assert contains_pii([{"role": "user", "content": "Call 06 12 34 56 78 tomorrow."}])
