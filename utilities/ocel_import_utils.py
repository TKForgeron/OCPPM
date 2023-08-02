import re
from typing import Any, Union


def natural_key(text):
    """
    Give this function to a sorting function in the `key` argument,
    to sort a list of strings human-like.
    """
    _atoi = lambda text: int(text) if text.isdigit() else text
    return [_atoi(c) for c in re.split(r"(\d+)", text)]


def extract_values_from_file_string(
    s: str,
    key: str,
    key_start_delimiter: str = "[",
    key_end_delimiter: str = "]",
    split_on: str = "_",
) -> list[str]:
    start_idx = s.find(key + key_start_delimiter)
    if start_idx == -1:
        raise ValueError(f"Could not find '{key+key_start_delimiter}' in {s}")
    start_idx += len(key) + 1
    end_idx = s.find(key_end_delimiter, start_idx)
    if end_idx == -1:
        raise ValueError(f"Could not find '{key_end_delimiter}' in {s}")
    value_str = s[start_idx:end_idx]
    values = value_str.split(split_on)
    return values
