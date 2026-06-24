"""Token resolution models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TokenResolution:
    token: str
    source: str
    script_output: str = ""

    @property
    def masked_token(self) -> str:
        if len(self.token) <= 12:
            return "***"
        return f"{self.token[:6]}...{self.token[-6:]}"
