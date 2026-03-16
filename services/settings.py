"""Environment-backed settings for the trade email workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SUBJECT = "[GQ_2026] - GrupoX"
DEFAULT_BODY = "Adjunto la operativa diaria generada automaticamente."
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = REPO_ROOT / ".env"

def _format_missing_env_error(missing: list[str], empty: list[str]) -> str:
    problems: list[str] = []
    if missing:
        problems.append(f"missing: {', '.join(missing)}")
    if empty:
        problems.append(f"empty: {', '.join(empty)}")

    details = [
        "Mail configuration is incomplete.",
        f"Problem with required environment variables: {'; '.join(problems)}",
        "",
        "Required variables:",
        "  GMAIL_SMTP_EMAIL   -> Gmail address that sends the message",
        "  GMAIL_APP_PASSWORD -> Gmail App Password, not your normal Gmail password",
        "  TRADE_EMAIL_TO     -> Recipient email(s), comma-separated if more than one",
        "",
        "Optional variables:",
        "  TRADE_EMAIL_CC       -> CC recipient(s), comma-separated",
        "  TRADE_EMAIL_SUBJECT  -> Default email subject",
        "  TRADE_EMAIL_BODY     -> Default plain-text email body",
        "",
        "Example .env:",
        "  GMAIL_SMTP_EMAIL=youraccount@gmail.com",
        "  GMAIL_APP_PASSWORD=abcdwxyzmnopqrst",
        "  TRADE_EMAIL_TO=professor@example.com",
        "  TRADE_EMAIL_CC=team1@example.com,team2@example.com",
        "",
        "How to load it in this shell before running the script:",
        f"  set -a && source {DEFAULT_ENV_PATH} && set +a",
        "",
        "Gmail setup:",
        "  1. Enable 2-Step Verification in your Google account",
        "  2. Create an App Password for Mail",
        "  3. Put that generated App Password in GMAIL_APP_PASSWORD",
    ]
    return "\n".join(details)


def _read_env(name: str, *, required: bool) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None

    value = value.strip()
    if required and not value:
        return ""

    return value or None


def parse_recipients(raw: str | None) -> list[str]:
    """Split comma/semicolon recipient strings into clean addresses."""
    if raw is None:
        return []

    normalized = raw.replace(";", ",")
    recipients = [item.strip() for item in normalized.split(",") if item.strip()]
    return recipients


@dataclass(frozen=True)
class TradeEmailSettings:
    smtp_email: str
    app_password: str
    to: list[str]
    cc: list[str]
    default_subject: str
    default_body: str

    @classmethod
    def from_env(cls) -> "TradeEmailSettings":
        missing: list[str] = []
        empty: list[str] = []

        smtp_email = _read_env("GMAIL_SMTP_EMAIL", required=True)
        if smtp_email is None:
            missing.append("GMAIL_SMTP_EMAIL")
        elif smtp_email == "":
            empty.append("GMAIL_SMTP_EMAIL")

        app_password = _read_env("GMAIL_APP_PASSWORD", required=True)
        if app_password is None:
            missing.append("GMAIL_APP_PASSWORD")
        elif app_password == "":
            empty.append("GMAIL_APP_PASSWORD")

        to = parse_recipients(_read_env("TRADE_EMAIL_TO", required=True))
        raw_to = _read_env("TRADE_EMAIL_TO", required=True)
        if raw_to is None:
            missing.append("TRADE_EMAIL_TO")
        elif raw_to == "":
            empty.append("TRADE_EMAIL_TO")
        elif not to:
            empty.append("TRADE_EMAIL_TO")

        if missing or empty:
            raise ValueError(_format_missing_env_error(missing, empty))

        cc = parse_recipients(_read_env("TRADE_EMAIL_CC", required=False))
        default_subject = _read_env("TRADE_EMAIL_SUBJECT", required=False) or DEFAULT_SUBJECT
        default_body = _read_env("TRADE_EMAIL_BODY", required=False) or DEFAULT_BODY

        return cls(
            smtp_email=smtp_email,
            app_password=app_password,
            to=to,
            cc=cc,
            default_subject=default_subject,
            default_body=default_body,
        )
