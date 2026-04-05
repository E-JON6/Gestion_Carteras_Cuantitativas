"""
Envio opcional del Excel de operaciones por correo (SMTP).

Variables de entorno (todas las requeridas para enviar):
  SMTP_HOST, SMTP_USER, SMTP_PASSWORD, MAIL_FROM, MAIL_TO
Opcionales: SMTP_PORT (587), SMTP_USE_TLS (1/0), MAIL_SUBJECT_PREFIX
"""

from __future__ import annotations

import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


def send_operaciones_excel(path: str | Path, as_of_label: str = "") -> bool:
    """
    Adjunta el xlsx y envia. Devuelve True si se envio, False si faltan credenciales o error.
    """
    path = Path(path)
    if not path.is_file():
        return False

    host = os.environ.get("SMTP_HOST", "").strip()
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "")
    mail_from = os.environ.get("MAIL_FROM", "").strip()
    mail_to_raw = os.environ.get("MAIL_TO", "").strip()

    if not (host and user and password and mail_from and mail_to_raw):
        return False

    try:
        port = int(os.environ.get("SMTP_PORT", "587"))
    except ValueError:
        port = 587
    use_tls = os.environ.get("SMTP_USE_TLS", "1").strip() not in ("0", "false", "False")
    prefix = os.environ.get("MAIL_SUBJECT_PREFIX", "[Cartera quant] ").strip()
    if prefix and not prefix.endswith(" "):
        prefix += " "

    recipients = [x.strip() for x in mail_to_raw.split(",") if x.strip()]

    subject = f"{prefix}Operaciones rebalanceo"
    if as_of_label:
        subject += f" — {as_of_label}"

    body = (
        "Adjunto el archivo de operaciones de rebalanceo generado por el pipeline.\n\n"
        f"Archivo: {path.name}\n"
    )
    if as_of_label:
        body += f"Fecha referencia: {as_of_label}\n"

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with open(path, "rb") as f:
        part = MIMEApplication(f.read(), Name=path.name)
    part["Content-Disposition"] = f'attachment; filename="{path.name}"'
    msg.attach(part)

    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            if use_tls:
                server.starttls()
            server.login(user, password)
            server.sendmail(mail_from, recipients, msg.as_string())
    except OSError:
        return False

    return True
