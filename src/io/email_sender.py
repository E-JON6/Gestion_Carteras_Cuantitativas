"""Email sender for daily trading pipeline.

Sends two emails after each run:
  1. Operativa Excel to Afi (professor)
  2. Rich HTML daily summary to the team
"""

from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

from src.domain.asset import Universe
from src.io.vl_tracker import VLTracker
from src.io.portfolio_state import PortfolioStateManager


# ── Palette (matches report_generator.py) ────────────────────────────────────
BG = "#0f172a"
CARD = "#1e293b"
BORDER = "#334155"
TEXT = "#f1f5f9"
MUTED = "#94a3b8"
BLUE = "#3b82f6"
GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"


@dataclass
class EmailSender:
    """Sends daily trading emails via SMTP."""

    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    sender_email: str
    afi_recipient: str
    summary_recipient: str
    group_name: str = "Grupo4"

    @classmethod
    def from_env(cls) -> EmailSender:
        """Build from environment variables. Raises ValueError if missing."""
        user = os.environ.get("SMTP_USER", "")
        password = os.environ.get("SMTP_PASSWORD", "")
        if not user or not password:
            raise ValueError(
                "SMTP_USER and SMTP_PASSWORD must be set in .env"
            )
        return cls(
            smtp_host=os.environ.get("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(os.environ.get("SMTP_PORT", "587")),
            smtp_user=user,
            smtp_password=password,
            sender_email=os.environ.get("SMTP_SENDER", user),
            afi_recipient=os.environ.get(
                "EMAIL_AFI_RECIPIENT", "pablolopezarauzo@gmail.com"
            ),
            summary_recipient=os.environ.get(
                "EMAIL_SUMMARY_RECIPIENT", "pablolopezarauzo@gmail.com"
            ),
            group_name=os.environ.get("EMAIL_GROUP_NAME", "Grupo4"),
        )

    # ── Public API ────────────────────────────────────────────────────

    def send_operativa(
        self,
        date: pd.Timestamp,
        excel_path: str | Path,
    ) -> bool:
        """Send Operativa Excel to Afi. No body text, just the attachment."""
        subject = f"[GQ_2026] \u2013 {self.group_name}"
        canonical_name = f"Operativa_{self.group_name}.xlsx"
        attachments = [(Path(excel_path), canonical_name)]

        msg = self._build_message(
            to=self.afi_recipient,
            subject=subject,
            attachments=attachments,
        )
        return self._try_send(msg, label="operativa")

    def send_daily_summary(
        self,
        result: dict,
        vl_tracker: VLTracker,
        state_mgr: PortfolioStateManager,
        universe: Universe,
        report_path: Path | None = None,
        seguimiento_path: Path | None = None,
    ) -> bool:
        """Send the rich HTML daily summary to the summary recipient."""
        date_str = result["date"].strftime("%Y-%m-%d")
        subject = f"[Daily Report] {self.group_name} \u2014 {date_str}"

        html = self._build_summary_html(result, vl_tracker, universe)

        attachments: list[tuple[Path, str | None]] = []
        if result.get("excel_path"):
            attachments.append((Path(result["excel_path"]), None))
        if seguimiento_path and seguimiento_path.exists():
            attachments.append((seguimiento_path, None))

        msg = self._build_message(
            to=self.summary_recipient,
            subject=subject,
            body_html=html,
            attachments=attachments,
        )
        return self._try_send(msg, label="summary")

    @property
    def is_test_mode(self) -> bool:
        """True when afi and summary go to the same address (test/dev)."""
        return self.afi_recipient.strip().lower() == self.summary_recipient.strip().lower()

    def send_all(
        self,
        result: dict,
        vl_tracker: VLTracker,
        state_mgr: PortfolioStateManager,
        universe: Universe,
        report_path: Path | None = None,
        seguimiento_path: Path | None = None,
    ) -> dict[str, bool]:
        """Send both emails. Each is independent — one can fail without
        affecting the other.

        When ``is_test_mode`` (afi == summary recipient), the operativa
        mail to Afi is skipped and only the summary is sent.
        """
        results: dict[str, bool] = {}

        # Email 1: Operativa to Afi (only if trades)
        if self.is_test_mode:
            results["operativa"] = True  # skipped — test mode
        elif result.get("excel_path"):
            results["operativa"] = self.send_operativa(
                date=result["date"],
                excel_path=result["excel_path"],
            )
        else:
            results["operativa"] = True  # nothing to send

        # Email 2: Daily summary
        results["summary"] = self.send_daily_summary(
            result=result,
            vl_tracker=vl_tracker,
            state_mgr=state_mgr,
            universe=universe,
            report_path=report_path,
            seguimiento_path=seguimiento_path,
        )

        return results

    # ── HTML Builder ──────────────────────────────────────────────────

    def _build_summary_html(
        self,
        result: dict,
        vl_tracker: VLTracker,
        universe: Universe,
    ) -> str:
        date_str = result["date"].strftime("%Y-%m-%d")
        post_vl = result["post_trade_vl"]
        pre_vl = result["pre_trade_vl"]
        pnl = post_vl - pre_vl
        cash = result["cash"]
        trades = result.get("trades", [])
        positions = result.get("positions", {})
        today_prices = result.get("today_prices")

        metrics = vl_tracker.metrics()
        total_return = metrics.get("total_return", 0.0)

        # Position values & weights
        pos_rows = []
        positions_value = 0.0
        for ticker, shares in sorted(positions.items()):
            if abs(shares) < 1e-9:
                continue
            price = today_prices.get_price(ticker) if today_prices else 0.0
            value = shares * price
            positions_value += abs(value)
            pos_rows.append({
                "ticker": ticker,
                "shares": shares,
                "price": price,
                "value": value,
                "weight": value / post_vl if post_vl else 0.0,
            })

        leverage = positions_value / post_vl if post_vl else 0.0

        # Market movers
        market_movers = self._compute_market_movers(result, universe)

        # Compliance
        compliance = self._check_compliance(pos_rows, leverage)

        # ── Build HTML ───────────────────────────────────────────────
        pnl_color = GREEN if pnl >= 0 else RED
        pnl_arrow = "\u25b2" if pnl >= 0 else "\u25bc"
        ret_color = GREEN if total_return >= 0 else RED

        html_parts = [
            f'<!DOCTYPE html><html><head><meta charset="utf-8"></head>',
            f'<body style="margin:0;padding:0;background:{BG};font-family:'
            f"'Inter','Segoe UI',system-ui,sans-serif;\">",
            # Container
            f'<div style="max-width:680px;margin:0 auto;padding:24px 16px;">',
            # Header
            f'<div style="text-align:center;padding:24px 0 16px;">',
            f'<h1 style="margin:0;color:{TEXT};font-size:22px;font-weight:700;">'
            f'{self.group_name} Daily Report</h1>',
            f'<p style="margin:6px 0 0;color:{MUTED};font-size:14px;">{date_str}</p>',
            f'</div>',
        ]

        # ── Section 1: Portfolio Overview ─────────────────────────────
        html_parts.append(self._card_start("Portfolio Overview"))
        html_parts.append(
            f'<table style="width:100%;border-collapse:collapse;">'
            f'<tr>'
            f'<td style="padding:8px 0;color:{MUTED};font-size:13px;">Valor Liquidativo</td>'
            f'<td style="padding:8px 0;text-align:right;color:{TEXT};font-size:18px;font-weight:700;">'
            f'\u20ac{post_vl:,.2f}</td></tr>'
            f'<tr>'
            f'<td style="padding:8px 0;color:{MUTED};font-size:13px;">P&L del d\u00eda</td>'
            f'<td style="padding:8px 0;text-align:right;color:{pnl_color};font-size:16px;font-weight:600;">'
            f'{pnl_arrow} \u20ac{pnl:+,.2f}</td></tr>'
            f'<tr>'
            f'<td style="padding:8px 0;color:{MUTED};font-size:13px;">Cash</td>'
            f'<td style="padding:8px 0;text-align:right;color:{TEXT};font-size:14px;">'
            f'\u20ac{cash:,.2f}</td></tr>'
            f'<tr>'
            f'<td style="padding:8px 0;color:{MUTED};font-size:13px;">Return total</td>'
            f'<td style="padding:8px 0;text-align:right;color:{ret_color};font-size:14px;font-weight:600;">'
            f'{total_return:+.2%}</td></tr>'
            f'<tr>'
            f'<td style="padding:8px 0;color:{MUTED};font-size:13px;">Leverage</td>'
            f'<td style="padding:8px 0;text-align:right;color:{TEXT};font-size:14px;">'
            f'{leverage:.1%}</td></tr>'
            f'</table>'
        )
        html_parts.append(self._card_end())

        # ── Section 2: Trades ─────────────────────────────────────────
        if trades:
            html_parts.append(self._card_start(f"Trades del d\u00eda ({len(trades)})"))
            html_parts.append(self._table_header(
                ["Ticker", "Dir", "Shares", "Price", "Value", "Cost"],
            ))
            for t in trades:
                direction = "BUY" if t.shares > 0 else "SELL"
                dir_color = GREEN if t.shares > 0 else RED
                value = abs(t.shares * t.price)
                html_parts.append(
                    f'<tr>'
                    f'<td style="{self._td_style()}">{t.ticker}</td>'
                    f'<td style="{self._td_style()}color:{dir_color};font-weight:600;">{direction}</td>'
                    f'<td style="{self._td_style()}text-align:right;">{abs(t.shares):,.2f}</td>'
                    f'<td style="{self._td_style()}text-align:right;">\u20ac{t.price:.4f}</td>'
                    f'<td style="{self._td_style()}text-align:right;">\u20ac{value:,.2f}</td>'
                    f'<td style="{self._td_style()}text-align:right;">\u20ac{t.cost:,.2f}</td>'
                    f'</tr>'
                )
            total_cost = sum(t.cost for t in trades)
            html_parts.append(
                f'<tr><td colspan="5" style="{self._td_style()}color:{MUTED};text-align:right;">'
                f'Total cost</td>'
                f'<td style="{self._td_style()}text-align:right;color:{AMBER};">'
                f'\u20ac{total_cost:,.2f}</td></tr>'
            )
            html_parts.append('</table>')
            html_parts.append(self._card_end())

        # ── Section 3: Positions ──────────────────────────────────────
        if pos_rows:
            html_parts.append(self._card_start(f"Posiciones ({len(pos_rows)})"))
            html_parts.append(self._table_header(
                ["Ticker", "Shares", "Price", "Value", "Weight"],
            ))
            for p in sorted(pos_rows, key=lambda x: -abs(x["weight"])):
                w_pct = p["weight"] * 100
                w_color = RED if w_pct < 1.0 else TEXT
                html_parts.append(
                    f'<tr>'
                    f'<td style="{self._td_style()}">{p["ticker"]}</td>'
                    f'<td style="{self._td_style()}text-align:right;">{p["shares"]:,.2f}</td>'
                    f'<td style="{self._td_style()}text-align:right;">\u20ac{p["price"]:.4f}</td>'
                    f'<td style="{self._td_style()}text-align:right;">\u20ac{p["value"]:,.2f}</td>'
                    f'<td style="{self._td_style()}text-align:right;color:{w_color};">{w_pct:.2f}%</td>'
                    f'</tr>'
                )
            html_parts.append('</table>')
            html_parts.append(self._card_end())

        # ── Section 4: Metrics ────────────────────────────────────────
        if metrics:
            html_parts.append(self._card_start("Performance Metrics"))
            html_parts.append(
                f'<table style="width:100%;border-collapse:collapse;">'
            )
            metric_labels = {
                "total_return": "Total Return",
                "annualized_return": "Annualized Return",
                "annualized_volatility": "Annualized Volatility",
                "sharpe_ratio": "Sharpe Ratio",
                "max_drawdown": "Max Drawdown",
                "calmar_ratio": "Calmar Ratio",
            }
            for key, label in metric_labels.items():
                val = metrics.get(key)
                if val is None:
                    continue
                if key in ("sharpe_ratio", "calmar_ratio"):
                    fmt_val = f"{val:.3f}"
                else:
                    fmt_val = f"{val:.2%}"
                    if key == "max_drawdown":
                        val_color = RED
                    elif val >= 0:
                        val_color = GREEN
                    else:
                        val_color = RED
                if key in ("sharpe_ratio", "calmar_ratio"):
                    val_color = GREEN if val > 0 else RED

                html_parts.append(
                    f'<tr>'
                    f'<td style="padding:6px 0;color:{MUTED};font-size:13px;">{label}</td>'
                    f'<td style="padding:6px 0;text-align:right;color:{val_color};'
                    f'font-size:14px;font-weight:600;">{fmt_val}</td>'
                    f'</tr>'
                )
            html_parts.append('</table>')

            # Costes acumulados
            costes = result.get("costes_acumulados", 0.0)
            n_ops = result.get("n_operaciones", 0)
            html_parts.append(
                f'<div style="margin-top:12px;padding-top:12px;border-top:1px solid {BORDER};">'
                f'<table style="width:100%;border-collapse:collapse;">'
                f'<tr><td style="padding:4px 0;color:{MUTED};font-size:12px;">Costes acumulados</td>'
                f'<td style="padding:4px 0;text-align:right;color:{AMBER};font-size:12px;">'
                f'\u20ac{costes:,.2f}</td></tr>'
                f'<tr><td style="padding:4px 0;color:{MUTED};font-size:12px;">Operaciones totales</td>'
                f'<td style="padding:4px 0;text-align:right;color:{TEXT};font-size:12px;">'
                f'{n_ops}</td></tr>'
                f'</table></div>'
            )
            html_parts.append(self._card_end())

        # ── Section 5: Market Movers ──────────────────────────────────
        if market_movers:
            html_parts.append(self._card_start("Market Movers (Universo)"))
            html_parts.append(self._table_header(
                ["Ticker", "Price", "Change"],
            ))
            for m in market_movers:
                chg_color = GREEN if m["change"] >= 0 else RED
                chg_arrow = "\u25b2" if m["change"] >= 0 else "\u25bc"
                html_parts.append(
                    f'<tr>'
                    f'<td style="{self._td_style()}">{m["ticker"]}</td>'
                    f'<td style="{self._td_style()}text-align:right;">\u20ac{m["price"]:.4f}</td>'
                    f'<td style="{self._td_style()}text-align:right;color:{chg_color};">'
                    f'{chg_arrow} {m["change"]:+.2%}</td>'
                    f'</tr>'
                )
            html_parts.append('</table>')
            html_parts.append(self._card_end())

        # ── Section 6: Compliance ─────────────────────────────────────
        if compliance:
            html_parts.append(self._card_start("Compliance"))
            for check in compliance:
                icon = "\u2705" if check["ok"] else "\u274c"
                color = GREEN if check["ok"] else RED
                html_parts.append(
                    f'<div style="padding:6px 0;color:{color};font-size:13px;">'
                    f'{icon} {check["label"]}: {check["detail"]}</div>'
                )
            html_parts.append(self._card_end())

        # ── Footer ────────────────────────────────────────────────────
        html_parts.append(
            f'<div style="text-align:center;padding:20px 0 8px;color:{MUTED};font-size:11px;">'
            f'Afi MFC Gesti\u00f3n Cuantitativa | {self.group_name} | Generated automatically'
            f'</div>'
        )

        html_parts.append('</div></body></html>')
        return '\n'.join(html_parts)

    # ── Helpers ───────────────────────────────────────────────────────

    def _compute_market_movers(
        self, result: dict, universe: Universe,
    ) -> list[dict]:
        history_df = result.get("history_df")
        if history_df is None or len(history_df) < 2:
            return []

        today = history_df.iloc[-1]
        yesterday = history_df.iloc[-2]

        movers = []
        for ticker in universe.tickers:
            if ticker not in today.index or ticker not in yesterday.index:
                continue
            p_today = today[ticker]
            p_yesterday = yesterday[ticker]
            if pd.isna(p_today) or pd.isna(p_yesterday) or p_yesterday == 0:
                continue
            change = (p_today - p_yesterday) / p_yesterday
            movers.append({
                "ticker": ticker,
                "price": p_today,
                "change": change,
            })

        movers.sort(key=lambda x: x["change"])
        # Top 3 losers + top 3 gainers
        n = min(3, len(movers))
        if len(movers) <= 6:
            return sorted(movers, key=lambda x: -abs(x["change"]))
        return movers[:n] + movers[-n:]

    def _check_compliance(
        self, pos_rows: list[dict], leverage: float,
    ) -> list[dict]:
        checks = []
        # Min weight >= 1%
        violations = [p for p in pos_rows if abs(p["weight"]) < 0.01]
        if violations:
            tickers = ", ".join(p["ticker"] for p in violations)
            checks.append({
                "ok": False,
                "label": "Min weight \u22651%",
                "detail": f"Violaciones: {tickers}",
            })
        else:
            checks.append({
                "ok": True,
                "label": "Min weight \u22651%",
                "detail": "Todas las posiciones cumplen",
            })

        # Leverage <= 200%
        checks.append({
            "ok": leverage <= 2.0,
            "label": "Leverage \u2264200%",
            "detail": f"Actual: {leverage:.1%}",
        })

        return checks

    @staticmethod
    def _card_start(title: str) -> str:
        return (
            f'<div style="background:{CARD};border:1px solid {BORDER};'
            f'border-radius:8px;padding:16px;margin-bottom:16px;">'
            f'<h2 style="margin:0 0 12px;color:{BLUE};font-size:15px;'
            f'font-weight:600;">{title}</h2>'
        )

    @staticmethod
    def _card_end() -> str:
        return '</div>'

    @staticmethod
    def _table_header(headers: list[str]) -> str:
        cols = ''.join(
            f'<th style="padding:8px 6px;text-align:left;color:{MUTED};'
            f'font-size:11px;font-weight:600;text-transform:uppercase;'
            f'border-bottom:1px solid {BORDER};">{h}</th>'
            for h in headers
        )
        return f'<table style="width:100%;border-collapse:collapse;"><tr>{cols}</tr>'

    @staticmethod
    def _td_style() -> str:
        return (
            f"padding:6px;color:{TEXT};font-size:13px;"
            f"border-bottom:1px solid {BORDER};"
        )

    # ── Email building & sending ──────────────────────────────────────

    def _build_message(
        self,
        to: str,
        subject: str,
        body_text: str | None = None,
        body_html: str | None = None,
        attachments: list[tuple[Path, str | None]] | None = None,
    ) -> MIMEMultipart:
        """Build a MIME message.

        Args:
            attachments: list of (path, display_name) tuples.
                         If display_name is None, uses the file's real name.
        """
        msg = MIMEMultipart("mixed")
        msg["From"] = self.sender_email
        msg["To"] = to
        msg["Subject"] = subject

        if body_html:
            alt = MIMEMultipart("alternative")
            alt.attach(MIMEText(
                "Este email requiere un cliente con soporte HTML.",
                "plain", "utf-8",
            ))
            alt.attach(MIMEText(body_html, "html", "utf-8"))
            msg.attach(alt)
        elif body_text:
            msg.attach(MIMEText(body_text, "plain", "utf-8"))

        for path, display_name in (attachments or []):
            if not path.exists():
                continue
            name = display_name or path.name
            with open(path, "rb") as f:
                part = MIMEApplication(f.read(), Name=name)
            part["Content-Disposition"] = f'attachment; filename="{name}"'
            msg.attach(part)

        return msg

    def _try_send(self, msg: MIMEMultipart, label: str) -> bool:
        try:
            self._send(msg)
            return True
        except Exception as e:
            print(f"  [email:{label}] ERROR: {e}")
            return False

    def _send(self, msg: MIMEMultipart) -> None:
        if self.smtp_port in (465, 2465):
            # SSL-wrapped connection (Resend, some providers)
            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=30) as server:
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
        else:
            # STARTTLS (Gmail, port 587, etc.)
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
