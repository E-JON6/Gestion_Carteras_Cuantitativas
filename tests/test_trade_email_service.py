from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from openpyxl import load_workbook

from services.trade_email_service import (
    DEFAULT_TEMPLATE_PATH,
    WORKSHEET_NAME,
    build_operativa_attachment,
    send_daily_trade_email,
    validate_trades,
)


class TradeEmailServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trades = [
            {
                "id": "ART SM Equity",
                "quantity": 5000,
                "price": 22.8,
                "ct": 0.009617639295533437,
            },
            {
                "id": "ART SM Equity",
                "quantity": -4500,
                "price": 22.8,
                "ct": 0.009617639295533437,
            },
        ]

    def test_validate_trades_rejects_zero_quantity(self) -> None:
        with self.assertRaisesRegex(ValueError, "quantity"):
            validate_trades(
                [{"id": "ART SM Equity", "quantity": 0, "price": 22.8, "ct": 0.01}]
            )

    def test_send_daily_trade_email_reports_detailed_missing_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "GMAIL_SMTP_EMAIL"):
                    send_daily_trade_email(
                        self.trades,
                        template_path=DEFAULT_TEMPLATE_PATH,
                        output_dir=temp_dir,
                    )

    def test_build_operativa_attachment_writes_expected_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = build_operativa_attachment(
                self.trades,
                template_path=DEFAULT_TEMPLATE_PATH,
                output_dir=temp_dir,
            )

            workbook = load_workbook(output_path)
            sheet = workbook[WORKSHEET_NAME]

            self.assertEqual(sheet.title, WORKSHEET_NAME)
            self.assertEqual(sheet["A1"].value, "ID")
            self.assertEqual(sheet["E1"].value, "Precio Ejecutado")
            self.assertEqual(sheet["A2"].value, "ART SM Equity")
            self.assertEqual(sheet["B2"].value, 5000)
            self.assertEqual(sheet["C2"].value, 22.8)
            self.assertEqual(sheet["D2"].number_format, "0.00%")
            self.assertEqual(sheet["E2"].value, "=C2*(1+D2)")
            self.assertEqual(sheet["E3"].value, "=C3*(1-D3)")

    @patch("services.trade_email_service.smtplib.SMTP_SSL")
    def test_send_daily_trade_email_sends_attachment(self, smtp_ssl: MagicMock) -> None:
        smtp_instance = smtp_ssl.return_value.__enter__.return_value

        with tempfile.TemporaryDirectory() as temp_dir:
            env = {
                "GMAIL_SMTP_EMAIL": "sender@example.com",
                "GMAIL_APP_PASSWORD": "app-password",
                "TRADE_EMAIL_TO": "prof@example.com",
                "TRADE_EMAIL_CC": "team@example.com",
            }
            with patch.dict(os.environ, env, clear=False):
                result = send_daily_trade_email(
                    self.trades,
                    template_path=DEFAULT_TEMPLATE_PATH,
                    output_dir=temp_dir,
                )

            self.assertEqual(result["status"], "sent")
            self.assertEqual(result["num_trades"], 2)
            self.assertTrue(Path(result["attachment_path"]).exists())
            smtp_instance.login.assert_called_once_with("sender@example.com", "app-password")
            smtp_instance.send_message.assert_called_once()


if __name__ == "__main__":
    unittest.main()
