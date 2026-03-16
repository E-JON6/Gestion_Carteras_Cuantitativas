"""Service layer exports."""

from services.trade_email_service import send_daily_trade_email

__all__ = ["send_daily_trade_email"]
