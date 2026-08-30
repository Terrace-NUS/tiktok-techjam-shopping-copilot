"""Shared external-provider transport seams."""

from .deepseek_http import DeepSeekTransport, HttpResponse, UrllibDeepSeekTransport

__all__ = ("DeepSeekTransport", "HttpResponse", "UrllibDeepSeekTransport")
