"""Logging estruturado em JSON, em substituição a chamadas de print()."""

import logging
import sys
from typing import Any


class _StructuredFormatter(logging.Formatter):
    """Formata registros de log como uma linha 'key=value' legível e parseável."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"level={record.levelname} logger={record.name} msg={record.getMessage()}"
        extras = getattr(record, "extra_fields", None)
        if extras:
            kv = " ".join(f"{k}={v}" for k, v in extras.items())
            return f"{base} {kv}"
        return base


class StructuredLogger:
    """Wrapper fino sobre logging.Logger que aceita kwargs como campos estruturados."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, event: str, **kwargs: Any) -> None:
        self._logger.log(level, event, extra={"extra_fields": kwargs})

    def info(self, event: str, **kwargs: Any) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, **kwargs)


def get_logger(name: str) -> StructuredLogger:
    """Cria (ou reutiliza) um logger estruturado configurado para stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return StructuredLogger(logger)
