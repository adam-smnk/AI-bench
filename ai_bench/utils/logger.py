import logging

from ai_bench.config.settings import get_settings


def setup_logger(name: str = "ai_bench", level=logging.INFO) -> logging.Logger:
    """
    Set up a logger with the given name and level.

    Args:
        name: Name of the logger.
        level: Default logging level if project settings is not set.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        # Disable propagation due to possible message duplication from PyTorch's logger.
        logger.propagate = False
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    settings = get_settings()
    if settings.log is not None:
        level = settings.log
    if isinstance(level, str):
        level = level.upper()
    if isinstance(level, int):
        level = logging.getLevelName(level)

    logger.setLevel(level)
    return logger
