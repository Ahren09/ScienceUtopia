"""Shared console logging configuration."""

import logging


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        log_fmt = "%(levelname)s | %(asctime)s | %(name)s | %(message)s"
        formatter = logging.Formatter(log_fmt, datefmt="%m-%d %H:%M:%S")
        log_msg = formatter.format(record)

        color = self.COLORS.get(record.levelname, self.RESET)
        return f"{color}{log_msg}{self.RESET}"


def configure_logging():
    """Configure console output and suppress noisy dependency loggers."""
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(asctime)s | %(filename)s:%(lineno)d - %(message)s",
        datefmt="%m-%d %H:%M:%S",
        handlers=[handler],
    )

    for package_name in [
        "urllib3",
        "httpx",
        "httpcore",
        "openai",
        "langchain",
        "PIL",
        "h5py",
    ]:
        logging.getLogger(package_name).setLevel(logging.ERROR)
