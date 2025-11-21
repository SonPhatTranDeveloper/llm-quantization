import logging
import sys
import textwrap
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.

        Args:
            record: The log record to format

        Returns:
            Formatted log message string with colors
        """
        log_color = self.COLORS.get(record.levelname, "")
        levelname = f"{log_color}{self.BOLD}{record.levelname:8}{self.RESET}"
        name = f"\033[90m{record.name}\033[0m"
        message = record.getMessage()
        return f"{levelname} │ {name} │ {message}"


def setup_logging() -> None:
    """Configure logging with custom formatter."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [handler]


def format_text_box(text: str, width: int = 76, title: Optional[str] = None) -> str:
    """
    Format text in a box with proper line wrapping.

    Args:
        text: The text to format in the box
        width: The width of the box (default: 76)
        title: Optional title to display at the top of the box

    Returns:
        Formatted text string with box borders
    """
    wrapped_lines = []
    for line in text.split("\n"):
        if line.strip():
            wrapped_lines.extend(textwrap.wrap(line, width=width))
        else:
            wrapped_lines.append("")

    if title:
        title_line = f"│ {title:^{width}} │"
        box_lines = [
            "┌" + "─" * (width + 2) + "┐",
            title_line,
            "├" + "─" * (width + 2) + "┤",
        ]
    else:
        box_lines = ["┌" + "─" * (width + 2) + "┐"]

    for line in wrapped_lines:
        padded_line = line.ljust(width)
        box_lines.append(f"│ {padded_line} │")

    box_lines.append("└" + "─" * (width + 2) + "┘")
    return "\n".join(box_lines)
