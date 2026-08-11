import logging
from typing import Literal, Union

Verbosity = Literal["warnings", "info", "debug"]

_LEVELS: dict[Union[Verbosity, bool], int] = {
    "warnings": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    True: logging.INFO,
    False: logging.WARNING,
}


def _normalize_verbose(verbose: Union[Verbosity, bool]) -> Verbosity:
    if isinstance(verbose, bool):
        return "info" if verbose else "warnings"
    return verbose


def configure_verbosity(verbose: Union[Verbosity, bool] = "info") -> None:
    """Configure root logger for library verbosity.

    Accepts either a Verbosity string or a bool for backward compatibility.
    Sets the root logger level and attaches a StreamHandler if the
    application has not already configured logging.  This is the
    standard entry point for library users who want output without
    manually configuring logging.
    """
    root = logging.getLogger()
    root.setLevel(_LEVELS[_normalize_verbose(verbose)])
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
