import logging


def get_patch_logger():
    """
    Create a logger for patching operations.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("FlagScalePatchLogger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[FlagScale-Patch] %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Prevent the logger from propagating messages to the root logger
        logger.propagate = False

    return logger


def get_unpatch_logger():
    """
    Create a logger for unpatching operations.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("FlagScaleUnpatchLogger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[FlagScale-Unpatch] %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Prevent the logger from propagating messages to the root logger
        logger.propagate = False

    return logger
