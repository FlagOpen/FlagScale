
import importlib
import logging
import traceback
from contextlib import contextmanager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def safe_import_from(module, symbol, *, msg=None, alt=None, fallback_module=None):
    """A function used to import symbols from modules that may not be available

    This function will attempt to import a symbol with the given name from
    the given module, but it will not throw an ImportError if the symbol is not
    found. Instead, it will return a placeholder object which will raise an
    exception only if used.

    Parameters
    ----------
    module: str
        The name of the module in which the symbol is defined.
    symbol: str
        The name of the symbol to import.
    msg: str or None
        An optional error message to be displayed if this symbol is used
        after a failed import.
    alt: object
        An optional object to be used in place of the given symbol if it fails
        to import
    fallback_module: str
        Alternative name of the model in which the symbol is defined. The function will first to
        import using the `module` value and if that fails will also try the `fallback_module`.

    Returns
    -------
    Tuple(object, bool)
        The imported symbol, the given alternate, or a class derived from
        UnavailableMeta, and a boolean indicating whether the intended import was successful.
    """
    try:
        imported_module = importlib.import_module(module)
        return getattr(imported_module, symbol), True
    except ImportError:
        exception_text = traceback.format_exc()
        logger.debug(f"Import of {module} failed with: {exception_text}")
    except AttributeError:
        # if there is a fallback module try it.
        if fallback_module is not None:
            return safe_import_from(fallback_module, symbol, msg=msg, alt=alt, fallback_module=None)
        exception_text = traceback.format_exc()
        logger.info(f"Import of {symbol} from {module} failed with: {exception_text}")
    except Exception:
        exception_text = traceback.format_exc()
        raise
    if msg is None:
        msg = f"{module}.{symbol} could not be imported"
    if alt is None:
        return UnavailableMeta(symbol, (), {"_msg": msg}), False
    else:
        return alt, False
