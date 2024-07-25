
_GLOBAL_EXTRA_VALID_DATASETS = None


def get_extra_valid_datasets():
    """Return extra_valid datasets."""""
    return _GLOBAL_EXTRA_VALID_DATASETS


def set_extra_valid_datasets(extra_valid_datasets):
    """Set extra_valid datasets."""""
    global _GLOBAL_EXTRA_VALID_DATASETS
    _GLOBAL_EXTRA_VALID_DATASETS = extra_valid_datasets
