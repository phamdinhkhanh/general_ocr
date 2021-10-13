# Copyright (c) GeneralOCR. All rights reserved.
from .registry import MODULE_WRAPPERS


def is_module_wrapper(module):
    """Check if a module is a module wrapper.

    The following 3 modules in GENERAL_OCR (and their subclasses) are regarded as
    module wrappers: DataParallel, DistributedDataParallel,
    MMDistributedDataParallel (the deprecated version). You may add you own
    module wrapper by registering it to general_ocr.parallel.MODULE_WRAPPERS.

    Args:
        module (nn.Module): The module to be checked.

    Returns:
        bool: True if the input module is a module wrapper.
    """
    module_wrappers = tuple(MODULE_WRAPPERS.module_dict.values())
    return isinstance(module, module_wrappers)
