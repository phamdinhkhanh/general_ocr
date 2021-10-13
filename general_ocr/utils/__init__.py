# flake8: noqa

from .config import Config, ConfigDict, DictAction
from .misc import (check_prerequisites, concat_list, deprecated_api_warning,
                   import_modules_from_strings, is_list_of,
                   is_method_overridden, is_seq_of, is_str, is_tuple_of,
                   iter_cast, list_cast, requires_executable, requires_package,
                   slice_list, to_1tuple, to_2tuple, to_3tuple, to_4tuple,
                   to_ntuple, tuple_cast)
from .path import (check_file_exist, fopen, is_filepath, mkdir_or_exist,
                   scandir, symlink)
from .progressbar import (ProgressBar, track_iter_progress,
                          track_parallel_progress, track_progress)
from .testing import (assert_attrs_equal, assert_dict_contains_subset,
                      assert_dict_has_keys, assert_is_norm_layer,
                      assert_keys_equal, assert_params_all_zeros,
                      check_python_script)
from .timer import Timer, TimerError, check_time
from .version_utils import digit_version, get_git_hash

# -------------------------------------------
from .registry import Registry, build_from_cfg

from .box_util import is_on_same_line, stitch_boxes_into_lines
from .check_argument import (equal_len, is_2dlist, is_3dlist, is_ndarray_list,
                             is_none_or_type, is_type_list, valid_boundary)
from .collect_env import collect_env
from .data_convert_util import convert_annotations
from .fileio import list_from_file, list_to_file
from .img_util import drop_orientation, is_not_png
from .lmdb_util import lmdb_converter
from .logger import get_root_logger
from .model import revert_sync_batchnorm
from .string_util import StringStrip
# ------------------------------------------
from .collect_env import collect_env
from .logger import get_root_logger
# ------------------------------------------
from .env import collect_env
from .logging_ import get_logger, print_log
from .parrots_jit import jit, skip_no_elena
from .parrots_wrapper import (
    TORCH_VERSION, BuildExtension, CppExtension, CUDAExtension, DataLoader,
    PoolDataLoader, SyncBatchNorm, _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd,
    _AvgPoolNd, _BatchNorm, _ConvNd, _ConvTransposeMixin, _InstanceNorm,
    _MaxPoolNd, get_build_config, is_rocm_pytorch, _get_cuda_home)
from .registry import Registry, build_from_cfg
from .trace import is_jit_tracing


# try:
#     import torch
# except ImportError:
#     __all__ = [
#         'Config', 'ConfigDict', 'DictAction', 'is_str', 'iter_cast',
#         'list_cast', 'tuple_cast', 'is_seq_of', 'is_list_of', 'is_tuple_of',
#         'slice_list', 'concat_list', 'check_prerequisites', 'requires_package',
#         'requires_executable', 'is_filepath', 'fopen', 'check_file_exist',
#         'mkdir_or_exist', 'symlink', 'scandir', 'ProgressBar',
#         'track_progress', 'track_iter_progress', 'track_parallel_progress',
#         'Timer', 'TimerError', 'check_time', 'deprecated_api_warning',
#         'digit_version', 'get_git_hash', 'import_modules_from_strings',
#         'assert_dict_contains_subset', 'assert_attrs_equal',
#         'assert_dict_has_keys', 'assert_keys_equal', 'check_python_script',
#         'to_1tuple', 'to_2tuple', 'to_3tuple', 'to_4tuple', 'to_ntuple',
#         'is_method_overridden', 
#         # ---------------------------------
#         'Registry', 'build_from_cfg', 'get_root_logger', 'collect_env',
#         'is_3dlist', 'is_ndarray_list', 'is_type_list', 'is_none_or_type',
#         'equal_len', 'is_2dlist', 'valid_boundary', 'lmdb_converter',
#         'drop_orientation', 'convert_annotations', 'is_not_png', 'list_to_file',
#         'list_from_file', 'is_on_same_line', 'stitch_boxes_into_lines',
#         'StringStrip', 'revert_sync_batchnorm',
#         #----------------------------------
#         'get_root_logger', 'collect_env'
#     ]
# else:
#     from .env import collect_env
#     from .logging_ import get_logger, print_log
#     from .parrots_jit import jit, skip_no_elena
#     from .parrots_wrapper import (
#         TORCH_VERSION, BuildExtension, CppExtension, CUDAExtension, DataLoader,
#         PoolDataLoader, SyncBatchNorm, _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd,
#         _AvgPoolNd, _BatchNorm, _ConvNd, _ConvTransposeMixin, _InstanceNorm,
#         _MaxPoolNd, get_build_config, is_rocm_pytorch, _get_cuda_home)
#     from .registry import Registry, build_from_cfg
#     from .trace import is_jit_tracing
__all__ = [
    'Config', 'ConfigDict', 'DictAction', 'collect_env', 'get_logger',
    'print_log', 'is_str', 'iter_cast', 'list_cast', 'tuple_cast',
    'is_seq_of', 'is_list_of', 'is_tuple_of', 'slice_list', 'concat_list',
    'check_prerequisites', 'requires_package', 'requires_executable',
    'is_filepath', 'fopen', 'check_file_exist', 'mkdir_or_exist',
    'symlink', 'scandir', 'ProgressBar', 'track_progress',
    'track_iter_progress', 'track_parallel_progress', 'Registry',
    'build_from_cfg', 'Timer', 'TimerError', 'check_time', 'SyncBatchNorm',
    '_AdaptiveAvgPoolNd', '_AdaptiveMaxPoolNd', '_AvgPoolNd', '_BatchNorm',
    '_ConvNd', '_ConvTransposeMixin', '_InstanceNorm', '_MaxPoolNd',
    'get_build_config', 'BuildExtension', 'CppExtension', 'CUDAExtension',
    'DataLoader', 'PoolDataLoader', 'TORCH_VERSION',
    'deprecated_api_warning', 'digit_version', 'get_git_hash',
    'import_modules_from_strings', 'jit', 'skip_no_elena',
    'assert_dict_contains_subset', 'assert_attrs_equal',
    'assert_dict_has_keys', 'assert_keys_equal', 'assert_is_norm_layer',
    'assert_params_all_zeros', 'check_python_script',
    'is_method_overridden', 'is_jit_tracing', 'is_rocm_pytorch',
    '_get_cuda_home', 
    # ---------------------------------
    'Registry', 'build_from_cfg', 'get_root_logger', 'collect_env',
    'is_3dlist', 'is_ndarray_list', 'is_type_list', 'is_none_or_type',
    'equal_len', 'is_2dlist', 'valid_boundary', 'lmdb_converter',
    'drop_orientation', 'convert_annotations', 'is_not_png', 'list_to_file',
    'list_from_file', 'is_on_same_line', 'stitch_boxes_into_lines',
    'StringStrip', 'revert_sync_batchnorm',
    #----------------------------------
    'get_root_logger', 'collect_env'
]
