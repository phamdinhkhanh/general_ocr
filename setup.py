import os
import os.path as osp
import glob
import shutil
import sys
import warnings
from setuptools import find_packages, setup


EXT_TYPE = ''
try:
    import torch
    if torch.__version__ == 'parrots':
        from parrots.utils.build_extension import BuildExtension
        EXT_TYPE = 'parrots'
    else:
        from torch.utils.cpp_extension import BuildExtension
        EXT_TYPE = 'pytorch'
    cmd_class = {'build_ext': BuildExtension}
except ModuleNotFoundError:
    cmd_class = {}
    print('Skip building ext ops due to the absence of torch.')


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'general_ocr/version.py'
is_windows = sys.platform == 'win32'


def add_mim_extention():
    """Add extra files that are required to support MIM into the package.
    These files will be added by creating a symlink to the originals if the
    package is installed in `editable` mode (e.g. pip install -e .), or by
    copying from the originals otherwise.
    """

    # parse installment mode
    if 'develop' in sys.argv:
        # installed by `pip install -e .`
        mode = 'symlink'
    elif 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
        # installed by `pip install .`
        # or create source distribution by `python setup.py sdist`
        mode = 'copy'
    else:
        return

    filenames = ['tools', 'configs', 'model-index.yml']
    repo_path = osp.dirname(__file__)
    mim_path = osp.join(repo_path, 'general_ocr', '.mim')
    os.makedirs(mim_path, exist_ok=True)

    for filename in filenames:
        if osp.exists(filename):
            src_path = osp.join(repo_path, filename)
            tar_path = osp.join(mim_path, filename)

            if osp.isfile(tar_path) or osp.islink(tar_path):
                os.remove(tar_path)
            elif osp.isdir(tar_path):
                shutil.rmtree(tar_path)

            if mode == 'symlink':
                src_relpath = osp.relpath(src_path, osp.dirname(tar_path))
                os.symlink(src_relpath, tar_path)
            elif mode == 'copy':
                if osp.isfile(src_path):
                    shutil.copyfile(src_path, tar_path)
                elif osp.isdir(src_path):
                    shutil.copytree(src_path, tar_path)
                else:
                    warnings.warn(f'Cannot copy file {src_path}.')
            else:
                raise ValueError(f'Invalid mode {mode}')


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    import sys
    # return short version for sdist
    if 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
        return locals()['short_version']
    else:
        return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strip
    specific version information.
    Args:
        fname (str): Path to requirements file.
        with_version (bool, default=False): If True, include version specs.
    Returns:
        info (list[str]): List of requirements items.
    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    import re
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


def get_extensions():
    extensions = []

    if os.getenv('GENERAL_OCR_WITH_TRT', '0') != '0':
        ext_name = 'general_ocr._ext_trt'
        from torch.utils.cpp_extension import include_paths, library_paths
        library_dirs = []
        libraries = []
        include_dirs = []
        tensorrt_path = os.getenv('TENSORRT_DIR', '0')
        tensorrt_lib_path = glob.glob(
            os.path.join(tensorrt_path, 'targets', '*', 'lib'))[0]
        library_dirs += [tensorrt_lib_path]
        libraries += ['nvinfer', 'nvparsers', 'nvinfer_plugin']
        libraries += ['cudart']
        define_macros = []
        extra_compile_args = {'cxx': []}

        include_path = os.path.abspath('./general_ocr/ops/csrc/common/cuda')
        include_trt_path = os.path.abspath('./general_ocr/ops/csrc/tensorrt')
        include_dirs.append(include_path)
        include_dirs.append(include_trt_path)
        include_dirs.append(os.path.join(tensorrt_path, 'include'))
        include_dirs += include_paths(cuda=True)

        op_files = glob.glob('./general_ocr/ops/csrc/tensorrt/plugins/*')
        define_macros += [('GENERAL_OCR_WITH_CUDA', None)]
        define_macros += [('GENERAL_OCR_WITH_TRT', None)]
        cuda_args = os.getenv('GENERAL_OCR_CUDA_ARGS')
        extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
        library_dirs += library_paths(cuda=True)

        from setuptools import Extension
        ext_ops = Extension(
            name=ext_name,
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            language='c++',
            library_dirs=library_dirs,
            libraries=libraries)
        extensions.append(ext_ops)

    if os.getenv('GENERAL_OCR_WITH_OPS', '0') == '0':
        return extensions

    if EXT_TYPE == 'parrots':
        ext_name = 'general_ocr._ext'
        from parrots.utils.build_extension import Extension
        # new parrots op impl do not use GENERAL_OCR_USE_PARROTS
        # define_macros = [('GENEARL_OCR_USE_PARROTS', None)]
        define_macros = []
        include_dirs = []
        op_files = glob.glob('./general_ocr/ops/csrc/pytorch/cuda/*.cu') +\
            glob.glob('./general_ocr/ops/csrc/parrots/*.cpp')
        include_dirs.append(os.path.abspath('./general_ocr/ops/csrc/common'))
        include_dirs.append(os.path.abspath('./general_ocr/ops/csrc/common/cuda'))
        cuda_args = os.getenv('GENERAL_OCR_CUDA_ARGS')
        extra_compile_args = {
            'nvcc': [cuda_args] if cuda_args else [],
            'cxx': [],
        }
        if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
            define_macros += [('GENERAL_OCR_WITH_CUDA', None)]
            extra_compile_args['nvcc'] += [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        ext_ops = Extension(
            name=ext_name,
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            cuda=True,
            pytorch=True)
        extensions.append(ext_ops)
    elif EXT_TYPE == 'pytorch':
        ext_name = 'general_ocr._ext'
        from torch.utils.cpp_extension import CppExtension, CUDAExtension

        # prevent ninja from using too many resources
        try:
            import psutil
            num_cpu = len(psutil.Process().cpu_affinity())
            cpu_use = max(4, num_cpu - 1)
        except (ModuleNotFoundError, AttributeError):
            cpu_use = 4

        os.environ.setdefault('MAX_JOBS', str(cpu_use))
        define_macros = []
        extra_compile_args = {'cxx': []}
        include_dirs = []

        is_rocm_pytorch = False
        try:
            from torch.utils.cpp_extension import ROCM_HOME
            is_rocm_pytorch = True if ((torch.version.hip is not None) and
                                       (ROCM_HOME is not None)) else False
        except ImportError:
            pass

        project_dir = 'general_ocr/ops/csrc/'
        if is_rocm_pytorch:
            from torch.utils.hipify import hipify_python

            hipify_python.hipify(
                project_directory=project_dir,
                output_directory=project_dir,
                includes='general_ocr/ops/csrc/*',
                show_detailed=True,
                is_pytorch_extension=True,
            )
            define_macros += [('GENERAL_OCR_WITH_CUDA', None)]
            define_macros += [('HIP_DIFF', None)]
            cuda_args = os.getenv('GENERAL_OCR_CUDA_ARGS')
            extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
            op_files = glob.glob('./general_ocr/ops/csrc/pytorch/hip/*')
            extension = CUDAExtension
            include_dirs.append(os.path.abspath('./general_ocr/ops/csrc/common/hip'))
        elif torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
            define_macros += [('GENERAL_OCR_WITH_CUDA', None)]
            cuda_args = os.getenv('GENERAL_OCR_CUDA_ARGS')
            extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
            op_files = glob.glob('./general_ocr/ops/csrc/pytorch/*.cpp') + \
                glob.glob('./general_ocr/ops/csrc/pytorch/cuda/*.cu')
            extension = CUDAExtension
            include_dirs.append(os.path.abspath('./general_ocr/ops/csrc/common'))
            include_dirs.append(os.path.abspath('./general_ocr/ops/csrc/common/cuda'))
        else:
            print(f'Compiling {ext_name} without CUDA')
            op_files = glob.glob('./general_ocr/ops/csrc/pytorch/*.cpp')
            extension = CppExtension
            include_dirs.append(os.path.abspath('./general_ocr/ops/csrc/common'))

        ext_ops = extension(
            name=ext_name,
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args)
        extensions.append(ext_ops)

    if EXT_TYPE == 'pytorch' and os.getenv('GENERAL_OCR_WITH_ORT', '0') != '0':
        ext_name = 'general_ocr._ext_ort'
        from torch.utils.cpp_extension import library_paths, include_paths
        import onnxruntime
        library_dirs = []
        libraries = []
        include_dirs = []
        ort_path = os.getenv('ONNXRUNTIME_DIR', '0')
        library_dirs += [os.path.join(ort_path, 'lib')]
        libraries.append('onnxruntime')
        define_macros = []
        extra_compile_args = {'cxx': []}

        include_path = os.path.abspath('./general_ocr/ops/csrc/onnxruntime')
        include_dirs.append(include_path)
        include_dirs.append(os.path.join(ort_path, 'include'))

        op_files = glob.glob('./general_ocr/ops/csrc/onnxruntime/cpu/*')
        if onnxruntime.get_device() == 'GPU' or os.getenv('FORCE_CUDA',
                                                          '0') == '1':
            define_macros += [('GENERAL_OCR_WITH_CUDA', None)]
            cuda_args = os.getenv('GENERAL_OCR_CUDA_ARGS')
            extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
            op_files += glob.glob('./general_ocr/ops/csrc/onnxruntime/gpu/*')
            include_dirs += include_paths(cuda=True)
            library_dirs += library_paths(cuda=True)
        else:
            include_dirs += include_paths(cuda=False)
            library_dirs += library_paths(cuda=False)

        from setuptools import Extension
        ext_ops = Extension(
            name=ext_name,
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            language='c++',
            library_dirs=library_dirs,
            libraries=libraries)
        extensions.append(ext_ops)

    return extensions


if __name__ == '__main__':
    add_mim_extention()
    library_dirs = [
        lp for lp in os.environ.get('LD_LIBRARY_PATH', '').split(':')
        if len(lp) > 1
    ]

    setup(
        name='general_ocr',
        version=get_version(),
        description='Text Detection, OCR, and NLP Toolbox',
        long_description=readme(),
        long_description_content_type='text/markdown',
        maintainer='General OCR Authors',
        maintainer_email='',
        keywords='Text Detection, OCR, KIE, NLP',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        include_package_data=True,
        url='',
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        license='Apache License 2.0',
        setup_requires=parse_requirements('requirements/build.txt'),
        tests_require=parse_requirements('requirements/tests.txt'),
        install_requires=parse_requirements('requirements/runtime.txt'),
        ext_modules=get_extensions(),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
        },
        cmdclass=cmd_class,
        zip_safe = False
        )