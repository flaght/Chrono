project(
  'bomber_ctp_md',
  'cpp',
  version: '0.1.0',
  license: 'MIT',
  meson_version: '>=1.7.0',
  default_options: ['buildtype=release', 'cpp_std=c++17', 'warning_level=2'],
)

if host_machine.system() != 'linux'
  error('The exp4 native demo currently targets the Linux deployment environment only')
endif

py = import('python').find_installation(pure: false)
py_dep = py.dependency()
fs = import('fs')
pybind11_include_dir = run_command(
  py,
  '-c',
  'import pybind11; print(pybind11.get_include())',
  check: true,
).stdout().strip()
message('pybind11 includes: ' + pybind11_include_dir)

# Prefer a self-contained SDK copied into this binding directory.  During the
# migration period, fall back to the SDK files already checked into 3rd/.
local_api_dir = meson.project_source_root()
fallback_api_dir = meson.project_source_root() / '../../../../3rd/vnpy_ctp/vnpy_ctp/api'

if fs.exists(local_api_dir / 'libthostmduserapi_se.so')
  ctp_lib_dir = local_api_dir
  ctp_lib_path = local_api_dir / 'libthostmduserapi_se.so'
else
  ctp_lib_dir = fallback_api_dir
  ctp_lib_path = fallback_api_dir / 'libthostmduserapi_se.so'
endif

if fs.exists(local_api_dir / 'include/ctp/ThostFtdcMdApi.h')
  ctp_include_dir = local_api_dir / 'include'
else
  ctp_include_dir = fallback_api_dir / 'include'
endif

if not fs.exists(ctp_lib_path)
  error('CTP library not found: ' + ctp_lib_path)
endif
if not fs.exists(ctp_include_dir / 'ctp/ThostFtdcMdApi.h')
  error('CTP headers not found below: ' + ctp_include_dir)
endif

message('CTP library: ' + ctp_lib_path)
message('CTP includes: ' + ctp_include_dir)
ctp_lib = declare_dependency(
  link_args: [
    '-L' + ctp_lib_dir,
    '-lthostmduserapi_se',
  ],
)

py.extension_module(
  'bomber_ctp_md',
  sources: ['src/bomber_ctp_md.cpp'],
  include_directories: include_directories('src'),
  dependencies: [py_dep, ctp_lib],
  cpp_args: [
    '-O3',
    '-I' + ctp_include_dir,
    '-I' + pybind11_include_dir,
    '-Wno-delete-incomplete',
    '-Wno-sign-compare',
  ],
  link_args: ['-Wl,-rpath,$ORIGIN'],
  install: true,
)

install_data(
  ctp_lib_path,
  install_dir: py.get_install_dir(),
)
