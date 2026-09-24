project(
  'bomber_ctp',
  'cpp',
  version: '0.1.1',
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

local_api_dir = meson.project_source_root()
ctp_lib_dir = local_api_dir
ctp_lib_path = local_api_dir / 'libthostmduserapi_se.so'
ctp_trader_lib_dir = local_api_dir
ctp_trader_lib_path = local_api_dir / 'libthosttraderapi_se.so'
ctp_include_dir = local_api_dir / 'include'

if not fs.exists(ctp_lib_path)
  error('CTP library not found: ' + ctp_lib_path)
endif
if not fs.exists(ctp_trader_lib_path)
  error('CTP TraderApi library not found: ' + ctp_trader_lib_path)
endif
if not fs.exists(ctp_include_dir / 'ctp/ThostFtdcMdApi.h')
  error('CTP headers not found below: ' + ctp_include_dir)
endif
if not fs.exists(ctp_include_dir / 'ctp/ThostFtdcTraderApi.h')
  error('CTP TraderApi header not found below: ' + ctp_include_dir)
endif

message('CTP library: ' + ctp_lib_path)
message('CTP includes: ' + ctp_include_dir)
ctp_lib = declare_dependency(
  link_args: [
    '-L' + ctp_lib_dir,
    '-lthostmduserapi_se',
  ],
)
ctp_trader_lib = declare_dependency(
  link_args: [
    '-L' + ctp_trader_lib_dir,
    '-lthosttraderapi_se',
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

py.extension_module(
  'bomber_ctp_td',
  sources: ['src/bomber_ctp_td.cpp'],
  include_directories: include_directories('src'),
  dependencies: [py_dep, ctp_trader_lib],
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
install_data(
  ctp_trader_lib_path,
  install_dir: py.get_install_dir(),
)
