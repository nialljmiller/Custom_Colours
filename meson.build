project(
  'custom_colours',
  'c', 'fortran',
  version: '0.1.0',
  meson_version: '>=1.1.0',
)

py = import('python').find_installation(pure: false)

np_inc = run_command(
  py, '-c', 'import numpy; print(numpy.get_include())',
  check: true,
).stdout().strip()

f2py_inc = run_command(
  py, '-c', 'import numpy.f2py; import pathlib; print(pathlib.Path(numpy.f2py.__file__).parent / "src")',
  check: true,
).stdout().strip()

py.extension_module(
  'cc_api',
  [
    'fortran/cc_kernels.f90',
    'fortran/cc_api.f90',
    f2py_inc / 'fortranobject.c',
  ],
  include_directories: [np_inc, f2py_inc],
  dependencies: py.dependency(),
  subdir: 'custom_colours',
  install: true,
  fortran_args: ['-O2', '-fPIC'],
  c_args: ['-O2', '-fPIC'],
)
