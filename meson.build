project(
  'emeans',
  'c',
  version: '0.1.0.dev0',
  license: 'BSD-3-Clause',
  meson_version: '>=1.2.0',
)

py = import('python').find_installation(pure: false)

# numpy headers for the Cython kernels (mirrors sklearn meson.build).
incdir_numpy = run_command(
  py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true,
).stdout().strip()
# A virtualenv inside the source tree makes include_directories() reject
# an absolute path, so the numpy include is passed as a compile argument.
np_dep = declare_dependency(compile_args: ['-I' + incdir_numpy])

# OpenMP is optional (D-044): Apple clang rejects -fopenmp, so a build without it
# compiles serial kernels; the probe extension reports which one was built.
openmp_dep = dependency('OpenMP', language: 'c', required: false)

if host_machine.system() == 'windows'
  openmp_dep = declare_dependency()
endif

cython = find_program('cython')
cython_gen = generator(
  cython,
  output: '@BASENAME@.c',
  arguments: ['--fast-fail', '-3', '--output-file', '@OUTPUT@', '@INPUT@'],
)

subdir('emeans')
