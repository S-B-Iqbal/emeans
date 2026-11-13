py.install_sources(
  ['__init__.py'],
  subdir: 'emeans/_kernels',
)

py.extension_module(
  '_probe',
  [cython_gen.process('_probe.pyx')],
  dependencies: [np_dep, openmp_dep],
  subdir: 'emeans/_kernels',
  install: true,
)
