py.install_sources(
  ['__init__.py'],
  subdir: 'emeans/_kernels',
)

foreach kernel : ['_probe', '_common', '_assign', '_reduce']
  py.extension_module(
    kernel,
    [cython_gen.process(kernel + '.pyx')],
    dependencies: [np_dep, openmp_dep],
    subdir: 'emeans/_kernels',
    install: true,
  )
endforeach
