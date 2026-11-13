python_sources = [
  '__init__.py',
  '_backend.py',
  '_compat.py',
  '_emeans.py',
  '_numpy_backend.py',
  '_revival.py',
  '_scaling_laws.py',
  '_schedule.py',
  '_validation.py',
]

py.install_sources(
  python_sources,
  subdir: 'emeans',
)

py.install_sources(
  ['py.typed'],
  subdir: 'emeans',
  pure: false,
)

subdir('_kernels')
subdir('tests')
