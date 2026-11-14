# Release runbook (never executed by the agent)

Every command below is printed for review only.  Rule 01 keeps tags, GitHub releases and
PyPI uploads behind an explicit confirmation in the chat, every single time.

## 0. Preconditions

- `main` is green: the `tests` matrix (3 OS x Python 3.10-3.13), the `docs` job
  (`sphinx -W -n`) and the `wheels` job (artefacts only) all pass.
- `CHANGELOG.md` has a dated `0.1.0` section and the version in `meson.build` matches.
- The PyPI name is `emeans-cluster` (D-052): `emeans` belongs to an unrelated Rust package.
- An API token for PyPI exists locally (e.g. `~/.pypirc` or `TWINE_PASSWORD`); never commit it.

## 1. Version bump (local commit)

```bash
# edit meson.build: version: 0.1.0   (it drives pyproject dynamic version and metadata)
# edit CHANGELOG.md: turn "## [0.1.0] - not yet released" into "## [0.1.0] - YYYY-MM-DD"
git switch -c release-0.1.0
git commit -am "chore(release): 0.1.0"
```

## 2. Build and verify the distributions

```bash
python -m build                     # sdist + wheel into dist/
python -m twine check dist/*        # metadata sanity
pip install dist/*.whl              # smoke test in a throwaway venv
python -c "import emeans; from emeans._kernels import has_openmp; print(emeans.__version__, has_openmp())"
```

> **Commit the version bump before building.** `meson-python` produces the sdist with
> `meson dist`, which archives **git HEAD**, not the working tree.  An uncommitted
> `meson.build` version bump therefore yields a mislabelled artefact: the first 0.1.0 build
> produced `emeans_cluster-0.1.0.tar.gz` whose *contents* still said `0.1.0.dev0`.  Always
> commit (or at least `git add`) the bump, then confirm with
> `tar -xzOf dist/*.tar.gz */meson.build | grep version` and
> `unzip -p dist/*.whl '*/METADATA' | grep ^Version`.

## 3. Merge, tag and release (each needs explicit confirmation)

```bash
git switch main && git merge --no-ff release-0.1.0 && git push origin main
git tag -a v0.1.0 -m "emeans 0.1.0" && git push origin v0.1.0
# PyPI upload - needs the token and an explicit "yes" in the chat:
python -m twine upload -r pypi dist/*
# Optional GitHub release (needs the same confirmation):
gh release create v0.1.0 dist/* --notes-from-tag
```

## 4. Post-release

```bash
pip install emeans-cluster          # from PyPI
python -c "from emeans import EMeans; print(EMeans())"
# open a "0.2.0" section in CHANGELOG.md and start the next phase branch
```

## 5. If something is wrong

- The upload cannot be undone; a yanked release is the remedy: `twine upload` a new patch
  and mark the bad one as yanked on PyPI (manual, web UI).
- The wheels job builds artefacts only, so a broken wheel is caught before any upload.
