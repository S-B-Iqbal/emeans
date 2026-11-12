# 08 — Git and commits

- Conventional Commits: `feat|fix|perf|test|docs|build|ci|refactor|chore(scope): summary` (≤ 72 chars).
  Scopes: `laws`, `schedule`, `revival`, `finalize`, `estimator`, `kernels`, `bench`, `docs`, `ci`, `cline`.
- Small commits: one logical change, with its tests in the same commit.
- Branch per phase: `phase-1-numpy-ref`, `phase-2-compliance`, `phase-3-kernels`, `phase-4-bench`,
  `phase-5-packaging`, `phase-6-upstream`. Merge to `main` only after the phase gate is approved.
- Run `pre-commit run --all-files` before each commit; never use `--no-verify`.
- Never commit: `_ref/`, datasets, build dirs, generated `.c`/`.so`, benchmark raw data > 1 MB, secrets.
- ⛔ Never `git push`, create remotes on GitHub, tag, or publish without explicit confirmation in chat.
