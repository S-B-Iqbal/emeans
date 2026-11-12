# /release.md — Phase 5+ only. Every ⛔ step needs explicit confirmation in chat.

1. Bump version in `emeans/__init__.py` / pyproject; update CHANGELOG.md.
2. `python -m build` (sdist + wheel) and `twine check dist/*`.
3. `make -C docs html` and confirm the sphinx-gallery example runs.
4. Print (do NOT run) the commands for the user:
   ```
   gh repo create <user>/emeans --public --source=. --remote=origin --description "e-means clustering for scikit-learn"
   git push -u origin main
   git tag v<X.Y.Z> && git push origin v<X.Y.Z>
   ```
5. ⛔ Wait for confirmation before running any of them.
6. ⛔ PyPI upload only via GitHub Actions trusted publishing triggered by the user.
