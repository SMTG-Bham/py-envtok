# Reference environment and golden fixtures

EnvTok's `reference-723696d` profile reproduces the tokeniser in
[`em819/omat24-chemenv-subset`](https://github.com/em819/omat24-chemenv-subset)
at commit `723696de2c26584b8a907ef900690c0caf3c7b68`. The golden fixtures in
`tests/fixtures/reference/` come from the reference code running in that
commit's own pinned dependency set (`envs/envA-chemenv`: Python 3.10.13,
ASE 3.25.0, pymatgen 2025.6.14). They do not come from the modern stack under
test.

| File | Role |
|---|---|
| `requirements.txt` | Pins copied from the reference `envA-chemenv` environment |
| `vendored_tokenizer.py` | Upstream lines 198–517 verbatim, plus `process_single_structure` without MPI/HDF5 (MIT, see `LICENSES/`) |
| `generate_golden.py` | Tokenises `tests/fixtures/corpus.extxyz` and writes the golden JSON |
| `introspect_defaults.py` | Dumps the effective library defaults the profile freezes |

## Regenerating

These commands use [uv](https://docs.astral.sh/uv/). Run them from the
repository root:

```bash
uv run python build_tools/make_fixture_corpus.py
uv venv --python 3.10.13 .ref-venv
uv pip install --python .ref-venv/bin/python -r build_tools/reference/requirements.txt
.ref-venv/bin/python build_tools/reference/generate_golden.py tests/fixtures/corpus.extxyz tests/fixtures/reference/tokens-723696d.json
.ref-venv/bin/python build_tools/reference/introspect_defaults.py > tests/fixtures/reference/defaults-723696d.json
```

The golden JSON records the full `pip freeze` of the generating environment.
Compare `introspect_defaults.py` output from the development environment with
`defaults-723696d.json`. Any difference must either be frozen in
`envtok.profiles.reference` or recorded in `docs/discrepancies.md`.

The vendored file is excluded from Ruff so that it stays byte-identical to
upstream.
