# Releasing EnvTok

Set the candidate version in [pyproject.toml](../pyproject.toml), run
`uv lock`, and move its release notes from `Unreleased` to a dated version
heading in the [changelog](../CHANGELOG.md). The Git tag is `v` followed by
the exact project version. The installed package and CLI read that version
from distribution metadata.

## Qualify the candidate

Run these commands from a clean checkout of the candidate commit, using
the default development interpreter, standard CPython 3.14, and uv 0.13.6 or
newer. The package supports standard CPython 3.11–3.14:

```bash
uv sync --locked
uv run --locked prek run --all-files --show-diff-on-failure
uv run --locked mypy
uv run --locked --no-default-groups --group test pytest
uv run --locked --extra molecular --no-default-groups --group test pytest
uv build --no-sources
uv run --locked python build_tools/qualify_artifacts.py
```

`dist/` must contain exactly the candidate wheel and sdist; move artifacts from
older versions elsewhere before qualification. The artifact check installs each
distribution with its declared dependencies in a separate fresh environment,
checks installed metadata, typing and licence files, CLI entry points and
dependency integrity, and runs the full periodic quickstart with export checks.
Its JSON report is written to `runs/artifact-qualification.json` and records the
artifact hashes. Keep that report with the candidate's verification results.

The [CI workflow](../.github/workflows/ci.yml) covers Python 3.11, 3.12, 3.13 and
3.14 with locked and lowest allowed direct dependencies, plus the locked
molecular extra. It also qualifies the bundled real periodic and molecular
samples on each interpreter. All jobs must pass on the candidate commit.
Compare the candidate with the previous tag and confirm the release notes
describe the changes.

Historical reports under `examples/` describe earlier artifacts and sample
runs. They are evidence of those runs, not a substitute for qualifying the final
release commit. These engineering gates do not establish scientific accuracy;
retain the limitations in the release notes.

## Configure publishing once

The [release workflow](../.github/workflows/release.yml) uses PyPI trusted
publishing. For a first publication, configure a pending publisher in the PyPI
account that will own the project. If the project already exists and you own it,
add the publisher under that project's publishing settings instead. See the
official [new-project](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [existing-project](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
instructions.

Use these exact values for this repository:

| Setting | Value |
| --- | --- |
| PyPI project | `envtok` |
| GitHub owner | `SMTG-Bham` |
| GitHub repository | `py-envtok` |
| Workflow filename | `release.yml` |
| GitHub environment | `pypi` |

Create the matching `pypi` environment in the repository settings. Ensure its
deployment rules allow the intended release tag; add required reviewers if the
maintainers want a final publication approval. No long-lived PyPI API token is
needed. A pending publisher does not reserve the project name. The repository's
historical public name check cannot verify account permissions or this setup;
confirm both in the authenticated PyPI and GitHub settings before publishing.

## Publish the qualified commit

1. Commit the final version, lockfile, release notes and fixes. Ensure CI passes
   for that exact commit.
2. Read the candidate version with `uv version --short`. Create and push its
   matching tag on the qualified commit, then prepare a GitHub release against
   that tag with the changelog's release notes.
3. Publishing the GitHub release starts the Release workflow. Its first job
   runs CI on the tagged revision. Once CI passes, the build job checks the
   tag against the project version, builds the wheel and sdist, and qualifies
   those files for the PyPI job.
4. Review any configured `pypi` environment approval and let the publish job
   upload the qualified distributions with attestations. The Actions run keeps
   the `dist` artifact and a separate `artifact-qualification` report containing
   their hashes.
5. Confirm the published files and version on PyPI. From the tagged checkout,
   install that exact version in a fresh Python 3.14 environment.

   ```bash
   release_version=$(uv version --short)
   check_dir=$(mktemp -d)
   uv venv --python 3.14 "$check_dir/venv"
   uv pip install --python "$check_dir/venv/bin/python" "envtok==$release_version"
   "$check_dir/venv/bin/envtok" --version
   "$check_dir/venv/bin/python" -m envtok --help
   ```

The trigger is the GitHub release's `published` event. A tag and a draft
release can both exist before the upload.
