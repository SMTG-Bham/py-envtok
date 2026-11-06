# Releasing EnvTok

The initial alpha prerelease is version `0.1.0a1`, with Git tag `v0.1.0a1`.
Update the version in `pyproject.toml`, refresh `uv.lock` and
update [CHANGELOG.md](../CHANGELOG.md) together for subsequent releases. The
package and CLI read the version from installed distribution metadata.

## Qualify the candidate

Run these commands from a clean checkout of the candidate commit, using
standard CPython 3.14 and uv 0.13.6 or newer:

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

The [CI workflow](../.github/workflows/ci.yml) also tests the lowest allowed
direct dependencies and qualifies the bundled real periodic and molecular
samples. All jobs must pass on the candidate commit. Inspect the change from the
previous release and confirm the release notes reflect the resulting behavior.
For the first release, change the changelog's `unreleased` label to the actual
release date before tagging.

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
2. Create and push the matching tag (`v0.1.0a1` for this release), then prepare a
   GitHub release against that tag with the changelog's release notes.
3. Publishing the GitHub release starts the Release workflow. It checks the tag
   against the package version, reruns the complete CI workflow on the tagged
   revision, then builds and qualifies the distributions before the PyPI job.
4. Review any configured `pypi` environment approval and let the publish job
   upload the qualified distributions with attestations. The Actions run keeps
   the `dist` artifact and a separate `artifact-qualification` report containing
   their hashes.
5. Confirm the published files and version on PyPI, then install from the index
   in a fresh Python 3.14 environment:

   ```bash
   uv venv --python 3.14 /tmp/envtok-release-check
   uv pip install --python /tmp/envtok-release-check/bin/python 'envtok==0.1.0a1'
   /tmp/envtok-release-check/bin/envtok --version
   /tmp/envtok-release-check/bin/python -m envtok --help
   ```

A pushed tag or a draft GitHub release alone does not publish the package. Do
not mark a failed workflow as a successful publication; inspect the failing gate
or account configuration before retrying.
