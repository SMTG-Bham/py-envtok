"""Check built wheel and sdist in separate fresh dependency environments.

Usage::

    uv build --no-sources
    python build_tools/qualify_artifacts.py

No system-site-packages or source checkout imports are permitted. Each artifact
installs its declared dependencies and runs the complete 20-structure quickstart.
Only the JSON report survives; temporary environments and outputs are removed.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROBE = """
import importlib.metadata
import json
import sys
from pathlib import Path
import envtok
expected_name, expected_version = sys.argv[1:]
prefix = Path(sys.prefix).resolve()
package = Path(envtok.__file__).resolve()
if sys.prefix == sys.base_prefix or not package.is_relative_to(prefix):
    raise RuntimeError('package was not imported from the isolated environment')
distribution = importlib.metadata.distribution(expected_name)
if distribution.metadata['Name'] != expected_name:
    raise RuntimeError('installed distribution name does not match the project name')
if distribution.version != expected_version or envtok.__version__ != expected_version:
    raise RuntimeError('installed version does not match the project version')
if not (package.parent / 'py.typed').is_file():
    raise RuntimeError('installed package is missing py.typed')
for license_file in (
    'LICENSE',
    'LICENSES/omat24-chemenv-subset-MIT.txt',
    'LICENSES/pymatgen-MIT.txt',
):
    if not distribution.read_text('licenses/' + license_file):
        raise RuntimeError('installed distribution is missing ' + license_file)
print(json.dumps({
    'python': sys.version.split()[0],
    'distribution': distribution.metadata['Name'],
    'version': distribution.version,
    'installed_package_in_isolated_environment': True,
    'typing_marker_and_licenses': 'passed',
    'dependencies': dict(sorted(
        (item.metadata['Name'], item.version)
        for item in importlib.metadata.distributions()
    )),
}))
"""


def _run(command: list[str], environment: dict[str, str], workspace: Path) -> str:
    """Run one bounded acceptance command and retain its diagnostics on error."""
    result = subprocess.run(  # noqa: S603 - argument lists; no shell expansion
        command,
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {command!r}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def qualify(
    artifact: Path,
    workspace: Path,
    uv: str,
    python: str,
    *,
    offline: bool,
    expected_name: str,
    expected_version: str,
) -> dict[str, Any]:
    """Install one artifact and verify the public pipeline outside the checkout."""
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(key, None)
    environment["MPLCONFIGDIR"] = str(workspace / "matplotlib")
    virtualenv = workspace / "venv"
    uv_command = [uv, "--offline"] if offline else [uv]
    started = time.perf_counter()
    _run(
        [*uv_command, "venv", "--python", python, str(virtualenv)],
        environment,
        workspace,
    )
    config = (virtualenv / "pyvenv.cfg").read_text()
    if "include-system-site-packages = false" not in config:
        raise RuntimeError("artifact check must not inherit system-site-packages")
    executable = virtualenv / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    _run(
        [*uv_command, "pip", "install", "--python", str(executable), str(artifact)],
        environment,
        workspace,
    )
    _run(
        [*uv_command, "pip", "check", "--python", str(executable)],
        environment,
        workspace,
    )
    installed = time.perf_counter()
    probe = json.loads(
        _run(
            [str(executable), "-I", "-c", PROBE, expected_name, expected_version],
            environment,
            workspace,
        )
    )
    cli = virtualenv / ("Scripts/envtok.exe" if os.name == "nt" else "bin/envtok")
    for command in (
        [str(cli), "--version"],
        [str(executable), "-I", "-m", "envtok", "--version"],
    ):
        if _run(command, environment, workspace).strip() != expected_version:
            raise RuntimeError("installed CLI version does not match the project")
    output = workspace / "quickstart"
    _run(
        [
            str(executable),
            "-I",
            str(ROOT / "examples" / "quickstart" / "quickstart.py"),
            "--workdir",
            str(output),
        ],
        environment,
        workspace,
    )
    coverage = json.loads((output / "selection" / "coverage.json").read_text())
    return {
        "artifact": artifact.name,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "status": "passed",
        "include_system_site_packages": False,
        "dependency_check": "passed",
        "cli_entrypoints": "passed",
        **probe,
        "structures": coverage["structures"],
        "atoms": coverage["atoms"],
        "classes": coverage["classes"],
        "presence": coverage["presence"],
        "manifest_and_export_round_trip": "passed",
        "seconds": {
            "install": round(installed - started, 3),
            "quickstart_and_import_check": round(time.perf_counter() - installed, 3),
        },
    }


def main() -> None:
    """Qualify the single wheel and sdist produced by a package build."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--python", default="3.14")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs" / "artifact-qualification.json"
    )
    args = parser.parse_args()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    uv = shutil.which(args.uv)
    if uv is None:
        parser.error("uv is unavailable; install it or pass --uv /path/to/uv")
    artifacts = []
    for pattern in ("*.whl", "*.tar.gz"):
        matching = sorted(args.dist.resolve().glob(pattern))
        if len(matching) != 1:
            parser.error(f"expected exactly one {pattern} in {args.dist}")
        artifacts.extend(matching)
    results: list[dict[str, Any]] = []
    for artifact in artifacts:
        print(f"Checking {artifact.name} in a fresh environment", flush=True)
        with TemporaryDirectory(prefix="envtok-artifact-") as temporary:
            result = qualify(
                artifact,
                Path(temporary),
                uv,
                args.python,
                offline=args.offline,
                expected_name=project["name"],
                expected_version=project["version"],
            )
        results.append(result)
    report = {
        "status": "passed",
        "purpose": "installed package acceptance; scientific accuracy unvalidated",
        "fixture_sha256": hashlib.sha256(
            (ROOT / "tests" / "fixtures" / "corpus.extxyz").read_bytes()
        ).hexdigest(),
        "artifacts": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Both artifacts passed; report: {args.output}")


if __name__ == "__main__":
    main()
