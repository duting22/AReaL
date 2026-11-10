#!/usr/bin/env python3
"""Copy a cached Codex runtime and private Arena profile into a container's home.

Publish this file at SOURCE_HOME/.local/bin/setup_codex_container.py, then run:
    python3 /shared/user/.local/bin/setup_codex_container.py
    source ~/.local/share/codex/env.sh

Alternatively pass --source-home when running the repository copy. Requires
Python 3.11+ and a Linux-compatible Codex cache at SOURCE_HOME/.local/codex.
Configuration and credentials remain outside this script and the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

MARKER_START = "# >>> codex-container-setup >>>"
MARKER_END = "# <<< codex-container-setup <<<"


def _check_destination(path: Path, target_home: Path) -> None:
    """Never follow an existing destination symlink back into shared storage."""
    if not path.is_relative_to(target_home):
        raise ValueError("Installation destinations must be inside the target home")
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError(f"Refusing to overwrite a symlink: {item}")


def _atomic_write(path: Path, data: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".codex-write-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            os.fchmod(stream.fileno(), mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _portable_config(data: bytes) -> tuple[bytes, dict]:
    config = tomllib.loads(data.decode())
    if config.get("model_provider") != "arena":
        raise ValueError("The source config.toml must select the Arena provider")
    auth = config["model_providers"]["arena"]["auth"]
    lines = data.decode().splitlines(keepends=True)
    in_auth = False
    replacements = 0
    for index, line in enumerate(lines):
        if line.lstrip().startswith("["):
            in_auth = line.strip() == "[model_providers.arena.auth]"
        if in_auth and re.match(r"\s*command\s*=", line):
            lines[index] = f"command = {json.dumps(sys.executable)}\n"
            replacements += 1
    if replacements != 1:
        raise ValueError("Expected one command in [model_providers.arena.auth]")
    result = "".join(lines).encode()
    auth = tomllib.loads(result.decode())["model_providers"]["arena"]["auth"]
    return result, auth


def _runtime_version(prefix: Path) -> str:
    for name in ("codex", "node"):
        binary = prefix / "bin" / name
        if not binary.is_file() or not binary.resolve().is_relative_to(
            prefix.resolve()
        ):
            raise ValueError(f"The local runtime is missing a self-contained {name}")
    environment = dict(
        os.environ, PATH=f"{prefix / 'bin'}:{os.environ.get('PATH', '')}"
    )
    for name in ("node", "codex"):
        result = subprocess.run(
            [str(prefix / "bin" / name), "--version"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    return result.stdout.strip()


def setup_codex(source_home: Path, target_home: Path) -> str:
    """Install local runtime/config atomically, restoring prior files on failure."""
    source_home = source_home.expanduser().resolve()
    target_home = target_home.expanduser().absolute()
    if target_home.resolve() == source_home:
        raise ValueError(
            "Run inside a container with a local home, not the source home"
        )

    runtime = target_home / ".local/codex"
    config_dir = target_home / ".codex"
    wrapper = target_home / ".local/bin/codex"
    env_file = target_home / ".local/share/codex/env.sh"
    bashrc = target_home / ".bashrc"
    source_config = source_home / ".codex"
    config, auth = _portable_config((source_config / "config.toml").read_bytes())
    profile = (source_config / "arena.config.toml").read_bytes()
    if tomllib.loads(profile.decode()).get("model_provider") != "arena":
        raise ValueError("arena.config.toml must select the Arena provider")
    token = (source_config / "credentials/arena.token").read_bytes()
    if not token.strip():
        raise ValueError("The source Arena token is empty")

    env_text = (
        'export PATH="$HOME/.local/bin:$HOME/.local/codex/bin:$PATH"\n'
        'export CODEX_HOME="$HOME/.codex"\n'
    )
    wrapper_text = (
        "#!/bin/sh\n" + env_text + 'exec "$HOME/.local/codex/bin/codex" "$@"\n'
    )
    block = f'{MARKER_START}\n. "$HOME/.local/share/codex/env.sh"\n{MARKER_END}\n'
    destinations = [
        runtime,
        config_dir / "config.toml",
        config_dir / "arena.config.toml",
        config_dir / "credentials/arena.token",
        wrapper,
        env_file,
        bashrc,
        target_home / ".local/state/codex-setup",
    ]
    for path in destinations:
        _check_destination(path, target_home)
    if runtime.exists() and not runtime.is_dir():
        raise ValueError("The local Codex prefix must be a directory")
    old_bashrc = bashrc.read_text() if bashrc.exists() else ""
    pattern = re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END) + r"\n?"
    new_bashrc = re.sub(pattern, "", old_bashrc, flags=re.DOTALL).rstrip()
    new_bashrc = (new_bashrc + "\n\n" if new_bashrc else "") + block
    payloads = {
        config_dir / "config.toml": (config, 0o600),
        config_dir / "arena.config.toml": (profile, 0o600),
        config_dir / "credentials/arena.token": (token, 0o600),
        wrapper: (wrapper_text.encode(), 0o755),
        env_file: (env_text.encode(), 0o600),
        bashrc: (
            new_bashrc.encode(),
            stat.S_IMODE(bashrc.stat().st_mode) if bashrc.exists() else 0o600,
        ),
    }
    runtime.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(
        prefix=".codex-setup-", dir=runtime.parent
    ) as work:
        work_dir = Path(work)
        staged = work_dir / "runtime"
        shutil.copytree(source_home / ".local/codex", staged, symlinks=True)
        node = staged / "bin/node"
        node_source = node.resolve() if node.is_file() else shutil.which("node")
        if node_source is None:
            raise ValueError("Cache bin/node or provide Node.js in the container")
        if node.is_symlink() or not node.is_file():
            node.unlink(missing_ok=True)
            shutil.copy2(node_source, node)
        version = _runtime_version(staged)

        # Probe the file credential in an isolated home; never log the token.
        probe_home = work_dir / "auth-probe"
        _atomic_write(probe_home / ".codex/credentials/arena.token", token)
        probe_env = dict(os.environ, HOME=str(probe_home))
        probe_env.pop("ARENA_API_KEY", None)
        probe = subprocess.run(
            [auth["command"], *auth.get("args", [])],
            env=probe_env,
            capture_output=True,
            timeout=30,
        )
        if probe.returncode or probe.stdout.strip() != token.strip():
            raise ValueError("The Arena auth helper cannot read the copied credential")

        backup_root = target_home / ".local/state/codex-setup"
        backup_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        backup = Path(tempfile.mkdtemp(prefix="backup-", dir=backup_root))
        applied: list[tuple[Path, Path | None]] = []
        old_runtime = work_dir / "previous-runtime"
        runtime_moved = False
        runtime_installed = False
        try:
            if runtime.exists():
                runtime.rename(old_runtime)
                runtime_moved = True
            staged.rename(runtime)
            runtime_installed = True
            for path, (data, mode) in payloads.items():
                previous = None
                if path.exists():
                    previous = backup / path.relative_to(target_home)
                    previous.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    shutil.copy2(path, previous)
                _atomic_write(path, data, mode)
                applied.append((path, previous))
            config_dir.chmod(0o700)
            (config_dir / "credentials").chmod(0o700)
            _runtime_version(runtime)
        except BaseException:
            for path, previous in reversed(applied):
                if previous is None:
                    path.unlink(missing_ok=True)
                else:
                    os.replace(previous, path)
            if runtime_installed:
                shutil.rmtree(runtime)
            if runtime_moved:
                old_runtime.rename(runtime)
            raise
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-home", type=Path, help="Shared home with cached Codex"
    )
    args = parser.parse_args()
    source_home = args.source_home
    if source_home is None:
        script = Path(__file__).resolve()
        if script.parent.name != "bin" or script.parent.parent.name != ".local":
            parser.error("Pass --source-home when running the repository copy")
        source_home = script.parent.parent.parent
    os.umask(0o077)
    try:
        version = setup_codex(source_home, Path.home())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        # Credential helper errors and TOML parser messages may contain secrets.
        print(
            "Setup failed; check source files, destination symlinks and runtime compatibility.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    print(f"Installed {version} with the Arena profile in {Path.home()}")
    print('Run: source "$HOME/.local/share/codex/env.sh"')


if __name__ == "__main__":
    main()
