"""Container bootstrap tests using synthetic credentials and a tiny CLI fixture."""

import json
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from scripts import setup_codex_container as installer


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    config = source / ".codex"
    (config / "credentials").mkdir(parents=True)
    helper = (
        "import os; from pathlib import Path; "
        "print(os.environ.get('ARENA_API_KEY') or "
        "(Path.home()/'.codex/credentials/arena.token').read_text())"
    )
    (config / "config.toml").write_text(
        'model_provider = "arena"\n'
        "[model_providers.arena.auth]\n"
        'command = "/missing/host/python3"\n'
        f"args = {json.dumps(['-c', helper])}\n"
    )
    (config / "arena.config.toml").write_text('model_provider = "arena"\n')
    (config / "credentials/arena.token").write_text("synthetic-credential\n")
    prefix = source / ".local/codex"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib").mkdir()
    cli = prefix / "lib/codex.sh"
    cli.write_text("#!/bin/sh\nprintf 'codex-cli fixture-version\\n'\n")
    cli.chmod(0o755)
    (prefix / "bin/codex").symlink_to("../lib/codex.sh")
    node = prefix / "bin/node"
    node.write_text("#!/bin/sh\nprintf 'v24.fixture\\n'\n")
    node.chmod(0o755)
    return source


def test_setup_fresh_home_installs_private_profile_and_local_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The credential probe must use the file even with an inherited API key."""
    source = _source(tmp_path)
    target = tmp_path / "container"
    monkeypatch.setenv("ARENA_API_KEY", "unrelated-inherited-credential")

    version = installer.setup_codex(source, target)

    assert version == "codex-cli fixture-version"
    config = target / ".codex"
    data = tomllib.loads((config / "config.toml").read_text())
    assert data["model_providers"]["arena"]["auth"]["command"] == sys.executable
    credential = config / "credentials/arena.token"
    assert credential.read_text() == "synthetic-credential\n"
    assert stat.S_IMODE(credential.stat().st_mode) == 0o600
    assert stat.S_IMODE(credential.parent.stat().st_mode) == 0o700
    assert (target / ".local/codex/bin/codex").resolve().is_relative_to(target)
    wrapper = (target / ".local/bin/codex").read_text()
    assert 'CODEX_HOME="$HOME/.codex"' in wrapper
    assert 'PATH="$HOME/.local/bin:$HOME/.local/codex/bin:$PATH"' in wrapper


def test_setup_repeat_backs_up_credentials_and_preserves_history(
    tmp_path: Path,
) -> None:
    """Repeated setup refreshes credentials without duplicating shell setup."""
    source = _source(tmp_path)
    target = tmp_path / "container"
    target.mkdir()
    (target / ".bashrc").write_text("# existing shell customization\n")
    installer.setup_codex(source, target)
    history = target / ".codex/history.jsonl"
    history.write_text("existing conversation\n")
    (source / ".codex/credentials/arena.token").write_text("rotated-credential\n")

    installer.setup_codex(source, target)

    assert (
        target / ".codex/credentials/arena.token"
    ).read_text() == "rotated-credential\n"
    assert history.read_text() == "existing conversation\n"
    bashrc = (target / ".bashrc").read_text()
    assert "# existing shell customization" in bashrc
    assert bashrc.count(installer.MARKER_START) == 1
    backups = target / ".local/state/codex-setup"
    assert any(
        path.read_text() == "synthetic-credential\n"
        for path in backups.rglob("arena.token")
    )


def test_setup_shared_destination_symlink_refuses_overwrite(tmp_path: Path) -> None:
    """A home symlink must not turn a local install into a shared config update."""
    source = _source(tmp_path)
    target = tmp_path / "container"
    target.mkdir()
    (target / ".codex").symlink_to(source / ".codex", target_is_directory=True)
    original = (source / ".codex/config.toml").read_bytes()

    with pytest.raises(ValueError, match="symlink"):
        installer.setup_codex(source, target)

    assert (source / ".codex/config.toml").read_bytes() == original
    assert not (target / ".local/bin/codex").exists()


def test_setup_bad_runtime_preserves_existing_install(tmp_path: Path) -> None:
    """An incompatible cached binary must fail before existing files change."""
    source = _source(tmp_path)
    (source / ".local/codex/lib/codex.sh").write_text("#!/bin/sh\nexit 7\n")
    target = tmp_path / "container"
    (target / ".codex").mkdir(parents=True)
    existing = target / ".codex/config.toml"
    existing.write_text("existing config\n")

    with pytest.raises(subprocess.CalledProcessError):
        installer.setup_codex(source, target)

    assert existing.read_text() == "existing config\n"
    assert not (target / ".local/codex").exists()


def test_setup_write_failure_restores_runtime_config_and_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later file-write failure rolls back earlier successful replacements."""
    source = _source(tmp_path)
    target = tmp_path / "container"
    old_runtime = target / ".local/codex"
    old_runtime.mkdir(parents=True)
    (old_runtime / "old-runtime").write_text("keep this runtime\n")
    config = target / ".codex"
    (config / "credentials").mkdir(parents=True)
    (config / "config.toml").write_text("original config\n")
    (config / "credentials/arena.token").write_text("original-credential\n")
    write = installer._atomic_write

    def fail_env_write(path: Path, data: bytes, mode: int = 0o600) -> None:
        if path == target / ".local/share/codex/env.sh":
            raise OSError("simulated full disk")
        write(path, data, mode)

    monkeypatch.setattr(installer, "_atomic_write", fail_env_write)

    with pytest.raises(OSError, match="simulated full disk"):
        installer.setup_codex(source, target)

    assert (old_runtime / "old-runtime").read_text() == "keep this runtime\n"
    assert (config / "config.toml").read_text() == "original config\n"
    assert (config / "credentials/arena.token").read_text() == "original-credential\n"
    assert not (target / ".local/bin/codex").exists()


@pytest.mark.parametrize("failure_step", ["backup", "install"])
def test_setup_rename_failure_keeps_original_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_step: str
) -> None:
    """Failed runtime renames must never remove the existing installation."""
    source = _source(tmp_path)
    target = tmp_path / "container"
    runtime = target / ".local/codex"
    runtime.mkdir(parents=True)
    (runtime / "original").write_text("original runtime\n")
    rename = Path.rename

    def fail_runtime_rename(path: Path, destination: Path) -> Path:
        if failure_step == "backup" and path == runtime:
            raise OSError("simulated runtime backup failure")
        if (
            failure_step == "install"
            and path.name == "runtime"
            and destination == runtime
        ):
            raise OSError("simulated runtime install failure")
        return rename(path, destination)

    monkeypatch.setattr(Path, "rename", fail_runtime_rename)

    with pytest.raises(OSError, match="simulated runtime"):
        installer.setup_codex(source, target)

    assert (runtime / "original").read_text() == "original runtime\n"
    assert not (target / ".codex/config.toml").exists()
