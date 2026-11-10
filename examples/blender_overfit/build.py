"""Build immutable local Arena materials and upload blueprints without publishing."""

import argparse
import copy
import json
import shutil
import tempfile
import zipapp
import zipfile
from pathlib import Path

from examples.blender_overfit.prepare import digest, write_json


def build(dataset: Path, source_env: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((dataset / "manifest.json").read_text())
    env = json.loads(source_env.read_text())["spec"]
    for role in ["agent", "reward"]:
        with tempfile.TemporaryDirectory(prefix="overfit-package-") as scratch:
            root = Path(scratch)
            package = root / "examples" / "blender_overfit"
            package.mkdir(parents=True)
            (root / "examples" / "__init__.py").write_text("")
            for path in Path(__file__).parent.glob("*.py"):
                if path.name in {
                    "control.py",
                    "gallery.py",
                    "build.py",
                    "history32.py",
                }:
                    continue
                if role == "agent" and path.name not in {
                    "__init__.py",
                    "arena.py",
                    "single_turn.py",
                    "prepare.py",
                    "scene.py",
                }:
                    continue
                shutil.copy2(path, package / path.name)
            if role == "reward":
                shutil.copytree(
                    Path(__file__).with_name("prompts"), package / "prompts"
                )
            (root / "__main__.py").write_text(
                "import sys,tempfile,zipfile\n"
                "from pathlib import Path\n"
                "with tempfile.TemporaryDirectory(prefix='overfit-runtime-') as runtime:\n"
                "    with zipfile.ZipFile(sys.argv[0]) as z: z.extractall(runtime)\n"
                "    sys.path.insert(0,runtime)\n"
                "    from examples.blender_overfit.arena import main\n"
                "    main()\n"
            )
            zipapp.create_archive(
                root,
                output / f"blender-overfit-{role}.pyz",
                interpreter="/usr/bin/env python3",
                compressed=True,
            )
    code_uri = "LOCAL:blender-overfit-agent.pyz"
    shutil.copy2(
        Path(__file__).with_name("history32.py"),
        output / "blender-overfit-history32.py",
    )
    setup = env["hooks"]["setup_script"].replace(
        'python3 "$ARENA_ENV_DIR/official/collector.pyz" --setup',
        'mkdir -p /problem_assets\ncp "$ARENA_WORKSPACE"/problem_assets/* /problem_assets/',
    )
    if "/problem_assets" not in setup:
        raise ValueError(
            "The frozen source setup hook changed; inspect it before adapting"
        )
    collect = 'python3 "$ARENA_ENV_DIR/overfit/runtime.pyz" collect\n'
    runtime_files = [
        item
        for item in env["files"]
        if not (
            item["dest"].startswith("fine-safe/")
            or item["dest"].startswith("official/")
            or item["dest"].startswith("native/task-skills/")
            or item["dest"] == "native/task-policy.json"
        )
    ]
    runtime_files.append({"uri": code_uri, "dest": "overfit/runtime.pyz"})
    all_envs = []
    for task in manifest["tasks"]:
        identity = task["task_id"]
        directory = output / "tasks" / identity
        directory.mkdir(parents=True)
        public = directory / "public"
        images = public / "problem_assets"
        images.mkdir(parents=True)
        for item in task["reference_images"]:
            shutil.copy2(dataset / item["path"], images / Path(item["path"]).name)
        # Private scoring inputs contain only query-derived evidence, never teacher code.
        single = {**manifest, "tasks": [task]}
        with zipfile.ZipFile(
            directory / "query-reference.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr("manifest.json", json.dumps(single, ensure_ascii=False))
            archive.writestr(
                "features.json",
                json.dumps(task["visual_features"], ensure_ascii=False),
            )
            for item in task["reference_images"]:
                archive.write(dataset / item["path"], item["path"])
        for mode in ["sft", "agent"]:
            spec = copy.deepcopy(env)
            spec["files"] = runtime_files
            if mode == "agent":
                spec["files"] = [
                    *runtime_files,
                    {"uri": "SOURCE:history32", "dest": "native/history32.pyz"},
                ]
            spec["bundle"] = {"uri": f"LOCAL:tasks/{identity}/public/"}
            spec["oracle"] = {
                "uri": f"LOCAL:tasks/{identity}/query-reference.zip",
                "media_type": "application/zip",
            }
            query = dataset / (task["sft_messages"] if mode == "sft" else task["query"])
            name = "sft-messages.json" if mode == "sft" else "TASK.md"
            shutil.copy2(query, directory / name)
            spec["query"] = {"uri": f"LOCAL:tasks/{identity}/{name}"}
            spec["hooks"] = {"setup_script": setup, "collect_script": collect}
            spec["metadata"] = {
                "source_task": identity,
                "mode": mode,
                "checkpoint": manifest["checkpoint"],
                "teacher_in_agent_input": "false",
            }
            spec["tags"] = [
                "task:blender-overfit16",
                "experiment:blender-overfit16",
                f"mode:{mode}",
            ]
            spec["workdir"] = "/envarena/workspace"
            all_envs.append(
                {"key": identity + "-" + mode, "version": "0.1.0", "spec": spec}
            )
    harness = {
        "key": "blender-overfit16-sft-one-shot",
        "version": "0.1.0",
        "spec": {
            "runtime": {"type": "runner", "uri": code_uri},
            "agent": {"delivery": "builtin", "entrypoint": ""},
            "io_contract": {
                "output_file": "$ARENA_OUTPUT_DIR/harness_result.json",
                "artifact_globs": [
                    "$ARENA_AGENT_OUTPUT_DIR/**/*",
                    "$ARENA_OUTPUT_DIR/harness_result.json",
                ],
            },
            "resource_spec": {"cpu": "2", "memory": "4Gi", "disk": "10Gi"},
            "ttl_seconds": 7200,
        },
    }
    write_json(output / "harness-blueprint.json", harness)
    write_json(output / "env-blueprints.json", all_envs)
    write_json(
        output / "stream-blueprints.json",
        [
            {
                "stream_id": "blender-overfit16-" + mode,
                "name": "Blender overfit16 " + mode,
                "task": "blender-overfit16",
                "selector": {
                    "include_tags": ["experiment:blender-overfit16", "mode:" + mode]
                },
                "default_harness_ref": {
                    "key": "blender-overfit16-sft-one-shot",
                    "version": "0.1.0",
                }
                if mode == "sft"
                else {"key": "areal-harness-rl-history32", "version": "1.0.0"},
                "default_reward_ref": {
                    "key": "blender-overfit16-visual",
                    "version": "0.1.0",
                },
            }
            for mode in ["sft", "agent"]
        ],
    )
    blender_files = [
        item
        for item in env["files"]
        if item["dest"].startswith("runtime/") and "arca-service" not in item["dest"]
    ]
    reward_setup = (
        "set -eu\nmkdir -p /tmp/dataprinciple-runtime/blender\n"
        'cd "$ARENA_REWARD_DIR/runtime"\n'
        "cat blender.tar.xz.00 blender.tar.xz.01 > /tmp/dataprinciple-runtime/blender.tar.xz\n"
        "tar -xJf /tmp/dataprinciple-runtime/blender.tar.xz --strip-components=1 -C /tmp/dataprinciple-runtime/blender\n"
        "dpkg -i libegl-mesa0.deb libegl1.deb\n"
        "export BLENDER_BIN=/tmp/dataprinciple-runtime/blender/blender\n"
        'exec python3 "$ARENA_REWARD_DIR/runtime.pyz" reward\n'
    )
    (output / "reward-entrypoint.sh").write_text(reward_setup)
    write_json(
        output / "reward-blueprint.json",
        {
            "key": "blender-overfit16-visual",
            "version": "0.1.0",
            "spec": {
                "runtime": env["runtime"],
                "entrypoint": "bash /envarena/reward/entrypoint.sh",
                "files": [
                    *blender_files,
                    {"uri": "LOCAL:blender-overfit-reward.pyz", "dest": "runtime.pyz"},
                    {"uri": "LOCAL:reward-entrypoint.sh", "dest": "entrypoint.sh"},
                ],
                "evaluation_mode": "separate",
                "normalization": ".score",
                "ttl_seconds": 7200,
                "io_contract": {"artifact_globs": ["$ARENA_REWARD_OUTPUT_DIR/**/*"]},
                "score_schema": {"normalization": "already_0_1", "sub_scores": []},
                "resource_spec": {"cpu": "4", "memory": "8Gi", "disk": "30Gi"},
            },
            "publication_requirement": "Copy eval_model from the source Reward at publication time; no credentials are stored in this blueprint",
        },
    )
    write_json(
        output / "plan.json",
        {
            "status": "LOCAL_CANDIDATE_ONLY",
            "checkpoint": manifest["checkpoint"],
            "task_count": len(manifest["tasks"]),
            "paired_task_count": 2 * len(manifest["tasks"]),
            "agent_harness": "areal-harness-rl-history32@1.0.0",
            "agent_model_environment": {"AREAL_ARENA_REASONING_EFFORT": "medium"},
            "agent_code_sha256": digest(
                (output / "blender-overfit-agent.pyz").read_bytes()
            ),
            "reward_code_sha256": digest(
                (output / "blender-overfit-reward.pyz").read_bytes()
            ),
            "local_uri_policy": "LOCAL entries are upload dependencies, not valid published Arena URIs",
            "source_env": "fine-unified115-isolated-20261023-fine-20261016-cn-add-a032@1.0.3-public-input-clean",
            "actions_taken": {
                "published": False,
                "stream_modified": False,
                "jobs_submitted": False,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-env", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.dataset, args.source_env, args.output)


if __name__ == "__main__":
    main()
