"""Freeze paired prompts from the actual, image-bearing SFT source JSONL."""

import argparse
import ast
import base64
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def extract_code(completion: str) -> str:
    """Accept precisely the single Python block requested by the SFT system."""
    match = re.fullmatch(r"\s*```python\s*\n(.*?)\n```\s*", completion, re.DOTALL)
    if not match:
        raise ValueError("Expected exactly one Python code block")
    code = match[1] + "\n"
    compile(code, "generated.py", "exec")
    return code


def prompt_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record["messages"]
    roles = [message["role"] for message in messages]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(f"Expected one SFT turn, got {roles}")
    if not isinstance(messages[-1]["content"], str):
        raise ValueError("Expected a text assistant completion")
    return messages[:-1]


def query_images(messages: list[dict[str, Any]]) -> list[bytes]:
    images = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image_url":
                continue
            url = part["image_url"]["url"]
            if not url.startswith("data:image/") or ";base64," not in url:
                raise ValueError("Expected embedded query images, not external URLs")
            images.append(base64.b64decode(url.split(";base64,", 1)[1], validate=True))
    if not images:
        raise ValueError("No reference images in the query")
    return images


def object_description(messages: list[dict[str, Any]]) -> str:
    texts = []
    for message in messages:
        if message["role"] != "user":
            continue
        content = message["content"]
        if isinstance(content, str):
            texts.append(content)
        else:
            texts.extend(part["text"] for part in content if part["type"] == "text")
    text = "\n".join(texts)
    marker = "The object: "
    if text.count(marker) != 1:
        raise ValueError("Cannot identify the unchanged SFT object description")
    return text.split(marker, 1)[1].strip()


def code_features(completion: str) -> dict[str, bool]:
    return {
        "camera": bool(
            re.search(r"bpy\.data\.cameras|bpy\.ops\.object\.camera_add", completion)
        ),
        "light": bool(
            re.search(r"bpy\.data\.lights|bpy\.ops\.object\.light_add", completion)
        ),
        "save_blend": "save_as_mainfile" in completion,
        "render": "bpy.ops.render.render" in completion,
    }


def generator_identity(completion: str) -> str:
    """Ignore formatting/comments and the factory's SEED environment default only."""
    tree = ast.parse(extract_code(completion))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "environ"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "SEED"
        ):
            node.args[1] = ast.Constant(value="<factory-seed>")
    return digest(ast.dump(tree, include_attributes=False).encode())


def detail_features(description: str) -> list[dict[str, Any]]:
    """Freeze candidate-blind checklist entries from the original query description."""
    sentences = [
        text.strip() for text in re.split(r"(?<=[.!?])\s+", description) if text.strip()
    ]
    return [
        {
            "id": f"query-detail-{index:02d}",
            "name": sentence,
            "reference_evidence": "Only geometry clearly visible in the four original query images",
            "regions": ["Visible corresponding regions across P1, P2, P3 and P4"],
            "good": "Visible fine geometry matches the reference-supported detail",
            "coarse": "Only a crude, partial or symbolic geometric representation",
            "absent": "Reference-supported visible geometry is missing or fundamentally wrong",
        }
        for index, sentence in enumerate(sentences, 1)
    ]


def checkpoint_complete(path: Path) -> bool:
    """A partially written index or tensor header is an incomplete export."""
    try:
        index_path = path / "model.safetensors.index.json"
        if not index_path.is_file() or not (path / "config.json").is_file():
            return False
        json.loads((path / "config.json").read_text())
        index = json.loads(index_path.read_text())
        if not index["weight_map"]:
            return False
        for filename in set(index["weight_map"].values()):
            shard = path / filename
            if Path(filename).name != filename or not shard.is_file():
                return False
            with shard.open("rb") as handle:
                header_size = int.from_bytes(handle.read(8), "little")
                if header_size <= 0 or header_size > 20_000_000:
                    return False
                header = json.loads(handle.read(header_size))
            end = max(
                tensor["data_offsets"][1]
                for name, tensor in header.items()
                if name != "__metadata__"
            )
            if shard.stat().st_size != 8 + header_size + end:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def latest_checkpoint(root: Path) -> Path:
    """Resolve the latest fully written HF export, never a partial checkpoint."""
    candidates = []
    for path in root.iterdir():
        match = re.fullmatch(r"epoch\d+epochstep\d+globalstep(\d+)", path.name)
        if match and path.is_dir() and checkpoint_complete(path):
            candidates.append((int(match[1]), path))
    if not candidates:
        raise ValueError(f"No complete checkpoint in {root}")
    return max(candidates, key=lambda pair: pair[0])[1].resolve()


def prepare(
    source: Path, checkpoint_root: Path, output: Path, seed: int, count: int
) -> dict:
    """Randomize generator identities, then choose an original seed/prompt variant."""
    if count <= 0:
        raise ValueError("count must be positive")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a frozen evaluation: {output}")
    checkpoint = latest_checkpoint(checkpoint_root)
    groups: dict[str, list[tuple[int, int]]] = {}
    image_groups: set[str] = set()
    features: Counter = Counter()
    source_engines: Counter = Counter()
    source_hash = hashlib.sha256()
    rows = 0
    eligible_rows = 0
    with source.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            source_hash.update(raw)
            record = json.loads(raw)
            row_index = rows
            rows += 1
            system = record["messages"][0]["content"]
            engine = (
                "blender"
                if "Blender" in system
                else "cadquery"
                if "CadQuery" in system
                else "threejs"
                if "three.js" in system
                else "other"
            )
            source_engines[engine] += 1
            if engine != "blender":
                continue
            messages = prompt_messages(record)
            images = query_images(messages)
            image_groups.add(
                digest(b"".join(bytes.fromhex(digest(image)) for image in images))
            )
            key = generator_identity(record["messages"][-1]["content"])
            groups.setdefault(key, []).append((row_index, offset))
            features.update(
                name
                for name, present in code_features(
                    record["messages"][-1]["content"]
                ).items()
                if present
            )
            eligible_rows += 1
    if len(groups) < count:
        raise ValueError(
            f"Only {len(groups)} distinct generator programs; requested {count}"
        )
    rng = random.Random(seed)
    selected = [
        (key, rng.choice(groups[key])) for key in rng.sample(sorted(groups), count)
    ]
    output.mkdir(parents=True)
    entries = []
    with source.open("rb") as handle:
        for rank, (key, (row_index, offset)) in enumerate(selected, 1):
            handle.seek(offset)
            record = json.loads(handle.readline())
            messages = prompt_messages(record)
            images = query_images(messages)
            image_key = digest(
                b"".join(bytes.fromhex(digest(image)) for image in images)
            )
            task_id = f"overfit-{rank:02d}-{image_key[:12]}"
            task = output / "tasks" / task_id
            task.mkdir(parents=True)
            write_json(task / "sft-messages.json", messages)
            images = query_images(messages)
            references = []
            for index, data in enumerate(images, 1):
                extension = "png" if data.startswith(b"\x89PNG\r\n\x1a\n") else "jpg"
                path = task / "images" / f"reference-{index:02d}.{extension}"
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(data)
                references.append(
                    {"path": str(path.relative_to(output)), "sha256": digest(data)}
                )
            description = object_description(messages)
            query = (
                "Create a 3D model of the object shown in the attached reference images.\n\n"
                + description
                + "\n\nDeliver the completed Blender scene as scene.blend in the current working directory.\n"
            )
            (task / "TASK.md").write_text(query)
            agent_messages = [
                {
                    "role": "user",
                    "content": [
                        *[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/"
                                    + (
                                        "png"
                                        if image.startswith(b"\x89PNG")
                                        else "jpeg"
                                    )
                                    + ";base64,"
                                    + base64.b64encode(image).decode()
                                },
                            }
                            for image in images
                        ],
                        {"type": "text", "text": query},
                    ],
                }
            ]
            write_json(task / "agent-messages.json", agent_messages)
            # Teacher code is kept outside every agent/public input directory.
            oracle = output / "private" / task_id
            oracle.mkdir(parents=True)
            (oracle / "teacher-response.txt").write_text(
                record["messages"][-1]["content"]
            )
            entry = {
                "task_id": task_id,
                "source_row_index": row_index,
                "source_byte_offset": offset,
                "image_group_sha256": image_key,
                "generator_sha256": key,
                "source_variants_in_group": len(groups[key]),
                "sft_messages": str((task / "sft-messages.json").relative_to(output)),
                "sft_messages_sha256": digest(
                    (task / "sft-messages.json").read_bytes()
                ),
                "agent_messages": str(
                    (task / "agent-messages.json").relative_to(output)
                ),
                "agent_messages_sha256": digest(
                    (task / "agent-messages.json").read_bytes()
                ),
                "query": str((task / "TASK.md").relative_to(output)),
                "query_sha256": digest((task / "TASK.md").read_bytes()),
                "description": description,
                "visual_features": detail_features(description),
                "reference_images": references,
                "teacher_code_features": code_features(
                    record["messages"][-1]["content"]
                ),
            }
            write_json(task / "task.json", entry)
            entries.append(entry)
    manifest = {
        "schema": "blender-overfit-paired/v1",
        "purpose": "Evaluate fitting on SFT-seen examples; no held-out claim",
        "source": str(source.resolve()),
        "source_sha256": source_hash.hexdigest(),
        "source_rows": rows,
        "source_engine_counts": dict(source_engines),
        "eligible_engine": "blender",
        "eligible_rows": eligible_rows,
        "distinct_reference_image_groups": len(image_groups),
        "distinct_generator_programs": len(groups),
        "source_code_feature_counts": {
            name: features[name] for name in ["camera", "light", "save_blend", "render"]
        },
        "code_feature_method": "Text indicators of bpy camera/light creation and save/render calls",
        "sampling": {
            "seed": seed,
            "count": count,
            "unit": "Unique generator programs, sampled without replacement; AST ignores formatting/comments and factory SEED defaults",
            "variant": "Uniform random original SFT row within each selected group",
        },
        "checkpoint": str(checkpoint),
        "checkpoint_index_sha256": digest(
            (checkpoint / "model.safetensors.index.json").read_bytes()
        ),
        "modes": {
            "sft": {"enable_thinking": False, "model_calls": 1, "tools": False},
            "agent": {
                "enable_thinking": True,
                "harness": "areal-harness-rl-history32@1.0.0",
            },
        },
        "tasks": entries,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    prepare(args.source, args.checkpoint_root, args.output, args.seed, args.count)


if __name__ == "__main__":
    main()
