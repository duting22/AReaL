"""Operate isolated overfit evaluation releases against an explicitly selected Arena."""

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import shlex
import shutil
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any

from examples.blender_overfit.prepare import write_json
from examples.blender_overfit.replay import delivered_scene


def credentials(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        for item in shlex.split(line, comments=True):
            if "=" in item:
                key, value = item.split("=", 1)
                values[key] = value
    return values


def redact(value: Any, key: str = "") -> Any:
    if any(
        part in key.lower()
        for part in [
            "api_key",
            "token",
            "secret",
            "password",
            "cookie",
            "authorization",
        ]
    ):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {key: redact(item, key) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


class APIError(RuntimeError):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(f"Arena HTTP {status}: {message}")


class Arena:
    def __init__(self, base: str, token: str, direct: bool = False):
        self.base = base.rstrip("/")
        self.token = token
        self.opener = (
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
            if direct
            else urllib.request.build_opener()
        )

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        data: bytes | None = None,
        content_type: str | None = None,
        timeout: float = 180,
    ) -> Any:
        headers = {"Authorization": "Bearer " + self.token}
        if body is not None:
            data = json.dumps(body).encode()
            content_type = "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(
            self.base + path, data=data, headers=headers, method=method
        )
        try:
            with self.opener.open(request, timeout=timeout) as response:
                raw = response.read()
            result = json.loads(raw) if raw else {}
            return result.get("data", result) if isinstance(result, dict) else result
        except urllib.error.HTTPError as exc:
            message = (
                exc.read(6000)
                .decode(errors="replace")
                .replace(self.token, "[REDACTED]")
            )
            raise APIError(exc.code, message) from None

    def get_optional(self, path: str) -> Any | None:
        try:
            return self.request("GET", path, timeout=30)
        except APIError as exc:
            if exc.status == 404:
                return None
            raise


class Evaluation:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.run_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = (self.root / "control.lock").open("a+")
        fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX)
        values = credentials(args.credentials_file)
        token = values.get("ARENA_ONLINE_API_TOKEN") or values.get(
            "ARENA_OPENAPI_TOKEN"
        )
        if not token:
            raise ValueError("The selected credential profile has no Arena token")
        self.api = Arena(args.api_base, token, direct=args.direct_api)
        self.dataset = json.loads((args.dataset / "manifest.json").read_text())
        self.state_path = self.root / "state.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.exists()
            else {
                "api_base": self.api.base,
                "model": args.model,
                "releases": {},
                "attempts": [],
            }
        )
        if self.state["api_base"] != self.api.base or self.state["model"] != args.model:
            raise ValueError("Run state belongs to a different Arena or deployed model")

    def save(self) -> None:
        write_json(self.state_path, self.state)
        self.state_path.chmod(0o600)

    def event(self, event_name: str, **fields: Any) -> None:
        item = {
            "time": dt.datetime.now(dt.UTC).isoformat(),
            "event": event_name,
            **redact(fields),
        }
        with (self.root / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.save()

    def upload(self, materials: Path) -> str:
        archive = self.root / ("materials-" + self.args.version + ".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as target:
            for path in sorted(materials.rglob("*")):
                if path.is_file():
                    target.write(path, str(path.relative_to(materials)))
        boundary = "overfit-" + uuid.uuid4().hex
        raw = archive.read_bytes()
        payload = (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="unpack"\r\n\r\nzip\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{archive.name}"\r\n'
                "Content-Type: application/zip\r\n\r\n"
            ).encode()
            + raw
            + f"\r\n--{boundary}--\r\n".encode()
        )
        result = self.api.request(
            "POST",
            "/uploads",
            data=payload,
            content_type="multipart/form-data; boundary=" + boundary,
            timeout=300,
        )
        self.event(
            "uploaded",
            uri=result["uri"],
            sha256=hashlib.sha256(raw).hexdigest(),
            bytes=len(raw),
        )
        return result["uri"].rstrip("/") + "/"

    def resolve(self, value: Any, prefix: str) -> Any:
        if isinstance(value, str) and value.startswith("LOCAL:"):
            return prefix + value.removeprefix("LOCAL:")
        if isinstance(value, dict):
            return {key: self.resolve(item, prefix) for key, item in value.items()}
        if isinstance(value, list):
            return [self.resolve(item, prefix) for item in value]
        return value

    def publish_version(self, kind: str, key: str, spec: dict) -> dict:
        if self.api.get_optional(f"/{kind}/{key}") is None:
            self.api.request(
                "POST",
                "/" + kind,
                {"key": key, "name": key, "tags": ["experiment:blender-overfit16"]},
            )
        path = f"/{kind}/{key}/versions/{self.args.version}"
        version = self.api.get_optional(path)
        if version is None:
            self.api.request(
                "POST",
                f"/{kind}/{key}/versions",
                {"version": self.args.version, "spec": spec},
            )
            self.event("draft_created", kind=kind, key=key, version=self.args.version)
        elif version["status"] == "PUBLISHED":
            return {"key": key, "version": self.args.version}
        self.api.request("POST", path + "/publish", {}, timeout=600)
        after = self.api.request("GET", path)
        if after["status"] != "PUBLISHED":
            raise ValueError("Publication did not produce a PUBLISHED version")
        self.event(
            "published",
            kind=kind,
            key=key,
            version=self.args.version,
            content_hash=after.get("content_hash"),
        )
        return {"key": key, "version": self.args.version}

    def publish(self) -> None:
        release = self.state["releases"].setdefault(self.args.version, {})
        if "upload_prefix" not in release:
            release["upload_prefix"] = self.upload(self.args.materials)
            self.save()
        prefix = release["upload_prefix"]
        h = self.resolve(
            json.loads((self.args.materials / "harness-blueprint.json").read_text()),
            prefix,
        )
        release["harness"] = self.publish_version("harnesses", h["key"], h["spec"])
        native = self.api.request(
            "GET", "/harnesses/areal-harness-rl-history32/versions/1.0.0"
        )["spec"]
        original_history_runtime = native["runtime"]["uri"]
        if (self.args.materials / "blender-overfit-history32.py").is_file():
            native["runtime"]["uri"] = prefix + "blender-overfit-history32.py"
        native["io_contract"]["artifact_globs"].extend(
            [
                "$ARENA_AGENT_OUTPUT_DIR/preview.png",
                "$ARENA_AGENT_OUTPUT_DIR/scene-native.blend",
                "$ARENA_AGENT_OUTPUT_DIR/presentation.json",
                "$ARENA_AGENT_OUTPUT_DIR/presentation.log",
                "$ARENA_AGENT_OUTPUT_DIR/model-failure.json",
                "$ARENA_AGENT_OUTPUT_DIR/history32-input-adaptation.json",
            ]
        )
        native.setdefault("metadata", {})["overfit_adaptation"] = (
            "Original history32 native binary and settings; bind four query images; collect scenes and preview"
        )
        release["agent_harness"] = self.publish_version(
            "harnesses", "blender-overfit16-agent-history32", native
        )
        r = self.resolve(
            json.loads((self.args.materials / "reward-blueprint.json").read_text()),
            prefix,
        )
        source = self.api.request(
            "GET",
            "/rewards/fine-unified115-isolated-router-20261023/versions/1.0.3-public-input-clean-v1",
        )
        evaluation_model = source["spec"].get("eval_model")
        if not evaluation_model:
            raise ValueError("Source Reward has no private eval_model configuration")
        r["spec"]["eval_model"] = evaluation_model
        release["reward"] = self.publish_version("rewards", r["key"], r["spec"])
        envs = self.resolve(
            json.loads((self.args.materials / "env-blueprints.json").read_text()),
            prefix,
        )
        release.setdefault("envs", {})
        selected = {
            task["task_id"] for task in self.dataset["tasks"][: self.args.limit]
        }
        for env in envs:
            if env["spec"]["metadata"]["source_task"] not in selected:
                continue
            for item in env["spec"]["files"]:
                if item["uri"] == "SOURCE:history32":
                    item["uri"] = original_history_runtime
            release["envs"][env["key"]] = self.publish_version(
                "envs", env["key"], env["spec"]
            )
            self.save()
        streams = json.loads(
            (self.args.materials / "stream-blueprints.json").read_text()
        )
        release.setdefault("streams", {})
        for stream in streams:
            mode = stream["stream_id"].split("-")[-1]
            stream["default_harness_ref"] = (
                release["harness"] if mode == "sft" else release["agent_harness"]
            )
            stream["default_reward_ref"] = release["reward"]
            if self.api.get_optional("/streams/" + stream["stream_id"]) is None:
                self.api.request("POST", "/streams", stream)
                self.event("stream_created", stream_id=stream["stream_id"])
            release["streams"][mode] = stream["stream_id"]
            page = self.api.request(
                "GET", "/streams/" + stream["stream_id"] + "/dataset?limit=100"
            )
            self.event(
                "dataset_checked",
                stream_id=stream["stream_id"],
                total=page.get("total"),
                returned=len(page.get("data_ids", [])),
            )
        self.state["current_release"] = self.args.version
        self.save()

    def launch(self) -> None:
        release = self.state["releases"][self.args.version]
        gateway = (
            self.args.model_base or self.api.base.removesuffix("/openapi/v1") + "/api"
        )
        selected = self.dataset["tasks"][: self.args.limit]
        if self.args.offset:
            selected = self.dataset["tasks"][
                self.args.offset : self.args.offset + self.args.limit
            ]
        for task in selected:
            for mode in self.args.modes:
                existing = [
                    a
                    for a in self.state["attempts"]
                    if a["source_task"] == task["task_id"]
                    and a["mode"] == mode
                    and a["phase"] == self.args.phase
                    and a["version"] == self.args.version
                    and a["status"] != "SUBMISSION_REJECTED"
                ]
                if existing:
                    continue
                env = release["envs"][task["task_id"] + "-" + mode]
                job_id = (
                    "bf16"
                    + mode[0]
                    + uuid.uuid4().hex[:6]
                    + "-"
                    + dt.datetime.now(dt.UTC).strftime("%y%m%d%H%M")
                )
                record = {
                    "source_task": task["task_id"],
                    "mode": mode,
                    "phase": self.args.phase,
                    "version": self.args.version,
                    "job_id": job_id,
                    "status": "SUBMITTING",
                    "include_in_comparison": self.args.phase == "evaluation",
                }
                self.state["attempts"].append(record)
                self.save()
                body = {
                    "job_id": job_id,
                    "session_id": job_id,
                    "data_id": "env:" + env["key"] + "@" + env["version"],
                    "base_url": gateway,
                    "api_key": self.api.token,
                    "model_name": self.args.model,
                    "experiment": "blender-overfit16-" + self.args.phase,
                    "harness_ref": release["harness"]
                    if mode == "sft"
                    else release["agent_harness"],
                    "reward_ref": release["reward"],
                    "envs": {"AREAL_ARENA_REASONING_EFFORT": "medium"}
                    if mode == "agent"
                    else {},
                }
                try:
                    result = self.api.request(
                        "POST",
                        "/streams/" + release["streams"][mode] + "/launch_one_task",
                        body,
                        timeout=180,
                    )
                except APIError as exc:
                    if 400 <= exc.status < 500:
                        record.update(status="SUBMISSION_REJECTED", error=str(exc))
                        self.event("submission_rejected", **record)
                    raise
                record.update(
                    {
                        key: result[key]
                        for key in ["job_id", "task_id", "session_id", "status"]
                        if key in result
                    }
                )
                self.event("launched", **record)

    def publish_reward(self) -> None:
        """Create a grader-only release while retaining the validated generation arms."""
        if not self.args.inherit_version:
            raise ValueError("--inherit-version is required for a grader-only release")
        release = self.state["releases"].setdefault(
            self.args.version,
            json.loads(json.dumps(self.state["releases"][self.args.inherit_version])),
        )
        if release.get("inherited_from") != self.args.inherit_version:
            release["upload_prefix"] = self.upload(self.args.materials)
            release["inherited_from"] = self.args.inherit_version
            self.save()
        reward = self.resolve(
            json.loads((self.args.materials / "reward-blueprint.json").read_text()),
            release["upload_prefix"],
        )
        source = self.api.request(
            "GET",
            "/rewards/fine-unified115-isolated-router-20261023/versions/1.0.3-public-input-clean-v1",
        )
        reward["spec"]["eval_model"] = source["spec"]["eval_model"]
        release["reward"] = self.publish_version(
            "rewards", reward["key"], reward["spec"]
        )
        self.state["current_release"] = self.args.version
        self.save()

    def replay_deliveries(self) -> None:
        """Score collected scenes without calling the generation model again."""
        source_version = self.args.version
        reward = self.state["releases"][source_version]["reward"]
        stream_id = "blender-overfit16-delivery-replay"
        for record in self.state["attempts"]:
            if not record.get("include_in_comparison") or record.get(
                "evaluation_task_id"
            ):
                continue
            if record.get("replay_scene_sha256"):
                job = self.api.request(
                    "GET", "/jobs/" + record["evaluation_job_id"] + "/tasks"
                )
                if job.get("items"):
                    record["evaluation_task_id"] = job["items"][0]["task_id"]
                    self.save()
                    continue
                raise RuntimeError(
                    "Reconcile the existing replay job before resubmitting"
                )
            if record["status"] not in {
                "HARNESS_FAILED",
                "MODEL_FAILURE",
                "DELIVERED_NEEDS_EVALUATION",
            }:
                continue
            scene = delivered_scene(
                self.root / "tasks" / record["job_id"] / "artifacts"
            )
            if scene is None:
                continue
            scene_hash = hashlib.sha256(scene.read_bytes()).hexdigest()
            material = (
                self.root
                / "delivery-replays"
                / (record["source_task"] + "-" + record["mode"])
            )
            material.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(__file__).with_name("replay.py"), material / "replay.py")
            shutil.copy2(scene, material / "scene.blend")
            write_json(
                material / "source.json",
                {
                    "source_task_id": record["task_id"],
                    "scene_sha256": scene_hash,
                    "source_artifact_name": scene.name,
                    "model_calls": 0,
                },
            )
            self.args.version = "delivery-" + scene_hash[:12]
            prefix = self.upload(material)
            self.args.version = "0.1.0"
            harness = self.publish_version(
                "harnesses",
                "blender-overfit16-delivery-replay",
                {
                    "runtime": {"type": "runner", "uri": prefix + "replay.py"},
                    "agent": {"delivery": "builtin", "entrypoint": ""},
                    "ttl_seconds": 600,
                    "io_contract": {
                        "output_file": "$ARENA_OUTPUT_DIR/harness_result.json",
                        "artifact_globs": ["$ARENA_AGENT_OUTPUT_DIR/**/*"],
                    },
                },
            )
            original_ref = self.state["releases"][record["version"]]["envs"][
                record["source_task"] + "-" + record["mode"]
            ]
            original = self.api.request(
                "GET",
                "/envs/" + original_ref["key"] + "/versions/" + original_ref["version"],
            )["spec"]
            env_key = (
                record["source_task"]
                + "-"
                + record["mode"]
                + "-replay-"
                + scene_hash[:8]
            )
            spec = {
                key: original[key] for key in ("runtime", "query", "bundle", "oracle")
            }
            spec.update(
                files=[
                    {"uri": prefix + name, "dest": "evidence/" + name}
                    for name in ("scene.blend", "source.json")
                ],
                hooks={},
                timeout_seconds=3600,
                tags=[
                    "task:blender-overfit16-replay",
                    "experiment:blender-overfit16-delivery-replay",
                ],
                metadata={
                    "source_task_id": record["task_id"],
                    "scene_sha256": scene_hash,
                    "model_calls": "0",
                },
            )
            env = self.publish_version("envs", env_key, spec)
            if self.api.get_optional("/streams/" + stream_id) is None:
                self.api.request(
                    "POST",
                    "/streams",
                    {
                        "stream_id": stream_id,
                        "name": "Blender overfit recorded-delivery grading",
                        "task": "blender-overfit16-replay",
                        "selector": {
                            "include_tags": [
                                "experiment:blender-overfit16-delivery-replay"
                            ]
                        },
                        "default_harness_ref": harness,
                        "default_reward_ref": reward,
                    },
                )
            job_id = (
                "bf16r"
                + uuid.uuid4().hex[:6]
                + "-"
                + dt.datetime.now(dt.UTC).strftime("%y%m%d%H%M")
            )
            record.update(
                evaluation_job_id=job_id,
                replay_scene_sha256=scene_hash,
                delivery_source_artifact=scene.name,
                evaluation_reward_version=reward["version"],
                score=None,
            )
            self.save()
            response = self.api.request(
                "POST",
                "/streams/" + stream_id + "/launch_one_task",
                {
                    "job_id": job_id,
                    "session_id": job_id,
                    "data_id": "env:" + env["key"] + "@" + env["version"],
                    "base_url": self.args.model_base,
                    "api_key": self.api.token,
                    "model_name": self.args.model,
                    "harness_ref": harness,
                    "reward_ref": reward,
                    "experiment": "blender-overfit16-recorded-delivery",
                },
                timeout=60,
            )
            record.update(evaluation_task_id=response["task_id"], status="PENDING_EVAL")
            self.event(
                "delivery_replay_launched",
                source_task=record["task_id"],
                evaluation_task_id=response["task_id"],
                scene_sha256=scene_hash,
                model_calls=0,
            )
            print(
                "recorded scene submitted for grading",
                record["mode"],
                record["source_task"],
                flush=True,
            )
        self.args.version = source_version
        self.save()

    def status(self) -> None:
        for record in self.state["attempts"]:
            if record.get("status") in {"SUBMISSION_REJECTED", "STOPPED_UNSTARTED"}:
                continue
            if not record.get("include_in_comparison") and record.get("status") in {
                "HARNESS_FAILED",
                "MODEL_FAILURE",
                "EVAL_FAILED",
                "DONE",
            }:
                continue
            if not record.get("task_id"):
                job = self.api.get_optional("/jobs/" + record["job_id"] + "/tasks")
                items = job.get("items", []) if job else []
                if items:
                    task = items[0].get("task", items[0])
                    record["task_id"] = task["task_id"]
            if not record.get("task_id"):
                continue
            path = "/tasks/" + urllib.parse.quote(record["task_id"], safe="")
            task = self.api.request("GET", path)
            task = task.get("task", task)
            if record.get("evaluation_task_id"):
                record["source_status"] = task["status"]
                path = "/tasks/" + urllib.parse.quote(
                    record["evaluation_task_id"], safe=""
                )
                task = self.api.request("GET", path)
            record["status"] = task["status"]
            record["timings"] = task.get("timings", {})
            if record["status"] in {
                "PENDING",
                "AGENT_RUNNING",
                "EVALUATING",
                "CREATING_REWARD_SBX",
                "WAITING_REWARD_SBX_SLOT",
            }:
                if record.get("score") is not None:
                    record["previous_score"] = record["score"]
                record["score"] = None
            if record["status"] in {
                "DONE",
                "FAILED",
                "HARNESS_FAILED",
                "EVAL_FAILED",
                "SETUP_FAILED",
                "COLLECT_FAILED",
                "TIMEOUT",
                "CANCELLED",
            }:
                result = self.api.get_optional(path + "/result")
                if result:
                    record["platform_score"] = result.get("score")
                    record["score"] = (
                        result.get("score") if record["status"] == "DONE" else None
                    )
                    harness_status = (
                        result.get("raw", {}).get("harness", {}).get("result_status")
                    )
                    if (
                        record["status"] == "HARNESS_FAILED"
                        and harness_status == "MODEL_FAILURE"
                    ):
                        record.update(
                            score=0.0,
                            failure_kind="model_failure",
                            platform_status="HARNESS_FAILED",
                            status="MODEL_FAILURE",
                        )
                    outcome = result.get("raw", {}).get("outcome", {})
                    if record.get(
                        "platform_status", record["status"]
                    ) == "HARNESS_FAILED" and outcome.get("class") in {
                        "agent",
                        "model",
                    }:
                        record.update(
                            score=0.0,
                            failure_kind="model_failure",
                            outcome_code=outcome.get("code"),
                            platform_status="HARNESS_FAILED",
                            status="MODEL_FAILURE",
                        )
                        listing = self.api.request("GET", path + "/artifacts")
                        if any(
                            item.get("name") in {"scene.blend", "scene-native.blend"}
                            for item in listing.get("items", [])
                        ):
                            record.update(
                                score=None,
                                status="DELIVERED_NEEDS_EVALUATION",
                                delivery_available=True,
                            )
                    write_json(
                        self.root / "tasks" / record["job_id"] / "result.json",
                        redact(result),
                    )
            self.event("task_status", **record)
            print(
                record["mode"],
                record["source_task"],
                record["status"],
                record.get("score"),
                flush=True,
            )
        self.save()

    def download(self) -> None:
        records = []
        for record in self.state["attempts"]:
            if not record.get("include_in_comparison"):
                continue
            records.append(record)
            if record.get("evaluation_task_id"):
                records.append(
                    {
                        **record,
                        "task_id": record["evaluation_task_id"],
                        "job_id": record["evaluation_job_id"],
                    }
                )
        for record in records:
            if not record.get("task_id"):
                continue
            task_path = "/tasks/" + urllib.parse.quote(record["task_id"], safe="")
            listing = self.api.request("GET", task_path + "/artifacts")
            directory = self.root / "tasks" / record["job_id"] / "artifacts"
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory.parent / "artifacts.json", listing)
            for item in listing.get("items", []):
                name = item["name"]
                if not item.get("size_bytes") and "." not in name:
                    continue  # Directory entries are also reported by broad artifact globs.
                if Path(name).name != name or name in {".", ".."}:
                    raise ValueError("Unsafe artifact name")
                target = directory / name
                if target.exists() and (
                    not item.get("size_bytes")
                    or target.stat().st_size == item["size_bytes"]
                ):
                    continue
                path = item["uri"]
                if not path.startswith("/api/artifacts/"):
                    continue
                request = urllib.request.Request(
                    (
                        self.args.model_base.removesuffix("/api")
                        if self.args.model_base
                        else self.api.base.removesuffix("/openapi/v1")
                    )
                    + path,
                    headers={"Authorization": "Bearer " + self.api.token},
                )
                temporary = target.with_name(target.name + ".partial")
                with (
                    urllib.request.urlopen(request, timeout=180) as response,
                    temporary.open("wb") as output,
                ):
                    shutil.copyfileobj(response, output)
                temporary.replace(target)
            print(
                record["mode"],
                record["source_task"],
                "artifacts",
                len(listing.get("items", [])),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "publish",
            "publish_reward",
            "replay_deliveries",
            "launch",
            "status",
            "download",
        ],
    )
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--direct-api", action="store_true")
    parser.add_argument("--model-base")
    parser.add_argument("--credentials-file", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--materials", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--inherit-version")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--phase", default="canary")
    parser.add_argument(
        "--modes", nargs="+", choices=["sft", "agent"], default=["sft", "agent"]
    )
    args = parser.parse_args()
    operation = Evaluation(args)
    getattr(operation, args.action)()


if __name__ == "__main__":
    main()
