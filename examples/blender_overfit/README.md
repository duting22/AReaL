# Blender SFT overfit evaluation

Freeze paired evaluation inputs from the SFT **training** source. This measures fitting
on seen examples; it does not estimate held-out generalization. Both modes use one
explicit, complete checkpoint.

- **SFT**: preserve the entire original system and user messages, including the image
  bytes and order. Send one request with `enable_thinking=false`, without tools, retries
  or model-based repair. Execute the returned Python block, add missing display camera
  and lights, and save `scene.blend` plus a colored human-review preview.
- **Agent**: retain the same reference images and object description in a natural
  delivery task. Derive from `areal-harness-rl-history32@1.0.0` with thinking enabled by
  the model configuration. Its original input guard binds exactly two named building
  images; `history32.py` changes that guard to exactly the four query images. The native
  binary, image ingress, model loop, settings and history behavior remain unchanged.
  Collect the delivered scene and add missing display equipment.
- **Verifier**: use the source Stream's Q/D/A visual rubrics, adapted to the four query
  images and generic objects. Render the same neutral-material Cycles128 evidence for
  both modes. No teacher program or reference `.blend` is consulted.

The source contains Blender, CadQuery and three.js examples. Only Blender rows are
eligible. Group target programs by their AST, ignoring formatting, comments and the
factory's `os.environ.get("SEED", ...)` default. Sample unique programs without
replacement, then choose one original seed/prompt variant per selected group. This
prevents near-identical seed variants of one object template from occupying multiple
slots. Freeze IDs, source offsets, input checksums and the checkpoint in
`manifest.json`. Teacher programs are used only to define sampling groups. Completions
stay under a separate `private/` directory and are excluded from every Arena
public/reward package.

## Prepare inputs and materials

```bash
python -m examples.blender_overfit.prepare \
  --source "$SFT_SOURCE_JSONL" --checkpoint-root "$CHECKPOINT_EXPORT_ROOT" \
  --output "$EVAL_DATASET" --count 16 --seed 42

python -m examples.blender_overfit.gallery --dataset "$EVAL_DATASET"

python -m examples.blender_overfit.build \
  --dataset "$EVAL_DATASET" --source-env "$FROZEN_SOURCE_ENV_JSON" \
  --output "$EVAL_MATERIALS"
```

The builder writes separate agent/reward zipapps, 32 Env blueprints, two Stream
blueprints, a one-shot Harness blueprint and one shared Reward blueprint. `LOCAL:`
entries are explicit upload dependencies. Blueprints are **not published registry
objects**. Publishing must resolve these entries and inherit the original Reward's
private `eval_model` settings in memory. Credentials are never included in the generated
materials. Existing Streams are not changed.

`SOURCE:history32` is resolved from the published source Harness during publication. The
wrapper verifies the original runner hash before adapting its image binding, updates the
derived release binding and records both hashes in an output receipt.

The agent package excludes the verifier implementation and grading prompts. Per-task
scoring checklists stay in the reward-only archive. Public bundles contain only the four
query images.

## Local one-shot invocation

```bash
export MODEL_NAME BASE_URL API_KEY
python -m examples.blender_overfit.single_turn \
  --messages "$SFT_MESSAGES_JSON" --blender "$BLENDER_BIN" \
  --output "$NEW_OUTPUT_DIR" --render
```

Use Blender 5 as required by the SFT system message. In Arena, reuse the original Env's
offline Blender 5.2.1 archives and runtime setup. The program runner does not repair
generated code. `scene.py` adds only presentation facilities; geometry and materials
stay as generated. `scene-native.blend` preserves an agent's original delivery.

The Arena one-shot runner reads a streaming response so long code completions keep the
gateway connection active. It assembles one completion from one HTTP request; thinking
remains disabled and no tools or feedback are supplied.

## Visual scoring contract

Original weights Q=0.10, D=0.10 and A=0.05 become `0.4Q + 0.4D + 0.2A` after excluding
the geometry-dependent shape and structure channels. Those channels are explicitly
`NOT_APPLICABLE`. Unknown evidence stays null. Missing visual components do not silently
become zero or cause another weight renormalization. Infrastructure and judge errors
return an error with a null score; attributed model delivery failures return zero.

The native agent can exhaust its token budget after writing a scene. Such a delivery is
still scored. If Arena rejects reward-only runs from the failed source status,
`replay_deliveries` registers an isolated delivery replay that checks the scene SHA256
and invokes the same separate Reward with zero model calls. Source outcomes and replay
grading identities remain linked in the run state.

If display conversion fails after a native scene was collected, retain and score
`scene-native.blend` directly. Record its original artifact name and SHA256. If no
colored preview is available, the review page can show the grader's first neutral view,
explicitly labeled, while linking the unchanged native scene.

`OVERFIT_RENDER_THREADS` configures the grader's positive worker count (default 2). It
changes rendering compute only; views, Cycles samples, seed and scoring prompts remain
fixed. Grader memory and CPU allocations are recorded with each recovery.

Scene framing includes static legacy particle hair rendered as paths. Its conservative
envelope includes strand length, independent tangent/random/aligned velocities,
thickness and child offsets. World-space velocities are not shrunk with the emitter. An
emitter mesh alone does not bound its hairs. Edited, dynamic, animated or forced hair is
rejected explicitly. These corrections affect only the agent fox and hairy coconut in
the frozen cohort; affected evidence is regraded. Revisions 0.1.13/0.1.14 keep the
visual judge, rubrics and weights unchanged. Blender exposes
[`hair_length` as four times the normal velocity](https://github.com/blender/blender/blob/blender-v5.2-release/source/blender/makesrna/intern/rna_particle.cc),
while its
[particle initializer adds tangent and random velocities independently](https://github.com/blender/blender/blob/blender-v5.2-release/source/blender/blenkernel/intern/particle_system.cc).

`OVERFIT_SAVE_DISPLAY_SCENE=1` exports an additional colored `display-scene.blend` after
all scoring images have been rendered. Its camera frames the complete candidate; the
original scored scene remains unchanged and separately downloadable. The review page
labels neutral scoring previews and links both files when available.

For rendering diagnostics, `OVERFIT_RENDER_ONLY=1` accepts fixed view indices and the
expected candidate SHA256 in a private configuration at `ARENA_QUERY_PATH`. It makes
zero judge calls and returns a diagnostic result, which is never a formal visual score.
Reward sandboxes do not mount the Env's query or public files; supply that configuration
as a Reward file and set its path explicitly.

The pinned native Core retains its internal 1 MiB reasoning-text guard. Preserve its
original error receipt and all existing model resource limits. Under the confirmed
experiment policy, explicit budget exhaustion without a delivered scene counts as zero
in the primary result. Also report a second paired result that removes the same question
from **both** modes whenever either mode exhausts a budget without delivery. Ordinary
generated-code failures score zero and do not independently exclude a pair. Delivered
scenes retain their visual scores even if generation ends at a budget limit. Grader or
infrastructure failures stay null until recovered; a renderer timeout after delivery is
not a model budget failure.

Q/A use the original workmanship and geometric detail rubric. D retains its original
0/25/50/75/100 anchors, using a candidate-blind checklist frozen from the original query
description. All four query images have equal evidence status. The nine whole-object
views and two fixed center crops are shared by Q/A and D. The same deterministic
candidate renderer is used for both modes; original building-specific role labels,
alignment and private geometry are absent.

Arena mounts the oracle as a directory containing the reference ZIP. The adapter accepts
that mount, a ZIP file, or an already extracted reference directory. Only the four query
images, description and frozen checklist are present in that archive.

## Operate an isolated Arena evaluation

`control.py` accepts an explicit API base and credential file. It uploads materials,
publishes new immutable versions, creates isolated Streams, launches paired tasks and
downloads artifacts. It preserves submission IDs before network calls so an uncertain
response can be reconciled without duplicate model sampling. Published source Streams
are never modified. Select a fresh release version whenever runtime materials change.

```bash
python -m examples.blender_overfit.control publish \
  --api-base "$EVALUATION_API_BASE" --credentials-file "$ARENA_CREDENTIAL_FILE" \
  --dataset "$EVAL_DATASET" --materials "$EVAL_MATERIALS" --run-dir "$EVAL_RUN" \
  --model "$DEPLOYED_MODEL" --version "$EVAL_VERSION" --limit 1
```

Use the same arguments with `launch` for a canary. `--modes sft` or `--modes agent`
selects one arm, and `--offset`/`--limit` selects a task slice. After validation,
publish all 16 and launch the remaining tasks. `status` records platform failures
separately from valid model scores; `download` retains scenes, previews, trajectories
and grader evidence. Failed infrastructure attempts do not become zero-valued model
evaluations, apart from the explicitly approved budget policy above.

The exact source artifact, prompt hashes, adaptations and judge identity are recorded in
[prompts/provenance.json](prompts/provenance.json). The recovered request receipts do
not expose sampling parameters. The adapter sends no sampling overrides; equivalence of
provider defaults still needs confirmation before treating it as a replication of the
original visual runtime. The candidate renderer and query-derived checklist are explicit
adaptations, not unchanged upstream implementations.

The live adapter requests a JSON object response and saves the complete judge response
before parsing. This protocol constraint was added after the judge returned malformed
JSON during integration; it does not change rubric text or score weights.

## Validation and remaining runtime checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest -q \
  tests/test_blender_overfit.py
ruff check examples/blender_overfit tests/test_blender_overfit.py
ruff format --check examples/blender_overfit tests/test_blender_overfit.py
```

The CPU tests check prompt fidelity, image integrity, object deduplication, complete
checkpoint selection, one-shot request semantics, Session header isolation, scoring
weights and failure handling. Real Blender rendering, the published native Harness,
gateway thinking behavior, source judge sampling defaults and a paired live canary must
be checked in the target runtime. No cluster job is submitted by these preparation
commands.
