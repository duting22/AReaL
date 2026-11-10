# Qwen rollout converter

Recover text-only Qwen rollouts using ChatML roles and XML
`<function=...><parameter=...>` tool calls into the nested OpenAI conversation format
accepted by `areal/dataset/swe_sft.py`.

```bash
python -m examples.prefix_replay.qwen_rollout_converter.convert \
  --input /path/to/rollout/19 \
  --output-dir /path/to/new-output-directory \
  --model-path /path/to/Qwen-model
```

The output directory must not already exist. Outputs:

- `all.jsonl`: all successfully parsed records, including flagged anomalies.
- `clean.jsonl`: SFT candidates with no error-level conversion findings. These can
  contain warning/info tags for normalization or a final unanswered tool call.
- `anomalous.jsonl`: records requiring inspection before training.
- `report.json`: counts, anomaly totals, failures, and template checksum.

Each record contains `conversations: [{messages, tools, model}]`, with thinking stored
in `reasoning_content` and tool arguments serialized as JSON strings. Original reward,
version, task, and session metadata are retained. `instance_id` falls back to
`arena_task_id`, then `task_id`; this does not reconstruct an absent SWE instance ID.
Source file, line, and a transcript checksum are included.

The converter checks that `segments` matches `prompt + completion`, renders recovered
messages through the selected model's actual template, and checks semantic stability by
parsing the rendered result. `conversion_roundtrip_exact` records byte equality apart
from trailing newlines. A clean record need not have identical whitespace: generated
calls can differ from the canonical template in blank lines. Clean means no detected
structural error, not task success or independently verified trajectory quality.

Original tool-call IDs are absent from the template. New deterministic IDs are paired
with responses in order. Missing responses, malformed calls, unknown tools, duplicate
parameters, invalid typed arguments, truncated thinking/messages, and missing visual
payloads are flagged. Text dumps cannot recover image bytes. Original inputs are never
modified. Source files remain necessary for forensic reconstruction of malformed XML;
normalized anomalous records may include last-value-wins duplicate arguments.

Terminal assistant tool calls do not require a response for SFT and carry only an `info`
tag (`terminal_tool_calls_without_responses`). Missing responses before a later
assistant turn remain errors. The experiment's Qwen reasoning parser accepts
`<tool_call>` as an implicit reasoning terminator when `</think>` is absent; this is
recovered with a `reasoning_terminated_by_tool_call` warning. Redundant closing
`</function>` tags carry a `redundant_function_close` warning. Both normalization
warnings retain the original assistant text or tool-call block in the audit fields.
Neither normalization is byte-for-byte preservation of the model's malformed syntax.

Unterminated thinking without a tool call remains in `reasoning_content` and is marked
as an error requiring review; it is never moved into the assistant answer. Arguments
without a schema remain strings, matching the rollout proxy, instead of guessing their
types from JSON-looking text. Unknown tools remain flagged for review.

Like SGLang's non-streaming Qwen3-Coder fallback, an incomplete function/parameter can
be recovered up to EOF. `incomplete_tool_call_recovered` retains the original assistant
text and remains an error: parsing a partial argument does not establish that the
generated value is complete.

The reference for the September 5 experiment is the startup-log-confirmed SGLang
`0.5.18.dev10+g85b539146` Python override, not just the package baked into the
`areal-dev-sglang-0821.sif` image. The conversion reference is **SGLang only**:
`Qwen3Detector` for reasoning and `Qwen3CoderDetector` for tools, with
`FunctionCallParser` preserving content when no calls are parsed. AReaL's XML
compatibility layer is not part of the reference contract.

The converter is an independent implementation, not a direct SGLang import. Full-corpus
audits execute the reference source methods and compare reasoning, content, tool names
and decoded arguments, excluding generated IDs and boundary whitespace. The pure SGLang
audit for all 17,107 assistant messages in rollout directories 17–21 found no
differences from v3. Source SHA-256 values are recorded in the audit report; this
validates those source revisions and the inspected corpus, not arbitrary future parser
revisions.

Conversion/audit tags are separate from parser semantics. `all.jsonl` includes every
successfully converted record. The `clean.jsonl`/`anomalous.jsonl` partition is an
additional review aid, not an SGLang accept/reject decision; unknown tools and
incomplete arguments can still have valid parser outputs.

Failed records are listed in the report and cause a nonzero exit; successful records
remain available. This command processes files sequentially and does not filter by
reward.
