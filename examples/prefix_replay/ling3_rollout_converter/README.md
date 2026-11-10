# Ling3 rollout converter

This utility reverses a rendered Bailing/Ling3 V3 rollout into the nested OpenAI
conversation layout consumed by prefix replay:

```json
{
  "instance_id": "107",
  "conversations": [
    {
      "messages": [],
      "tools": [],
      "model": "ling-v3-rl"
    }
  ]
}
```

It reads the selected model's `chat_template.jinja` and rejects templates that do not
match the supported Ling3 format. Reasoning is stored in `assistant.reasoning_content`;
tool arguments become OpenAI JSON strings. Since the rendered template does not preserve
tool-call IDs, deterministic IDs are generated and paired with tool responses in order.
Rollout metadata such as `task_id`, `sample_idx`, `reward`, and sequence lengths is
retained at the top level; the large rendered `prompt`, `completion`, and `segments`
fields are omitted.

If a rollout has no `instance_id`, the converter falls back to its numeric `task_id`.
For a per-task rollout file, pass `--instance-id` when the original SWE instance ID is
known; the override is applied to every sample in that file.

Run from the repository root:

```bash
python -m examples.prefix_replay.ling3_rollout_converter.convert \
  --input /path/to/rollout.jsonl \
  --output /path/to/openai_format.jsonl \
  --model-path /path/to/ling-v3-rl \
  --max-tokens 262143 \
  --instance-id owner_repo_pr123
```

By default each converted transcript is rendered again. Canonical generations are
compared byte-for-byte with `prompt + completion`; non-canonical generations are parsed
again and compared semantically. This catches silent parser drift while making every
normalization visible in the output. Existing output is protected unless `--overwrite`
is explicitly passed.

Every converted record contains these audit fields:

```json
{
  "conversion_status": "anomalous",
  "conversion_tags": ["unterminated_tool_call"],
  "conversion_anomalies": [
    {
      "code": "unterminated_tool_call",
      "severity": "error",
      "recovery": "preserved_as_content"
    }
  ]
}
```

Important anomaly codes include:

- `reasoning_terminated_by_tool_call`: missing `</think>`; the Ling3 reasoning parser
  uses `<tool_call>` as the terminator.
- `unterminated_reasoning_to_content`: reasoning has neither `</think>` nor a tool call;
  Ling3's non-streaming fallback exposes it as assistant content.
- `reasoning_only_to_content`: even a closed reasoning-only response is exposed as
  content by Ling3's non-streaming `force_nonempty_content` fallback.
- `unterminated_tool_call`: a dangling tool call is preserved as content if no complete
  valid call exists; after a valid call it is discarded, matching SGLang's non-streaming
  tool parser. The `recovery` field distinguishes these cases.
- `malformed_tool_arguments`, `duplicate_tool_argument`, and
  `tool_argument_type_mismatch`: complete argument pairs follow SGLang's fallback
  parsing, duplicate keys use the last value, and ignored or schema-incompatible values
  are explicitly marked.
- `tool_calls_without_responses` and `tool_responses_without_calls`: the rendered
  history has a call/response count mismatch. SGLang's generation parser does not
  resolve this historical ambiguity; the converter pairs the available values in order
  and marks the record.
- `noncanonical_roundtrip`: parsing normalized the source bytes, although the normalized
  transcript is semantically stable when parsed again.

## Batch conversion

The batch entry point accepts directories or repeated `--input` paths, discovers JSONL
files recursively, and merges their converted records into one JSONL. It uses worker
processes per input file while writing results in deterministic natural path order:

```bash
python -m examples.prefix_replay.ling3_rollout_converter.batch \
  --input /path/to/experiment/rollout \
  --output /path/to/openai_format.jsonl \
  --report /path/to/conversion_report.json \
  --model-path /storage/openpsi/models/ling-v3-rl \
  --max-tokens 262143 \
  --workers 16
```

`--workers 0` selects the available CPU count capped at 16, and `--workers 1` runs
serially. Conversion is strict and atomic by default: any invalid record prevents the
merged output from being published. `--skip-errors` instead records failures in
`--report` and keeps valid records. Successfully converted anomalous records are not
silently counted as clean: the report includes `records_with_anomalies`, aggregated
occurrence-level `anomaly_counts`, per-tag `anomaly_record_counts`, and an
`anomalous_records` list with source file and line number.

The rendered chat template does not contain the original SWE instance ID. Without an
`instance_id` field in each rollout row, batch conversion therefore falls back to
`task_id`. The fixed `--instance-id` override is intentionally allowed only for a single
input file.
