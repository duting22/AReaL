#!/bin/bash
# Flash v3 SWE RL: Megatron + SGLang colocated AWEX run (8 nodes).
# One sbatch + one yaml (v3_swe_rl_grpo_awex_slurm.yaml), REF-aligned formal
# parameters baked in as defaults; every V3_* var below can be overridden
# from the environment at submit time.
#
# 正式长跑:
#   sbatch /storage/openpsi/users/chucai.dzq/projects/v3/sbatch_flash_v3.sh
# 冒烟验证(改代码/升 fork 后, ~40min; 验收: fork-contract 通过、accepted
# 满额、AWEX-SENTINEL 分片 norm 8 组分布、Train step 2 done):
#   V3_TOTAL_TRAIN_STEPS=2 V3_BATCH_SIZE=8 V3_N_SAMPLES=2 \
#   V3_MAX_NEW_TOKENS=4096 V3_MAX_TOKENS=32768 V3_CONTEXT_LENGTH=32768 \
#   V3_MAX_CONCURRENT_ROLLOUTS=8 \
#   V3_SGLANG_MAX_RUNNING_REQUESTS=8 V3_SHOULD_ACCEPT_FN=null \
#SBATCH -J sxj-ling-v3-flash-rl
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# The controller step below requests 100000M; actor workers have their own
# per-worker memory request in the YAML (64G).
#SBATCH --mem=100G
#SBATCH --time=168:00:00
#SBATCH --oversubscribe
#SBATCH --gres=gpu:0
#SBATCH -o /storage/openpsi/experiments/logs/ling-v3-flash-swe-rl-%j.out
#SBATCH -e /storage/openpsi/experiments/logs/ling-v3-flash-swe-rl-%j.err

set -euo pipefail

# AWEX weight-transfer sentinels: opt-in probe, on for validation runs.
export AREAL_AWEX_SENTINEL=${V3_AWEX_SENTINEL:-1}
# tiny 功能验证请用 sbatch_tiny_v3.sh。

# Slurm executes a batch script from its spool copy, so BASH_SOURCE points at
# /var/spool/slurmd/... rather than the submission directory. Prefer the
# original submit directory for files that live beside this script.
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
export AREAL_DIR=${AREAL_DIR:-/storage/openpsi/users/shenxujie.sxj/AReaL_rl}
IMAGE=${IMAGE:-/storage/openpsi/images/areal-dev-20260508.sif}
CONFIG_PATH=${CONFIG_PATH:-${SCRIPT_DIR}/examples/swe/bailing_v3/v3_swe_rl_grpo_awex_slurm.yaml}

MODEL_PATH=${MODEL_PATH:-/storage/openpsi/experiments/checkpoints/admin/sxj-swe-sft/0720_lingv3_flash_bs2560_g64_lr1.5e-4_midtrain_base_all_swe_data_train_128k/default/epoch2epochstep91globalstep275}
# TRAIN_DATA=${TRAIN_DATA:-/storage/openpsi/users/miumiu.zwh/dataset_miumiu/rl_mix_scaleswe_goldpass_pass4_4k_20260710_nonperfect_0to7.jsonl}
TRAIN_DATA=${TRAIN_DATA:-/storage/openpsi/users/shenxujie.sxj/datasets/swe_issues_data/rl/zwh_swe_cc_colocation_scale_swe_pass1to6_dedup_emptyclean_20260717.jsonl}
AGENT_TYPE=${AGENT_TYPE:-cc}
AGENT_CONFIG=${AGENT_CONFIG:-1_0_0/cc_sxj}
AGENT_WEIGHT=${AGENT_WEIGHT:-1.0}
EXP_NAME=${EXP_NAME:-sxj-ling-v3-flash-swe-rl}
FILEROOT=${FILEROOT:-/storage/openpsi/experiments}
TRIAL_NAME=${TRIAL_NAME:-$(date +%m%d)-ling-v3-flash-rl-16n}
# TRIAL_NAME=ling-v3-flash-rl-16n-0724-1828
N_NODES=${N_NODES:-16}
N_GPUS=${N_GPUS:-8}

WANDB_API_KEY="${WANDB_API_KEY:-local-3bca3d5f00a980f3075b3e8ff2e16adc4ef43ffe}"
WANDB_BASE_URL=http://8.150.1.98:8080
AENV_SYSTEM_URL=${AENV_SYSTEM_URL:-http://33.180.184.68}
AWEAGENT_ROOT=${AWEAGENT_ROOT:-${SWE_AGENT_ROOT:-/storage/openpsi/users/shenxujie.sxj/scaleswe_k8s}}
SWE_AGENT_ROOT=${SWE_AGENT_ROOT:-${AWEAGENT_ROOT}}
AWEX_ROOT=${AWEX_ROOT:-/storage/openpsi/users/chucai.dzq/codes/ant-code/asystem-awex}
THETA_SGLANG=${THETA_SGLANG:-/storage/openpsi/users/chucai.dzq/codes/v3/SGLang-rel-ling-v3-20260629/python}
SGLANG_V3_WORK=${SGLANG_V3_WORK:-/storage/openpsi/users/chucai.dzq/pyuser/sglang-v3}
CACHE_ROOT=${CACHE_ROOT:-${FILEROOT}/cache/${EXP_NAME}}
# Optional local checkout; keep unset-safe because this script uses `set -u`.
# run_rl.sh uses `:-` and otherwise falls back to the public host checkout.
# Use a non-existent sentinel so the container-provided FLA is used instead.
FLASH_LINEAR_ATTENTION="${FLASH_LINEAR_ATTENTION:-/__image_builtin_fla__}"
FLASH_LINEAR_ATTENTION_ROOT="${FLASH_LINEAR_ATTENTION_ROOT:-}"


V3_TOTAL_TRAIN_EPOCHS=${V3_TOTAL_TRAIN_EPOCHS:-1000}
V3_TOTAL_TRAIN_STEPS=${V3_TOTAL_TRAIN_STEPS:-5000}
V3_ALLOCATION_MODE=${V3_ALLOCATION_MODE:-${ALLOCATION_MODE:-"megatron[actor]:(attn:d4p2t2c8|ffn:d8p2e8)|sglang[rollout]:d32t4p1"}}
# Keep smoke runs small so weight transfer is reached quickly. Override these to
# 8/8 for experiment-aligned runs.
V3_BATCH_SIZE=${V3_BATCH_SIZE:-64}
V3_N_SAMPLES=${V3_N_SAMPLES:-8}
V3_MAX_NEW_TOKENS=${V3_MAX_NEW_TOKENS:-199999}
V3_MAX_TOKENS=${V3_MAX_TOKENS:-199999}
V3_CONTEXT_LENGTH=${V3_CONTEXT_LENGTH:-200000}
V3_MAX_CONCURRENT_ROLLOUTS=${V3_MAX_CONCURRENT_ROLLOUTS:-1500}
V3_SGLANG_MAX_RUNNING_REQUESTS=${V3_SGLANG_MAX_RUNNING_REQUESTS:-128}
V3_SGLANG_MAX_PREFILL_TOKENS=${V3_SGLANG_MAX_PREFILL_TOKENS:-8192}
V3_SGLANG_CHUNKED_PREFILL_SIZE=${V3_SGLANG_CHUNKED_PREFILL_SIZE:-8192}
V3_SGLANG_MEM_FRACTION_STATIC=${V3_SGLANG_MEM_FRACTION_STATIC:-0.65}
V3_SGLANG_ATTENTION_BACKEND=${V3_SGLANG_ATTENTION_BACKEND:-fa3}
V3_SGLANG_DISABLE_CUDA_GRAPH=${V3_SGLANG_DISABLE_CUDA_GRAPH:-false}
V3_SGLANG_ENABLE_MEMORY_SAVER=${V3_SGLANG_ENABLE_MEMORY_SAVER:-true}
V3_ACTOR_MAX_TOKENS_PER_MB=${V3_ACTOR_MAX_TOKENS_PER_MB:-200000}
V3_ACTOR_WORKER_MEM_GB=${V3_ACTOR_WORKER_MEM_GB:-64}
V3_PAUSE_GRACE_PERIOD=${V3_PAUSE_GRACE_PERIOD:-60}
V3_WANDB_MODE=${V3_WANDB_MODE:-online}
V3_SAVER_MODE=${V3_SAVER_MODE:-auto}
V3_SAVER_FREQ_STEPS=${V3_SAVER_FREQ_STEPS:-10}
V3_RECOVER_MODE=${V3_RECOVER_MODE:-auto}
V3_RECOVER_FREQ_STEPS=${V3_RECOVER_FREQ_STEPS:-10}
V3_RECOVER_NO_SAVE_OPTIM=${V3_RECOVER_NO_SAVE_OPTIM:-false}
V3_RECOVER_NO_LOAD_OPTIM=${V3_RECOVER_NO_LOAD_OPTIM:-false}
V3_SHOULD_ACCEPT_FN=${V3_SHOULD_ACCEPT_FN:-areal.examples.filter_function.filter_function}
V3_DETERMINISTIC_ALGOS=${V3_DETERMINISTIC_ALGOS:-false}
V3_WARMUP_PROP=${V3_WARMUP_PROP:-0.0006544}
V3_DUMP_ROLLOUT_DATA=${V3_DUMP_ROLLOUT_DATA:-false}
V3_ECONFIG_TIMEOUT=${V3_ECONFIG_TIMEOUT:-7500}
V3_SESSION_TIMEOUT=${V3_SESSION_TIMEOUT:-7500}
V3_REQUEST_TIMEOUT=${V3_REQUEST_TIMEOUT:-20000}
# Thinking switches for the adaptive-thinking Ling v3 chat template. Both
# switches are explicitly "on" so every rollout group uses thinking; there is
# no per-group fusion/random on-off sampling.
V3_THINKING_OPTION=${V3_THINKING_OPTION:-on}
V3_THINK_MODE=${V3_THINK_MODE:-on}
V3_THINK_PROB=${V3_THINK_PROB:-1} # TODO:?
V3_EXTRA_OVERRIDES=${V3_EXTRA_OVERRIDES:-}
AWEX_WEIGHT_TRANSFER_VERSION=${AWEX_WEIGHT_TRANSFER_VERSION:-1}
AREAL_AWEX_PROCESS_QUEUE_WHEN_IDLE=${AREAL_AWEX_PROCESS_QUEUE_WHEN_IDLE:-1}
CC_MAX_OUTPUT_TOKENS=${CC_MAX_OUTPUT_TOKENS:-200000}
CC_MAX_THINKING_TOKENS=${CC_MAX_THINKING_TOKENS:-199999}
CC_DISABLE_EXPERIMENTAL_BETAS=${CC_DISABLE_EXPERIMENTAL_BETAS:-1}
CLAUDE_CODE_MAX_OUTPUT_TOKENS=${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-${CC_MAX_OUTPUT_TOKENS}}
CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=${CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING:-0}
MAX_THINKING_TOKENS=${MAX_THINKING_TOKENS:-199999}
CLAUDE_CODE_ATTRIBUTION_HEADER=${CLAUDE_CODE_ATTRIBUTION_HEADER:-0}
CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=${CLAUDE_AUTOCOMPACT_PCT_OVERRIDE:-100}
CLAUDE_CODE_AUTO_COMPACT_WINDOW=${CLAUDE_CODE_AUTO_COMPACT_WINDOW:-200000}

mkdir -p "${FILEROOT}/logs/${EXP_NAME}"

if [ "${V3_WANDB_MODE}" = "online" ]; then
  if [ -z "${WANDB_API_KEY}" ] || [ -z "${WANDB_BASE_URL}" ]; then
    echo "WANDB_API_KEY and WANDB_BASE_URL must be provided for online wandb mode." >&2
    exit 2
  fi
fi

V3_OVERRIDES=(
  total_train_epochs=${V3_TOTAL_TRAIN_EPOCHS}
  total_train_steps=${V3_TOTAL_TRAIN_STEPS}
  train_dataset.batch_size=${V3_BATCH_SIZE}
  valid_dataset.batch_size=${V3_BATCH_SIZE}
  train_dataset.num_workers=0
  valid_dataset.num_workers=0
  gconfig.n_samples=${V3_N_SAMPLES}
  gconfig.reward_normalization=true
  gconfig.drop_incomplete_group=false
  gconfig.max_new_tokens=${V3_MAX_NEW_TOKENS}
  gconfig.max_tokens=${V3_MAX_TOKENS}
  eval_gconfig.n_samples=${V3_N_SAMPLES}
  eval_gconfig.reward_normalization=true
  eval_gconfig.drop_incomplete_group=false
  eval_gconfig.max_new_tokens=${V3_MAX_NEW_TOKENS}
  eval_gconfig.max_tokens=${V3_MAX_TOKENS}
  econfig.agent_pool.0.agent_type=${AGENT_TYPE}
  econfig.agent_pool.0.agent_config=${AGENT_CONFIG}
  econfig.agent_pool.0.weight=${AGENT_WEIGHT}
  actor.max_new_tokens=${V3_MAX_NEW_TOKENS}
  rollout.consumer_batch_size=${V3_BATCH_SIZE}
  rollout.max_concurrent_rollouts=${V3_MAX_CONCURRENT_ROLLOUTS}
  rollout.pause_grace_period=${V3_PAUSE_GRACE_PERIOD}
  rollout.openai.engine_max_tokens=${V3_MAX_TOKENS}
  actor.mb_spec.max_tokens_per_mb=${V3_ACTOR_MAX_TOKENS_PER_MB}
  ref.mb_spec.max_tokens_per_mb=${V3_ACTOR_MAX_TOKENS_PER_MB}
  actor.scheduling_spec.0.mem=${V3_ACTOR_WORKER_MEM_GB}
  sglang.context_length=${V3_CONTEXT_LENGTH}
  train_dataset.max_length=${V3_CONTEXT_LENGTH}
  sglang.max_prefill_tokens=${V3_SGLANG_MAX_PREFILL_TOKENS}
  sglang.chunked_prefill_size=${V3_SGLANG_CHUNKED_PREFILL_SIZE}
  sglang.max_running_requests=${V3_SGLANG_MAX_RUNNING_REQUESTS}
  sglang.mem_fraction_static=${V3_SGLANG_MEM_FRACTION_STATIC}
  sglang.attention_backend=${V3_SGLANG_ATTENTION_BACKEND}
  sglang.disable_cuda_graph=${V3_SGLANG_DISABLE_CUDA_GRAPH}
  sglang.enable_memory_saver=${V3_SGLANG_ENABLE_MEMORY_SAVER}
  stats_logger.wandb.mode=${V3_WANDB_MODE}
  should_accept_fn=${V3_SHOULD_ACCEPT_FN}
  should_accept_fn_kwargs={}
  actor.megatron.use_deterministic_algorithms=${V3_DETERMINISTIC_ALGOS}
  actor.optimizer.warmup_steps_proportion=${V3_WARMUP_PROP}
  ++debug.dump_rollout_data=${V3_DUMP_ROLLOUT_DATA}
  ++saver.mode=${V3_SAVER_MODE}
  saver.freq_steps=${V3_SAVER_FREQ_STEPS}
  recover.mode=${V3_RECOVER_MODE}
  recover.freq_steps=${V3_RECOVER_FREQ_STEPS}
  recover.no_save_optim=${V3_RECOVER_NO_SAVE_OPTIM}
  recover.no_load_optim=${V3_RECOVER_NO_LOAD_OPTIM}
  actor.megatron.enable_fp32_lm_head=true
  ++actor.scheduling_spec.0.env_vars.NCCL_DEBUG=WARN
  ++actor.scheduling_spec.0.env_vars.NCCL_ASYNC_ERROR_HANDLING=1
  ++actor.scheduling_spec.0.env_vars.TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  ++actor.scheduling_spec.0.env_vars.NCCL_P2P_PXN_LEVEL=0
  econfig.timeout=${V3_ECONFIG_TIMEOUT}
  rollout.openai.session_timeout_seconds=${V3_SESSION_TIMEOUT}
  rollout.request_timeout=${V3_REQUEST_TIMEOUT}
  ++rollout.openai.chat_template_kwargs.thinking_option=${V3_THINKING_OPTION}
  rollout.openai.think_mode=${V3_THINK_MODE}
  rollout.openai.think_prob=${V3_THINK_PROB}
)

RUN_RL_OVERRIDES=""
for override in "${V3_OVERRIDES[@]}"; do
  printf -v quoted "%q" "${override}"
  RUN_RL_OVERRIDES+="${quoted} "
done
if [ -n "${V3_EXTRA_OVERRIDES}" ]; then
  RUN_RL_OVERRIDES+="${V3_EXTRA_OVERRIDES} "
fi

echo "=== Job ${SLURM_JOB_ID:-local} started at $(date) on $(hostname) ==="
echo "AREAL_DIR=${AREAL_DIR}"
echo "IMAGE=${IMAGE}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "TRIAL_NAME=${TRIAL_NAME}"
echo "AGENT=${AGENT_TYPE}:${AGENT_CONFIG}@${AGENT_WEIGHT}"
echo "STEPS=${V3_TOTAL_TRAIN_STEPS} BATCH=${V3_BATCH_SIZE}x${V3_N_SAMPLES} TOKENS=${V3_MAX_TOKENS} ALLOC=${V3_ALLOCATION_MODE}"
echo "FILEROOT=${FILEROOT}"

export AREAL_APPTAINER_STAGGER_SECONDS=${AREAL_APPTAINER_STAGGER_SECONDS:-15}
export AREAL_SGLANG_LAUNCH_STAGGER_SECONDS=${AREAL_SGLANG_LAUNCH_STAGGER_SECONDS:-0}
export AREAL_SGLANG_PROCESS_MONITOR_INTERVAL_SECONDS=${AREAL_SGLANG_PROCESS_MONITOR_INTERVAL_SECONDS:-10}
export AREAL_SGLANG_FORCE_LAUNCH_MODULE=${AREAL_SGLANG_FORCE_LAUNCH_MODULE:-}
export AREAL_SGLANG_DROP_LD_PRELOAD=${AREAL_SGLANG_DROP_LD_PRELOAD:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_NET=${NCCL_NET:-IB}
if [ "${NCCL_NET}" = "IB" ]; then
  export NCCL_IB_DISABLE=0
else
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
fi
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_bond}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_IB_TC=${NCCL_IB_TC:-136}
export NCCL_IB_SL=${NCCL_IB_SL:-5}
export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-8}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-7}
export NCCL_SET_THREAD_NAME=${NCCL_SET_THREAD_NAME:-1}
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_PXN_LEVEL=0
export V3_RL_CACHE_ROOT=${CACHE_ROOT}
export UV_CACHE_DIR=${CACHE_ROOT}/uv
export PIP_CACHE_DIR=${CACHE_ROOT}/pip
export XDG_CACHE_HOME=${CACHE_ROOT}/xdg
export HF_HOME=${CACHE_ROOT}/hf
export HUGGINGFACE_HUB_CACHE=${CACHE_ROOT}/hf/hub
export TORCH_HOME=${CACHE_ROOT}/torch
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" \
  "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TORCH_HOME}"

srun --mpi=pmi2 --ntasks=1 --cpus-per-task=1 --mem-per-cpu=100000M \
  singularity exec --pid --writable-tmpfs \
    --env "TRIAL_NAME=${TRIAL_NAME}" \
    --env "MODEL_PATH=${MODEL_PATH}" \
    --env "TRAIN_DATA=${TRAIN_DATA}" \
    --env "AGENT_TYPE=${AGENT_TYPE}" \
    --env "AGENT_CONFIG=${AGENT_CONFIG}" \
    --env "AGENT_WEIGHT=${AGENT_WEIGHT}" \
    --env "EXP_NAME=${EXP_NAME}" \
    --env "FILEROOT=${FILEROOT}" \
    --env "N_NODES=${N_NODES}" \
    --env "N_GPUS=${N_GPUS}" \
    --env "ALLOCATION_MODE=${V3_ALLOCATION_MODE}" \
    --env "CONFIG_PATH=${CONFIG_PATH}" \
    --env "WANDB_API_KEY=${WANDB_API_KEY}" \
    --env "WANDB_BASE_URL=${WANDB_BASE_URL}" \
    --env "WANDB_MODE=${V3_WANDB_MODE}" \
    --env "AWEX_WEIGHT_TRANSFER_VERSION=${AWEX_WEIGHT_TRANSFER_VERSION}" \
    --env "AREAL_AWEX_PROCESS_QUEUE_WHEN_IDLE=${AREAL_AWEX_PROCESS_QUEUE_WHEN_IDLE}" \
    --env "AENV_SYSTEM_URL=${AENV_SYSTEM_URL}" \
    --env "AWEAGENT_ROOT=${AWEAGENT_ROOT}" \
    --env "SWE_AGENT_ROOT=${SWE_AGENT_ROOT}" \
    --env "CC_MAX_OUTPUT_TOKENS=${CC_MAX_OUTPUT_TOKENS}" \
    --env "CC_MAX_THINKING_TOKENS=${CC_MAX_THINKING_TOKENS}" \
    --env "CC_DISABLE_EXPERIMENTAL_BETAS=${CC_DISABLE_EXPERIMENTAL_BETAS}" \
    --env "CLAUDE_CODE_MAX_OUTPUT_TOKENS=${CLAUDE_CODE_MAX_OUTPUT_TOKENS}" \
    --env "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=${CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING}" \
    --env "MAX_THINKING_TOKENS=${MAX_THINKING_TOKENS}" \
    --env "CLAUDE_CODE_ATTRIBUTION_HEADER=${CLAUDE_CODE_ATTRIBUTION_HEADER}" \
    --env "CLAUDE_CODE_AUTO_COMPACT_WINDOW=${CLAUDE_CODE_AUTO_COMPACT_WINDOW}" \
    --env "FLASH_LINEAR_ATTENTION=${FLASH_LINEAR_ATTENTION}" \
    --env "FLASH_LINEAR_ATTENTION_ROOT=${FLASH_LINEAR_ATTENTION_ROOT}" \
    --env "AWEX_ROOT=${AWEX_ROOT}" \
    --env "THETA_SGLANG=${THETA_SGLANG}" \
    --env "SGLANG_V3_WORK=${SGLANG_V3_WORK}" \
    --env "V3_RL_CACHE_ROOT=${V3_RL_CACHE_ROOT}" \
    --env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
    --env "PIP_CACHE_DIR=${PIP_CACHE_DIR}" \
    --env "XDG_CACHE_HOME=${XDG_CACHE_HOME}" \
    --env "HF_HOME=${HF_HOME}" \
    --env "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}" \
    --env "TORCH_HOME=${TORCH_HOME}" \
    --env "AREAL_APPTAINER_STAGGER_SECONDS=${AREAL_APPTAINER_STAGGER_SECONDS}" \
    --env "AREAL_SGLANG_LAUNCH_STAGGER_SECONDS=${AREAL_SGLANG_LAUNCH_STAGGER_SECONDS}" \
    --env "AREAL_SGLANG_PROCESS_MONITOR_INTERVAL_SECONDS=${AREAL_SGLANG_PROCESS_MONITOR_INTERVAL_SECONDS}" \
    --env "AREAL_SGLANG_FORCE_LAUNCH_MODULE=${AREAL_SGLANG_FORCE_LAUNCH_MODULE}" \
    --env "AREAL_SGLANG_DROP_LD_PRELOAD=${AREAL_SGLANG_DROP_LD_PRELOAD}" \
    --env "NCCL_DEBUG=${NCCL_DEBUG}" \
    --env "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}" \
    --env "NCCL_NET=${NCCL_NET}" \
    --env "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}" \
    --env "NCCL_IB_HCA=${NCCL_IB_HCA}" \
    --env "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}" \
    --env "NCCL_IB_TC=${NCCL_IB_TC}" \
    --env "NCCL_IB_SL=${NCCL_IB_SL}" \
    --env "NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION}" \
    --env "NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT}" \
    --env "NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT}" \
    --env "NCCL_SET_THREAD_NAME=${NCCL_SET_THREAD_NAME}" \
    --env "NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING}" \
    --env "TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING}" \
    --env "NCCL_P2P_PXN_LEVEL=${NCCL_P2P_PXN_LEVEL}" \
    --bind /storage:/storage \
    --bind /home:/home \
    --bind /etc/slurm/:/etc/slurm/ \
    --bind /etc/passwd:/etc/passwd:ro \
    --bind /etc/group:/etc/group:ro \
    --bind /etc/munge:/etc/munge:ro \
    --bind /var/run/munge:/var/run/munge \
    --bind /usr/bin/sbatch:/usr/bin/sbatch \
    --bind /usr/bin/srun:/usr/bin/srun \
    --bind /usr/bin/squeue:/usr/bin/squeue \
    --bind /usr/bin/scancel:/usr/bin/scancel \
    --bind /usr/bin/scontrol:/usr/bin/scontrol \
    --bind /usr/lib64/slurm:/usr/lib64/slurm \
    "${IMAGE}" \
    bash -lc "(/usr/sbin/munged 2>/dev/null || true) && cd ${AREAL_DIR} && . /opt/.venv/bin/activate && export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ && uv pip install --python /opt/.venv/bin/python -e . && uv pip install --python /opt/.venv/bin/python datasets torch_memory_saver concurrent-log-handler && cd ${AWEAGENT_ROOT} && uv pip install --python /opt/.venv/bin/python -r requirements.txt && cd ${AREAL_DIR} && python -c 'import sys,datasets,concurrent_log_handler; print(\"controller_python\", sys.executable, \"datasets\", datasets.__version__)' && export PYTHONPATH=${AWEAGENT_ROOT}:${AWEAGENT_ROOT}/src:${AREAL_DIR}:${THETA_SGLANG}:${SGLANG_V3_WORK}/ossstub:${AWEX_ROOT}:\${PYTHONPATH:-} && if [ -d \"${FLASH_LINEAR_ATTENTION}\" ]; then export PYTHONPATH=${FLASH_LINEAR_ATTENTION}:\${PYTHONPATH}; fi && bash examples/swe/rl/run_rl.sh ${RUN_RL_OVERRIDES}"

echo "=== Job ${SLURM_JOB_ID:-local} finished at $(date), exit code: $? ==="
