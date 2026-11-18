#!/usr/bin/env bash
# Read-only, CPU-only regression checks for the Thor GPU occupation policy.
# Never starts CUDA, touches GPU devices, or modifies a running service.
set -euo pipefail
cd "$(dirname "$0")/.."
guard="scripts/native_arm64_gpu_occupant.sh"
bash -n "$guard"

cases=0
assert_code() {
  local expected="$1" label="$2"
  shift 2
  local actual
  if bash "$guard" "$@" >/dev/null 2>&1; then
    actual=0
  else
    actual=$?
  fi
  if [[ "$actual" -ne "$expected" ]]; then
    echo "::error::GPU_OCCUPANT_POLICY_FAIL label=$label expected=$expected got=$actual" >&2
    exit 1
  fi
  cases=$((cases + 1))
}

# All previously recognised services remain deferred, not admitted.
assert_code 0 scirust-pretrain root 1 \
  "0::/user.slice/user-0.slice/session-31.scope" \
  "/root/scirust/target/release/examples/cuda_pretrain" \
  "/root/scirust/target/release/examples/cuda_pretrain --epochs 1"
assert_code 0 viggle-managed remoteops 1 \
  "0::/system.slice/memorithm-viggle-lan.service" \
  "/srv/models/qwen-image-2.1-viggle-turbo/python" \
  "python /srv/models/qwen-image-2.1-viggle-turbo/lan-server.py"
assert_code 0 clm-managed memorithm-clm 19 \
  "0::/system.slice/memorithm-clm.service" "clm-serve" \
  "/srv/models/clm-v0.1-8b/target/clm-serve"
assert_code 0 clm-encoder memorithm-clm 27 \
  "0::/system.slice/memorithm-clm-encoder.service" \
  "VLLM::EngineCore" "VLLM::EngineCore --worker"

# Exact service identities observed on Thor in GitHub job 135149502283.
checkup_cmd="/srv/models/checkupauto-character/venv/bin/python /srv/models/checkupauto-character/app/worker.py"
ollama_cmd="/usr/local/lib/ollama/llama-server --model /usr/share/ollama/.ollama/models/blobs/sha256-abcdef01234 --port 45875 --host 127.0.0.1 --offline"
assert_code 0 checkup-character remoteops 1 \
  "0::/system.slice/checkupauto-character.service" \
  "/srv/models/checkupauto-character/venv/bin/python" "$checkup_cmd"
assert_code 0 ollama-managed ollama 1973 \
  "0::/system.slice/ollama.service" \
  "/usr/local/lib/ollama/llama-server" "$ollama_cmd"

# Name-only matches, wrong cgroups, wrong users, altered binaries,
# nonlocal endpoints and unknown GPU workloads must not be whitelisted.
assert_code 1 checkup-wrong-user root 1 \
  "0::/system.slice/checkupauto-character.service" \
  "/srv/models/checkupauto-character/venv/bin/python" "$checkup_cmd"
assert_code 1 checkup-wrong-parent remoteops 2 \
  "0::/system.slice/checkupauto-character.service" \
  "/srv/models/checkupauto-character/venv/bin/python" "$checkup_cmd"
assert_code 1 checkup-wrong-unit remoteops 1 \
  "0::/system.slice/checkupauto-character-copy.service" \
  "/srv/models/checkupauto-character/venv/bin/python" "$checkup_cmd"
assert_code 1 checkup-wrong-binary remoteops 1 \
  "0::/system.slice/checkupauto-character.service" \
  "/tmp/python" "$checkup_cmd"
assert_code 1 ollama-wrong-user root 1973 \
  "0::/system.slice/ollama.service" \
  "/usr/local/lib/ollama/llama-server" "$ollama_cmd"
assert_code 1 ollama-wrong-unit ollama 1973 \
  "0::/system.slice/custom.service" \
  "/usr/local/lib/ollama/llama-server" "$ollama_cmd"
assert_code 1 ollama-nonlocal ollama 1973 \
  "0::/system.slice/ollama.service" \
  "/usr/local/lib/ollama/llama-server" \
  "/usr/local/lib/ollama/llama-server --model /usr/share/ollama/.ollama/models/blobs/sha256-abcdef01234 --host 0.0.0.0 --offline"
assert_code 1 unknown-workload unknown 1 \
  "0::/system.slice/unknown.service" \
  "/tmp/cuda-burn" "/tmp/cuda-burn"
assert_code 2 malformed-identity remoteops 1 "" "/tmp/proc" "/tmp/proc"
assert_code 2 missing-fields remoteops 1

echo "NATIVE_ARM64_GPU_OCCUPANT_POLICY_PASS cases=$cases (no hardware touched)"
