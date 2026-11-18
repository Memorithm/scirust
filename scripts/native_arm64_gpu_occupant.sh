#!/usr/bin/env bash
# Exact process provenance classifier for the SciRust Thor occupancy guard.
# Exit 0: verified managed *busy* GPU service (CUDA must be deferred).
# Exit 1: unrecognised occupant (CUDA must be refused).
# Exit 2: bad inputs; never grants GPU admission. No device interaction.
set -euo pipefail
if [[ "$#" -ne 5 ]]; then
  echo "expected: USER PPID CGROUP PROCESS_NAME CMDLINE" >&2
  exit 2
fi
proc_user="$1"
ppid="$2"
cgroup="$3"
process_name="$4"
cmdline="$5"
if [[ -z "$proc_user" || -z "$ppid" || -z "$cgroup" || -z "$process_name" || -z "$cmdline" ]]; then
  exit 2
fi

            if [[ "$process_name" == "/root/scirust/target/release/examples/cuda_pretrain" ]] \
              && [[ "$cmdline" == *"/root/scirust/target/release/examples/cuda_pretrain"* ]] \
              && [[ "$proc_user" == "root" ]] \
              && [[ "$ppid" == "1" ]] \
              && [[ "$cgroup" == 0::/user.slice/user-0.slice/session-*.scope ]]; then
              exit 0
            elif [[ "$proc_user" == "remoteops" ]] \
              && [[ "$ppid" == "1" ]] \
              && [[ "$cgroup" == "0::/system.slice/memorithm-viggle-lan.service" ]] \
              && [[ "$cmdline" == *"/srv/models/qwen-image-2.1-viggle-turbo/lan-server.py"* ]]; then
              exit 0
            elif [[ "$proc_user" == "memorithm-clm" ]] \
              && [[ "$cgroup" == "0::/system.slice/memorithm-clm.service" ]] \
              && [[ "$cmdline" == *"/srv/models/clm-v0.1-8b/"* ]] \
              && [[ "$cmdline" == *"clm-serve"* ]]; then
              exit 0
            elif [[ "$proc_user" == "memorithm-clm" ]] \
              && [[ "$cgroup" == "0::/system.slice/memorithm-clm-encoder.service" ]] \
              && [[ "$process_name" == "VLLM::EngineCore" ]]; then
              exit 0
            # Existing CheckupAuto/OLLAMA system services can reserve the GPU.
            # They are never stopped or preempted by SciRust. Require exact
            # service account, cgroup and executable/command provenance to
            # classify them as *occupied*, not as safe to run CUDA alongside.
            elif [[ "$proc_user" == "remoteops" ]] \
              && [[ "$ppid" == "1" ]] \
              && [[ "$cgroup" == "0::/system.slice/checkupauto-character.service" ]] \
              && [[ "$process_name" == "/srv/models/checkupauto-character/venv/bin/python" ]] \
              && [[ "$cmdline" == *"/srv/models/checkupauto-character/app/worker.py"* ]]; then
              exit 0
            elif [[ "$proc_user" == "ollama" ]] \
              && [[ "$cgroup" == "0::/system.slice/ollama.service" ]] \
              && [[ "$process_name" == "/usr/local/lib/ollama/llama-server" ]] \
              && [[ "$cmdline" == *"/usr/local/lib/ollama/llama-server --model /usr/share/ollama/.ollama/models/blobs/sha256-"* ]] \
              && [[ "$cmdline" == *" --host 127.0.0.1 "* ]]; then
              exit 0
            fi
exit 1
