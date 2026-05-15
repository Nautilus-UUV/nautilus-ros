#!/usr/bin/env bash
# Thin wrapper around `apptainer exec` for the Nautilus sim SIF.
#
# Matches the documented direct command (just --cleanenv + GZ_IP +
# the sim_data bind), and adds two opt-in conveniences:
#
#     --scenarios-dir DIR   bind-mount host DIR at /ros2_ws/scenarios:ro
#                           so `scenario:=/ros2_ws/scenarios/<file>.yaml`
#                           reaches a host-authored YAML
#     --cpus CPULIST        pin the apptainer process tree to the given
#                           host CPUs via taskset (e.g. 0-3 or 0,2,4) so
#                           one sim run doesn't fight others for cores
#
# Run from the workspace root so `./sim_data` resolves correctly.
#
# Example:
#     ./src/nautilus-ros/scripts/apptainer_exec.sh \
#         --scenarios-dir ./scenarios --cpus 0-3 nautilus_sim.sif \
#         ros2 launch nautilus_hal sawtooth_sim.launch.py \
#             headless:=true mission_autostart:=true \
#             target_pressure_pa:=147150.0 angle_rad:=0.6109 n_resurfaces:=5 \
#             record:=true sampler_id:=my_run run_id:=baseline \
#             scenario:=/ros2_ws/scenarios/baseline.yaml

set -euo pipefail

SCENARIOS_DIR=""
CPUS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenarios-dir) SCENARIOS_DIR="$2"; shift 2 ;;
        --cpus)          CPUS="$2";          shift 2 ;;
        --) shift; break ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        --*)
            echo "$(basename "$0"): unknown flag: $1" >&2
            exit 2
            ;;
        *) break ;;
    esac
done

if [[ $# -lt 2 ]]; then
    echo "usage: $(basename "$0") [--scenarios-dir DIR] [--cpus CPULIST] <sif> <cmd> [args...]" >&2
    exit 2
fi

SIF="$1"; shift

if [[ ! -f "$SIF" ]]; then
    echo "$(basename "$0"): SIF not found: $SIF" >&2
    exit 1
fi

mkdir -p ./sim_data
binds=( --bind "$(pwd)/sim_data:/ros2_ws/sim_data" )

if [[ -n "$SCENARIOS_DIR" ]]; then
    if [[ ! -d "$SCENARIOS_DIR" ]]; then
        echo "$(basename "$0"): --scenarios-dir not a directory: $SCENARIOS_DIR" >&2
        exit 1
    fi
    binds+=( --bind "$(readlink -f "$SCENARIOS_DIR"):/ros2_ws/scenarios:ro" )
fi

cmd=( apptainer exec --cleanenv --env GZ_IP=127.0.0.1 "${binds[@]}"
      "$SIF" /entrypoint.sh "$@" )

if [[ -n "$CPUS" ]]; then
    if ! command -v taskset >/dev/null; then
        echo "$(basename "$0"): taskset not found on PATH; needed for --cpus" >&2
        exit 1
    fi
    cmd=( taskset -c "$CPUS" "${cmd[@]}" )
fi

exec "${cmd[@]}"
