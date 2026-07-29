#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
isaac_root="${ISAAC_SIM_ROOT:-/home/limx/apps/isaacsim-6.0.1}"
prebundle="${isaac_root}/exts/isaacsim.pip.newton/pip_prebundle"
warp_root="$(find "${isaac_root}/extscache" -maxdepth 1 -type d -name 'omni.warp.core-1.13.*' -print -quit)"

if [[ ! -x "${isaac_root}/isaac-sim.newton.sh" ]]; then
    echo "Isaac Sim Newton launcher not found: ${isaac_root}/isaac-sim.newton.sh" >&2
    exit 1
fi
if [[ ! -f "${repo_root}/newton/__init__.py" ]]; then
    echo "Newton checkout not found: ${repo_root}" >&2
    exit 1
fi
if [[ -z "${warp_root}" || ! -d "${prebundle}" ]]; then
    echo "Isaac Sim Newton dependencies were not found beneath ${isaac_root}" >&2
    exit 1
fi

export PYTHONPATH="${repo_root}:${script_dir}:${warp_root}:${prebundle}${PYTHONPATH:+:${PYTHONPATH}}"
"${isaac_root}/python.sh" "${script_dir}/verify_source.py" --expected-repo "${repo_root}"
exec env -u OMNI_KIT_ACCEPT_EULA "${isaac_root}/isaac-sim.newton.sh" "$@"
