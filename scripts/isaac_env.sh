# Environment for running Isaac Lab scripts against a BINARY Isaac Sim install.
#
#   source scripts/isaac_env.sh
#   cd IsaacLab && python scripts/environments/train_motion_imitation.py ...
#
# Not needed if you pip-installed Isaac Sim into the venv (`pip install
# "isaacsim[all,extscache]==4.5.0"`), in which case `python` already finds it.
#
# Why each line is here -- all four were failures hit on a clean machine:
#   1. setup_python_env.sh references $PYTHONPATH / $LD_LIBRARY_PATH, so they must
#      exist first or it aborts under `set -u`.
#   2. It puts isaacsim / omni / pxr on the path. Without it: ModuleNotFoundError:
#      No module named 'isaacsim'.
#   3. ROS Noetic's python 3.8 on PYTHONPATH crashes Isaac's bundled interpreter
#      ("cannot import name 'text_encoding' from 'io'"), so strip it AFTER sourcing.
#   4. setup_python_env.sh sets neither ISAAC_PATH nor EXP_PATH / CARB_APP_PATH.
#      Unset -> IndexError in isaacsim/__init__.py, then KeyError: 'EXP_PATH',
#      then KeyError: 'CARB_APP_PATH'. Non-interactive shells always hit these.
#
# `IsaacLab/_isaac_sim` must be a valid symlink to YOUR Isaac Sim install:
#   ln -s ~/isaacsim /path/to/CopyQuad/IsaacLab/_isaac_sim

_copyquad_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
_isaac="$_copyquad_root/IsaacLab/_isaac_sim"

if [ ! -e "$_isaac/setup_python_env.sh" ]; then
    echo "isaac_env.sh: $_isaac is not a valid Isaac Sim install." >&2
    echo "  ln -s /path/to/isaacsim $_copyquad_root/IsaacLab/_isaac_sim" >&2
else
    export PYTHONPATH="${PYTHONPATH:-}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    # shellcheck disable=SC1091
    source "$_isaac/setup_python_env.sh"
    export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -vi ros/noetic | paste -sd:)
    export ISAAC_PATH="$_isaac"
    export EXP_PATH="$_isaac/apps"
    export CARB_APP_PATH="$_isaac/kit"
    echo "isaac_env.sh: ISAAC_PATH=$ISAAC_PATH"
fi
unset _copyquad_root _isaac
