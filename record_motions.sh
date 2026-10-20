#!/bin/bash
# ============================================================================
# record_motions.sh -- one mp4 per deployed motion policy, for the paper/demo reel.
#
# Records the 5 released behaviours (dog walk, horse walk, horse trot, dog bow,
# dog give-paw) from the policies deployed on the physical ANYmal-D.
#
# The eval mode writes rollout.npz / eval.log / eval_config.yaml NEXT TO THE
# CHECKPOINT, and those rollouts are what the paper's tab:gait and the give-paw
# support-polygon margin are computed from. So this script never runs against the
# real checkpoint dirs: it copies model.pt + train_config.yaml into a throwaway
# stage dir and evaluates that. The originals are left untouched.
#
# Runs WITH the GUI by default -- the "Grey Studio" light rig lives in the viewport
# menubar extension, which the headless rendering app does not load. HEADLESS=1
# still records, but with the scene's own lighting.
#
#   bash ~/Research/CopyDog/record_motions.sh              # all five
#   bash ~/Research/CopyDog/record_motions.sh dog_bow      # just one or more
#   LEN=1500 RES_W=2560 RES_H=1440 bash .../record_motions.sh
# ============================================================================
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ISAAC=$ROOT/IsaacLab
PY=$ROOT/venv/bin/python
S=scripts/environments/train_motion_imitation.py
C=../checkpoints
M=scripts/environments/data/motions

OUT=${OUT:-$ROOT/data/motion_videos}
STAGE=${STAGE:-$OUT/_ckpt_stage}
LEN=${LEN:-1000}              # env steps @ 50 Hz -> 1000 = 20 s = one full episode
RES_W=${RES_W:-1920}
RES_H=${RES_H:-1080}
EYE=${EYE:-5 0 2}             # camera offset from the robot base (x y z)
HEADLESS=${HEADLESS:-0}
GRACE=${GRACE:-40}                # seconds to let Isaac exit by itself once the mp4 is done
RUN_TIMEOUT=${RUN_TIMEOUT:-600}   # outer backstop per motion

# --- shutting Isaac down ------------------------------------------------------
# A recording is FINISHED the moment RecordVideo reaches --video-length: it muxes
# the mp4 right there, mid-run, not at exit. Everything after that is Isaac tearing
# itself down, and with the GUI + cameras `simulation_app.close()` can leave the
# window sitting there, which also stalls the remaining motions.
#
# So each run gets a watchdog: poll for the mp4, and once its size stops changing
# (moviepy writes it progressively, so "exists" is not the same as "finished"),
# give Isaac $GRACE seconds to close on its own and then take it down. `timeout`
# is the outer backstop if the mp4 never appears at all.
#
# Killing after the mp4 is written costs nothing -- verified: a hard-killed run
# still produced a complete, playable file -- and the eval outputs land in the
# throwaway stage dir either way. Headless does not need any of this (measured:
# clean exit in 34 s, no orphans); the watchdog simply never fires there.

child=""
watchdog=""
pat=""            # pgrep pattern matching the current run's python, set per motion

# Take a run down for good: TERM the `timeout` wrapper, then TERM any python it
# orphaned on the way out, escalating to KILL. Isaac ignores TERM for a while
# during teardown, so each stage gets 15 s before the hammer. Returns only once
# nothing matching $pat is left.
stop_run() {
  local pid="$1" i
  kill -TERM "$pid" 2>/dev/null
  for ((i = 0; i < 15; i++)); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL "$pid" 2>/dev/null
  [ -n "$pat" ] || return 0
  pkill -TERM -f "$pat" 2>/dev/null
  for ((i = 0; i < 15; i++)); do pgrep -f "$pat" >/dev/null || break; sleep 1; done
  pkill -KILL -f "$pat" 2>/dev/null
  for ((i = 0; i < 5; i++)); do pgrep -f "$pat" >/dev/null || break; sleep 1; done
}

# Poll $1 (the mp4) while $2 (the run) lives; close the run once the file settles.
watch_and_close() {
  local f="$1" pid="$2" sz1 sz2 i
  while kill -0 "$pid" 2>/dev/null; do
    if [ -f "$f" ]; then
      # moviepy writes the file progressively, so "exists" != "finished".
      sz1=$(stat -c %s "$f" 2>/dev/null || echo 0)
      sleep 3
      sz2=$(stat -c %s "$f" 2>/dev/null || echo 0)
      if [ "$sz1" = "$sz2" ] && [ "$sz1" != "0" ]; then
        for ((i = 0; i < GRACE; i++)); do
          kill -0 "$pid" 2>/dev/null || return 0      # closed on its own, nothing to do
          sleep 1
        done
        echo "   (recording finished; Isaac still up after ${GRACE}s, closing it)"
        stop_run "$pid"
        return 0
      fi
    else
      sleep 3
    fi
  done
}
cleanup() {
  [ -n "$watchdog" ] && kill "$watchdog" 2>/dev/null
  if [ -n "$child" ] && kill -0 "$child" 2>/dev/null; then
    echo
    echo "-- interrupted, shutting Isaac down --"
    stop_run "$child"
  fi
  # Anything this script started but no longer has a PID for. Scoped to $STAGE,
  # which only this script writes into, so it can never hit another session.
  pkill -KILL -f "train_motion_imitation.py.*$STAGE" 2>/dev/null
  exit 130
}
trap cleanup INT TERM

# Isaac dies if ROS Noetic's python 3.8 is on PYTHONPATH (see the run-environment notes).
export PYTHONPATH=$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -vi ros/noetic | paste -sd:)
cd "$ISAAC" || exit 1
mkdir -p "$OUT" "$STAGE"

# name : checkpoint dir (under $C) : reference clip (under $M)
RUNS=(
  "dog_walk:dog1_slow/policy_rl_8c5eb2:dog1_slow"
  "horse_walk:horses_walk/policy_rl_ca7822:horses_walk"
  "horse_trot:horses_trot_v2/policy_rl_846766:horses_trot_v2"
  "dog_bow:dog_bow_2/policy_rl_97e40f:dog_bow_2"
  "dog_givepaw:dog_givepaw_gentle/policy_rl_4f2367:dog_givepaw_gentle"
)

selected=("$@")                # no args = record everything
for row in "${RUNS[@]}"; do
  name="${row%%:*}"; rest="${row#*:}"; ckpt="${rest%%:*}"; clip="${rest#*:}"

  if [ ${#selected[@]} -gt 0 ]; then
    hit=0
    for a in "${selected[@]}"; do [ "$a" = "$name" ] && hit=1; done
    [ $hit -eq 1 ] || continue
  fi

  if [ ! -f "$C/$ckpt/model.pt" ]; then
    echo "!! $name: missing $C/$ckpt/model.pt -- skipped"; continue
  fi

  sdir="$STAGE/$name"
  mkdir -p "$sdir"
  cp "$C/$ckpt/model.pt" "$C/$ckpt/train_config.yaml" "$sdir/"

  echo
  echo "======== $name  <-  $ckpt  (clip $clip) ========"
  hl=""; [ "$HEADLESS" = "1" ] && hl="--headless"
  log="$OUT/logs/$name.log"
  mkdir -p "$OUT/logs"
  src="$OUT/$clip-step-0.mp4"     # RecordVideo names it after the clip
  rm -f "$src"                    # so the watchdog can't latch onto a previous take
  pat="train_motion_imitation.py.*$sdir"   # matches only this run's python

  # Run in the background (not through a pipe) so we hold the child's PID and can
  # guarantee it is gone before the next motion opens another window.
  # shellcheck disable=SC2086
  timeout -k 20 "$RUN_TIMEOUT" \
    $PY $S $hl --video \
      --task Isaac-Velocity-Flat-Anymal-D-v0 --mode eval \
      --motion "$M/$clip.npz" --ckpt "$sdir/model.pt" \
      --eval-episodes 1 \
      --video-length "$LEN" --video-res "$RES_W" "$RES_H" \
      --video-dir "$OUT" --cam-eye $EYE \
      > "$log" 2>&1 &
  child=$!

  watch_and_close "$src" "$child" &
  watchdog=$!

  # stderr is muted around the waits: when the watchdog closes a run, bash would
  # otherwise print its own "Killed <the whole command line>" / "Terminated" job
  # notices, which read like failures when they are the intended shutdown.
  { wait "$child"; rc=$?; } 2>/dev/null
  child=""
  { kill "$watchdog" 2>/dev/null; wait "$watchdog"; } 2>/dev/null
  watchdog=""

  grep -E "\[Video\]|\[Viewer\]|PHYSICAL error" "$log" | head -10
  case $rc in
    0|143|137)  ;;   # closed itself, or the watchdog closed it after the recording
    124)  echo "   (no mp4 after ${RUN_TIMEOUT}s -- killed by the backstop, see $log)" ;;
    *)    echo "   (exit $rc -- see $log)"
          grep -E "Traceback|Error:" "$log" | head -5 ;;
  esac
  # A forced close leaves Isaac teardown noise ("ReferenceError: weakly-referenced
  # object no longer exists") in the log. That is expected after 143/137 and is not
  # shown, which is why Traceback lines are only printed for an unexpected rc.

  # Nothing may still be holding a GPU when the next motion starts. Scoped to
  # $sdir, so this can only ever match this script's own run.
  if pgrep -f "$pat" >/dev/null; then
    pkill -TERM -f "$pat" 2>/dev/null
    for _ in $(seq 15); do pgrep -f "$pat" >/dev/null || break; sleep 1; done
    pkill -KILL -f "$pat" 2>/dev/null
    sleep 1
    pgrep -f "$pat" >/dev/null && echo "   (WARNING: an Isaac process survived the sweep)" \
                               || echo "   (swept a leftover Isaac process)"
  fi

  # RecordVideo names the file after the clip; rename to the behaviour.
  if [ -f "$src" ]; then
    mv "$src" "$OUT/$name.mp4"
    echo "   -> $OUT/$name.mp4"
  else
    echo "!! $name: no mp4 produced (expected $src)"
  fi
done

echo
echo "==== done. videos in $OUT ===="
ls -la "$OUT"/*.mp4 2>/dev/null
echo "(the real checkpoint dirs were not written to; scratch evals are under $STAGE)"
