#!/usr/bin/env bash
# Sync truck_scheduling to a remote host and run ALNS-vs-hybrid tuning in tmux.
#
# Usage:
#   export REMOTE_HOST=65.109.171.215   # or read from ./ip
#   export REMOTE_USER=root            # optional
#   ./scripts/run_alns_vs_hybrid_cloud.sh [--smoke] [extra tuner args...]
#
# Env:
#   REMOTE_HOST / reads ./ip
#   REMOTE_USER (default: root)
#   REMOTE_DIR  (default: ~/truck_scheduling)
#   SSH_KEY     (default: ~/.ssh/id_ed25519 if present)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${REMOTE_HOST:-}" ]]; then
  if [[ -f "$ROOT/ip" ]]; then
    REMOTE_HOST="$(tr -d '[:space:]' < "$ROOT/ip")"
  else
    echo "REMOTE_HOST unset and $ROOT/ip missing" >&2
    exit 1
  fi
fi

REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_DIR="${REMOTE_DIR:-~/truck_scheduling}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
if [[ -n "${SSH_KEY:-}" ]]; then
  SSH_OPTS+=(-i "$SSH_KEY")
elif [[ -f "${HOME}/.ssh/id_ed25519" ]]; then
  SSH_OPTS+=(-i "${HOME}/.ssh/id_ed25519")
elif [[ -f "${HOME}/.ssh/id_rsa" ]]; then
  SSH_OPTS+=(-i "${HOME}/.ssh/id_rsa")
fi

SMOKE=0
TUNER_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--smoke" ]]; then
    SMOKE=1
  else
    TUNER_ARGS+=("$arg")
  fi
done

if [[ "$SMOKE" -eq 1 ]]; then
  TUNER_ARGS=(--smoke "${TUNER_ARGS[@]}")
  SESSION="alns-hybrid-smoke"
else
  # Full large-K loop defaults (override via extra args)
  if [[ ${#TUNER_ARGS[@]} -eq 0 ]]; then
    TUNER_ARGS=(
      --buckets 50x50,100x100,200x200,400x400
      --n-configs 6
      --max-rounds 8
      --seeds 2
      --target-win-rate 0.8
      --max-time-ratio 1.5
      --max-wall-sec 43200
    )
  fi
  SESSION="alns-hybrid-tune"
fi

echo "==> Remote ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"

RSYNC_BIN="$(command -v rsync || true)"
if [[ -n "$RSYNC_BIN" ]]; then
  "$RSYNC_BIN" -az --delete \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '.git' \
    --exclude '.pytest_cache' \
    --exclude 'data/results/*.png' \
    -e "ssh ${SSH_OPTS[*]}" \
    "$ROOT/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
else
  echo "rsync not found; using tar-over-ssh"
  tar --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
    --exclude='.pytest_cache' -C "$ROOT" -czf - . \
    | ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
      "mkdir -p ${REMOTE_DIR} && tar -xzf - -C ${REMOTE_DIR}"
fi

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
cd ${REMOTE_DIR}
python3 -m venv .venv
. .venv/bin/activate
pip install -q -U pip
pip install -q -r requirements.txt
mkdir -p data/results
# Smoke one 50x50 pair unless --smoke (tiny) mode
if [[ "$SMOKE" -eq 0 ]]; then
  python - <<'PY'
from src.instance_gen import gen_instance
from src.solvers.alns import ALNS
from src.solvers.ga_tabu import HybridGATabu
from src.tuning.tune_alns_vs_hybrid import budget_for_k
from src.validate import validate

inst = gen_instance(seed=0, M=50, N=50, G=2)
b = budget_for_k(len(inst.ops))
print(f"smoke50 K={len(inst.ops)} B={b}")
hy = HybridGATabu().solve(inst, time_limit_sec=b, seed=0)
al = ALNS().solve(inst, time_limit_sec=1.5 * b, seed=0)
validate(inst, hy); validate(inst, al)
print(f"hybrid obj={hy.objective(inst):.1f} t={hy.runtime_sec:.2f}")
print(f"alns    obj={al.objective(inst):.1f} t={al.runtime_sec:.2f}")
PY
fi
ARGS=(${TUNER_ARGS[*]})
nohup python -m src.tuning.tune_alns_vs_hybrid "\${ARGS[@]}" \
  > data/results/alns_vs_hybrid_run.log 2>&1 &
echo \$! > data/results/alns_vs_hybrid_run.pid
echo "started pid=\$(cat data/results/alns_vs_hybrid_run.pid)"
tail -n 5 data/results/alns_vs_hybrid_run.log || true
EOF
)

# Prefer tmux if available remotely
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "command -v tmux >/dev/null" \
  && USE_TMUX=1 || USE_TMUX=0

if [[ "$USE_TMUX" -eq 1 ]]; then
  echo "==> Launching in remote tmux session ${SESSION}"
  # Escape for remote shell: write a runner script then tmux new-session
  ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" bash -s <<EOF
set -euo pipefail
cd ${REMOTE_DIR}
python3 -m venv .venv
. .venv/bin/activate
pip install -q -U pip
pip install -q -r requirements.txt
mkdir -p data/results
cat > /tmp/run_${SESSION}.sh <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
cd ${REMOTE_DIR}
source .venv/bin/activate
exec python -m src.tuning.tune_alns_vs_hybrid ${TUNER_ARGS[*]} \
  2>&1 | tee data/results/alns_vs_hybrid_run.log
RUN
chmod +x /tmp/run_${SESSION}.sh
tmux has-session -t ${SESSION} 2>/dev/null && tmux kill-session -t ${SESSION} || true
tmux new-session -d -s ${SESSION} "/tmp/run_${SESSION}.sh"
echo "tmux session ${SESSION} started"
tmux ls | grep ${SESSION} || true
EOF
else
  echo "==> Launching via nohup (no tmux on remote)"
  ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" bash -c "$REMOTE_CMD"
fi

echo "==> To pull results later:"
echo "  scp ${SSH_OPTS[*]} ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/config/alns_params.json config/"
echo "  scp ${SSH_OPTS[*]} ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/data/results/alns_vs_hybrid_*.{csv,json,log} data/results/"
