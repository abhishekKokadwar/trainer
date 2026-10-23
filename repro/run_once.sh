#!/usr/bin/env bash
# One run that mirrors the Kubeflow SDK container backend (kubeflow/sdk main,
# kubeflow/trainer/backends/container): two containers on a user network, a shared
# /workspace volume, the same pip snippet, master-wait loop and torchrun command,
# and the training function from examples/local/local-container-mnist.ipynb.
#
# Usage: run_once.sh <image> <run-id>
# Prints one "RESULT <run-id> <OK|SLOW|HANG|NOFINISH> ..." line. On SLOW or HANG,
# writes ps, py-spy dumps and logs for both containers into diag-<job>/.
set -u
IMG=$1; id=$2
job="r$(date +%s)${id//[^0-9a-z]/}"; net="$job-net"; vol="$job-ws"
SCRIPT=$(cat "$(dirname "$0")/train.py")
PIP='if PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1 python -m pip install --no-warn-script-location --index-url https://pypi.org/simple  --user torchvision >/tmp/pip_install.log 2>&1; then echo "Successfully installed Python packages (user): torchvision"; elif PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1 python -m pip install --no-warn-script-location --index-url https://pypi.org/simple  torchvision >>/tmp/pip_install.log 2>&1; then echo "Successfully installed Python packages (system-wide): torchvision"; else echo "ERROR: Failed to install Python packages: torchvision" >&2; cat /tmp/pip_install.log >&2; exit 1; fi && '

cleanup() { docker rm -f "$job-node-0" "$job-node-1" >/dev/null 2>&1; docker network rm "$net" >/dev/null 2>&1; docker volume rm "$vol" >/dev/null 2>&1; }

dump() { # $1 = label
  local d="diag-$job"; mkdir -p "$d"
  for n in 0 1; do
    docker exec "$job-node-$n" ps -eo pid,ppid,stat,etime,wchan:32,cmd > "$d/$1-ps-node-$n.txt" 2>&1
    docker run --rm --pid=container:"$job-node-$n" --cap-add SYS_PTRACE "$IMG" bash -lc \
      'pip install -q py-spy >/dev/null 2>&1; for p in $(pgrep -f "python|torchrun"); do echo "=== PID $p: $(tr "\0" " " </proc/$p/cmdline)"; py-spy dump --native --locals --pid $p 2>&1; done' \
      > "$d/$1-pyspy-node-$n.txt" 2>&1
  done
}

docker network create "$net" >/dev/null && docker volume create "$vol" >/dev/null
for rank in 0 1; do
  addr="$job-node-0"; wait=""
  if [ $rank -gt 0 ]; then wait="echo 'Waiting for master node $addr:29500...'; for i in {1..60}; do   if timeout 1 bash -c 'cat < /dev/null > /dev/tcp/$addr/29500' 2>/dev/null; then     echo 'Master node is reachable'; break;   fi;   if [ \$i -eq 60 ]; then echo 'Timeout waiting for master node'; exit 1; fi;   sleep 2; done; "; fi
  cmd="${PIP}${wait}cat > /tmp/train.py << 'TRAINING_SCRIPT_EOF'
${SCRIPT}
TRAINING_SCRIPT_EOF
if command -v torchrun >/dev/null 2>&1; then   torchrun --nproc_per_node=1 --nnodes=2   --node-rank=$rank --rdzv-backend=static   --rdzv-endpoint=$addr:29500   /tmp/train.py; else   python /tmp/train.py; fi"
  docker run -d --name "$job-node-$rank" --network "$net" -w /workspace -v "$vol:/workspace" "$IMG" bash -lc "$cmd" >/dev/null
done
for n in 0 1; do
  if ! docker inspect "$job-node-$n" >/dev/null 2>&1; then echo "RESULT $id $job ERROR container node-$n did not start"; cleanup; exit 0; fi
done

t0=$(date +%s); fin=""; dumped=""
while true; do
  s0=$(docker inspect -f '{{.State.Running}}' "$job-node-0" 2>/dev/null); s1=$(docker inspect -f '{{.State.Running}}' "$job-node-1" 2>/dev/null)
  [ "$s0" != true ] && [ "$s1" != true ] && break
  now=$(date +%s)
  if [ -z "$fin" ] && docker logs "$job-node-0" 2>&1 | grep -q "Training is finished"; then fin=$now; fi
  # Something still running 90s after rank 0 finished training: capture state once,
  # then keep waiting to see whether it is merely slow or really stuck.
  if [ -n "$fin" ] && [ -z "$dumped" ] && [ $((now - fin)) -ge 90 ]; then dump "t90"; dumped=1; fi
  # Rank 0's torchrun exit barrier gives up after 300s; past 420s the run is a hang.
  if [ -n "$fin" ] && [ $((now - fin)) -ge 420 ]; then
    dump "t420"
    for n in 0 1; do docker logs "$job-node-$n" > "diag-$job/log-node-$n.txt" 2>&1; done
    echo "RESULT $id $job HANG still_running=node0:$s0,node1:$s1 total=$((now - t0))s"
    cleanup; exit 0
  fi
  if [ -z "$fin" ] && [ $((now - t0)) -ge 1200 ]; then
    mkdir -p "diag-$job"; for n in 0 1; do docker logs "$job-node-$n" > "diag-$job/log-node-$n.txt" 2>&1; done
    echo "RESULT $id $job NOFINISH total=$((now - t0))s"; cleanup; exit 0
  fi
  sleep 5
done
e0=$(docker inspect -f '{{.State.ExitCode}}' "$job-node-0"); e1=$(docker inspect -f '{{.State.ExitCode}}' "$job-node-1")
bar=$(docker logs "$job-node-0" 2>&1 | grep -c "exit barrier")
if [ "$e0" != 0 ] || [ "$e1" != 0 ]; then
  mkdir -p "diag-$job"; for n in 0 1; do docker logs "$job-node-$n" > "diag-$job/log-node-$n.txt" 2>&1; done
  echo "RESULT $id $job FAILED exit=$e0/$e1 barrierErr=$bar total=$(( $(date +%s) - t0 ))s"
elif [ -n "$dumped" ]; then
  for n in 0 1; do docker logs "$job-node-$n" > "diag-$job/log-node-$n.txt" 2>&1; done
  echo "RESULT $id $job SLOW exit=$e0/$e1 barrierErr=$bar total=$(( $(date +%s) - t0 ))s"
else
  echo "RESULT $id $job OK exit=$e0/$e1 barrierErr=$bar total=$(( $(date +%s) - t0 ))s"
fi
cleanup
