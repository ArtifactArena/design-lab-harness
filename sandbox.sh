#!/bin/bash
set -euo pipefail
# Called by linux_sandbox.py with paths prepared by the trusted API driver.
RUN_ROOT="$1"
WORKSPACE="$2"
ENV_ROOT="$3"
MODE=tool_worker
SANDBOX_ROOT="$RUN_ROOT/rootfs"
mkdir -p "$SANDBOX_ROOT"/{usr,bin,lib,lib64,etc,proc,dev,dev/shm,tmp,workspace,engine,code,venv}
exec unshare --user --map-root-user --mount --pid --fork --net --kill-child bash -s -- "$SANDBOX_ROOT" "$RUN_ROOT" "$ENV_ROOT" "$WORKSPACE" "$MODE" <<'INNER'
set -euo pipefail
SANDBOX_ROOT="$1"; RUN_ROOT="$2"; ENV_ROOT="$3"; WORKSPACE="$4"; MODE="$5"
mount --make-rprivate /
mount --bind "$SANDBOX_ROOT" "$SANDBOX_ROOT"
printf 'root:x:0:0:Run workspace:/workspace:/bin/bash\n' > "$SANDBOX_ROOT/etc/passwd"
printf 'root:x:0:\n' > "$SANDBOX_ROOT/etc/group"
for d in usr bin lib lib64; do
 mount --bind "/$d" "$SANDBOX_ROOT/$d"
 mount -o remount,bind,ro "$SANDBOX_ROOT/$d"
done
for pair in "$RUN_ROOT/engine:engine" "$RUN_ROOT/code:code" "$ENV_ROOT:venv"; do
 src=${pair%:*};dst=${pair##*:}
 mount --bind "$src" "$SANDBOX_ROOT/$dst"
 mount -o remount,bind,ro "$SANDBOX_ROOT/$dst"
done
mount --bind "$WORKSPACE" "$SANDBOX_ROOT/workspace"
mount -t proc proc "$SANDBOX_ROOT/proc"
mount -t tmpfs -o size=4g tmpfs "$SANDBOX_ROOT/tmp"
mount -t tmpfs -o size=1g tmpfs "$SANDBOX_ROOT/dev/shm"
for dev in null zero random urandom; do
 touch "$SANDBOX_ROOT/dev/$dev"
 mount --bind "/dev/$dev" "$SANDBOX_ROOT/dev/$dev"
done
for f in ld.so.cache localtime; do
 if [ -f "/etc/$f" ]; then
  touch "$SANDBOX_ROOT/etc/$f"
  mount --bind "/etc/$f" "$SANDBOX_ROOT/etc/$f"
  mount -o remount,bind,ro "$SANDBOX_ROOT/etc/$f"
 fi
done
# Detach the host filesystem before running model-authored code. A chroot alone
# is insufficient when a process can create a nested user namespace.
cd "$SANDBOX_ROOT"
mkdir -p .old_root
pivot_root . .old_root
cd /
/usr/bin/umount -l /.old_root
rmdir /.old_root
# Drop namespace-root privileges, including mount/chroot, and prevent exec from
# regaining privileges. The provider/API process stays outside this sandbox.
exec /usr/bin/setpriv --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs /usr/bin/env -i PATH=/venv/bin:/usr/bin:/bin LANG=C.UTF-8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=/code:/engine ARENA_ENGINE=/engine ARENA_REPO_ROOT=/engine XDG_CACHE_HOME=/workspace/.cache DSPY_CACHEDIR=/workspace/.cache/dspy MPLCONFIGDIR=/workspace/.cache/matplotlib TMPDIR=/tmp /venv/bin/python -u "/code/$MODE.py"
INNER
