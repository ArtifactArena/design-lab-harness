"""Score-free seed telemetry through the engine's existing progress callback."""
import json
import os
from pathlib import Path
import time
import uuid


def live_seed_callback(callback, *, seed, match_time, max_steps, match_dir):
    directory = os.environ.get('MH_TOOL_PROGRESS_DIR') or os.environ.get('MH_LIVE_MATCH_DIR')
    if not directory:
        return callback
    path = Path(directory) / ('live-' + uuid.uuid4().hex + '.json')
    started = time.time()
    last_write = 0

    def report(event):
        nonlocal last_write
        if callback:
            callback(event)
        now = time.time()
        complete = event.get('type') == 'done'
        if last_write and now - last_write < 5 and not complete:
            return
        steps = event.get('max_steps') or max_steps
        limit = match_time if isinstance(match_time, (float, int)) and match_time > 0 else None
        row = dict(pid=os.getpid(), seed=seed, started_at=started, updated_at=now,
                   sampled_at=now, limit_seconds=limit,
                   sim_seconds=event.get('step', 0) * limit / steps if limit and steps else None,
                   step=event.get('step'), max_steps=steps, complete=complete,
                   match_dir=str(match_dir))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(row))
            temporary.replace(path)
            last_write = now
        except OSError:
            pass  # Monitoring failure must not affect the simulation or its result.

    return report
