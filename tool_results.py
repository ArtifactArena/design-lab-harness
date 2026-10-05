"""Durable, score-free tool progress and recoverable timeout results."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
import uuid


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value))
    temp.replace(path)


def start_progress(workspace, request):
    workspace = Path(workspace)
    relative = request.get('_progress_dir') or ('_tool_progress/' + uuid.uuid4().hex)
    progress = workspace / relative
    if not progress.resolve().is_relative_to(workspace.resolve()):
        raise ValueError('Tool progress must stay inside the workspace')
    progress.mkdir(parents=True, exist_ok=True)
    write_json(workspace / '_tool_progress_current.json', {
        'path': relative, 'started_at': time.time(),
        'tool': request.get('tool'), 'args': request.get('args', {}),
    })
    os.environ['MH_TOOL_PROGRESS_DIR'] = str(progress)


def checkpoint_seed(record, match_dir):
    """Write a completed seed immediately; exclude monitoring/combat scores."""
    directory = os.environ.get('MH_TOOL_PROGRESS_DIR')
    if not directory:
        return
    get = record.get if isinstance(record, dict) else lambda k: getattr(record, k, None)
    result = {key: get(key) for key in ('seed', 'winner', 'termination_reason', 'num_steps')}
    result.update(match_dir=str(match_dir), complete=True)
    write_json(Path(directory) / ('seed-' + uuid.uuid4().hex + '.json'), result)


def timeout_result(workspace, request, timeout_seconds=None, error=None, log_tail='', started_at=0):
    workspace = Path(workspace).resolve()
    completed = []
    progress = None
    relative = request.get('_progress_dir')
    if relative:
        progress = workspace / relative
    else:
        marker = workspace / '_tool_progress_current.json'
        try:
            if marker.resolve().is_relative_to(workspace):
                state = json.loads(marker.read_text())
                if (state.get('tool') == request.get('tool') and state.get('args') == request.get('args', {})
                        and state.get('started_at', 0) >= started_at):
                    progress = workspace / state['path']
        except (OSError, ValueError, KeyError):
            pass
    if progress and progress.resolve().is_relative_to(workspace):
        for file in sorted(progress.glob('seed-*.json')):
            try:
                # Regular files only: a model-made FIFO would block this host-side read.
                if file.resolve().is_relative_to(workspace) and file.is_file():
                    data = json.loads(file.read_text())
                    if data.get('complete') is True:
                        completed.append({key: data.get(key) for key in
                            ('seed', 'winner', 'termination_reason', 'num_steps', 'match_dir', 'complete')})
            except (OSError, ValueError, AttributeError):
                continue
    result = {
        'ok': False, 'timed_out': True, 'tool': request.get('tool'),
        'error': error or f'Tool exceeded {timeout_seconds} seconds and hit its wall-clock timeout.',
        'timeout_seconds': timeout_seconds,
        'partial_results': {'completed_seed_count': len(completed), 'completed_seeds': completed},
        'feedback': 'The tool took too long and hit a timeout. Only the completed results listed here are available; '
                    'unfinished trials have no result and must not be counted as wins, losses, or draws. '
                    'No overall match or qualification verdict is available. Earlier tool results remain valid. '
                    'The run will continue. Inspect the current files before retrying: edits made before the timeout may persist.',
    }
    args = request.get('args') or {}
    requested = args.get('n_seeds')
    if isinstance(requested, int) and requested >= len(completed):
        result['partial_results'].update(requested_seed_count=requested, unfinished_seed_count=requested-len(completed))
    if not completed:
        result['partial_results']['unavailable_reason'] = 'No completed seed results were checkpointed for this tool invocation.'
    if log_tail:
        result['partial_output'] = log_tail[-8000:]
    return result
