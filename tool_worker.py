"""Sandbox entry point. No API credentials or other run directories are mounted."""
import json
import os
from pathlib import Path
import traceback
from tool_results import start_progress, timeout_result


def main():
    os.chdir("/workspace")
    request = json.loads(Path("_tool_request.json").read_text())
    try:
        if request.get("operation") == "probe":
            status = Path("/proc/self/status").read_text()
            assert "NoNewPrivs:\t1" in status
            assert all(int(line.split()[1], 16) == 0 for line in status.splitlines()
                       if line.startswith(("CapEff:", "CapPrm:", "CapBnd:")))
            assert not Path("/.old_root").exists()
            assert not Path("/storage").exists()
            result = "isolated"
        else:
            import agent
            if request.get("operation") == "finalize":
                agent._finalize(Path("/workspace"), request.get("summary"), request["steps"])
                result = "finalized"
            else:
                start_progress(Path("/workspace"), request)
                result = agent.dispatch(agent.Tools(Path("/workspace")),
                                        request["tool"], request.get("args", {}))
                # Inner match guards may return an error before the outer sandbox deadline.
                try:
                    parsed = json.loads(result)
                except (ValueError, TypeError):
                    parsed = {}
                error = str(parsed.get("error", "")) if isinstance(parsed, dict) else ""
                if error and ("timed out" in error.lower() or "match exceeded" in error.lower()):
                    result = json.dumps(timeout_result(Path("/workspace"), request, error=error))
        response = {"result": result}
    except TimeoutError as exc:
        response = {"result": json.dumps(timeout_result(Path("/workspace"), request, error=str(exc)))}
    except Exception:
        response = {"error": traceback.format_exc()}
    Path("_tool_result.json").write_text(json.dumps(response))


if __name__ == "__main__":
    main()
