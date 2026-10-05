#!/usr/bin/env python3
"""Exercise the model's actual Linux tool boundary without making an API call."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import agent
import linux_sandbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics", action="store_true")
    args = parser.parse_args()
    workspace = agent.new_run_dir()
    os.environ["HARNESS_ISOLATION_CANARY"] = "must_not_reach_tools"
    runtime = linux_sandbox.prepare(workspace)
    probe = '''import os, pathlib, socket
assert not pathlib.Path('/storage').exists()
assert not pathlib.Path('/.old_root').exists()
assert 'HARNESS_ISOLATION_CANARY' not in os.environ
assert not any(k.endswith('API_KEY') for k in os.environ)
try:
    os.chroot('/')
except PermissionError:
    pass
else:
    raise AssertionError('chroot capability retained')
try:
    socket.create_connection(('1.1.1.1', 443), timeout=1)
except OSError:
    pass
else:
    raise AssertionError('network is reachable')
print('ISOLATION_OK')
'''
    result = linux_sandbox.execute(runtime, workspace, {"tool": "run_python", "args": {"code": probe}})
    assert "ISOLATION_OK" in result, result
    checks = {"isolation": True}
    if args.physics:
        xml = '''<mujoco><worldbody><body name="root" pos="0 0 0.3">
<freejoint/><geom name="chassis" type="box" size="0.2 0.2 0.2" material="plastic"/>
<body name="rotor" pos="0 0 0.25"><joint name="hinge" type="hinge" axis="0 0 1"/>
<geom name="rotor_geom" type="cylinder" size="0.1 0.03" material="plastic"/>
</body></body></worldbody><actuator><motor name="motor" joint="hinge" gear="1"/></actuator></mujoco>'''
        controller = "def policy_step(obs):\n    return {'motor': 0.0}\n"
        for name, text in [("robot.xml", xml), ("controller.py", controller)]:
            result = linux_sandbox.execute(runtime, workspace, {
                "tool": "write_file", "args": {"path": name, "content": text}})
            assert result.startswith("wrote"), result
        result = json.loads(linux_sandbox.execute(runtime, workspace, {
            "tool": "does_bot_qualify", "args": {"ref": "draft"}}))
        assert result.get("ok") and result.get("validation_passed"), result
        assert result.get("qualification_passed") is False, result
        assert "inactiv" in result["feedback"].lower(), result
        checks["three_seed_qualification"] = True
    (runtime / "isolation_check.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps({"workspace": str(workspace), "checks": checks}))


if __name__ == "__main__":
    main()
