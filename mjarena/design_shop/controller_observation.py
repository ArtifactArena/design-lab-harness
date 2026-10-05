"""Construct validation observations from the actual processed robot."""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


def initial_controller_observation(processed_xml: str):
    from mjarena.envs.sumo import compose_sumo_model
    from mjarena.runner.episode import run_match
    root = Path(__file__).resolve().parents[2]
    names = [n.attrib['name'] for n in ET.fromstring(processed_xml).findall('actuator/*')]
    captured = []
    def capture(obs):
        captured.append(obs)
        return dict.fromkeys(names, 0.0)
    with tempfile.TemporaryDirectory(prefix='controller_observation_') as temporary:
        directory = Path(temporary)
        robot = directory/'robot.xml'; robot.write_text(processed_xml)
        composed = directory/'composed.xml'
        compose_sumo_model(str(root/'mjarena/assets/sumo_ring_env_studio_3d.xml'), str(robot),
                           str(root/'mjarena/core/assets/stationary_block_3d.xml'), str(composed),
                           randomize_spawn_3d=True, spawn_seed=0)
        run_match(composed_xml=composed, red_policy_py=capture, blue_policy_py=lambda obs: {},
                  out_dir=directory, max_steps=1, use_gui=False, save_video=False,
                  camera_mode='tracking', quiet=True, seed=0)
    if not captured:
        raise RuntimeError('Could not construct the controller observation')
    obs=captured[0]
    obs['max_t']=2000; obs['time_remaining']=20.0
    return obs
