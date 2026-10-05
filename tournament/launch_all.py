from pathlib import Path
import json,subprocess,os,time,hashlib
B=Path(__file__).resolve().parent
assert os.uname().nodename.split('.')[0]=='host_a'
assert json.loads((B/'physics_smoke_passed.json').read_text())['passed']
assert (B/'tests_passed.json').exists()
registry=json.loads((B/'registry.json').read_text());assert len(registry)==46
for r in registry:
 root=Path(r['path']);assert json.loads((root/'prepared.json').read_text())['isolation_verified']
 assert not (root/'launch_manifest.json').exists(),r['id']
 assert subprocess.run(['tmux','has-session','-t','='+r['session']],capture_output=True).returncode!=0
manifest=[]
for r in registry:
 root=Path(r['path']);subprocess.run(['tmux','new-session','-d','-s',r['session'],'-c',str(root),'bash launch.sh'],check=True)
 pid=int(subprocess.check_output(['tmux','list-panes','-t','='+r['session'],'-F','#{pane_pid}'],text=True).strip())
 entry={**r,'pid':pid,'host':os.uname().nodename,'launched_at':time.time(),'proc_start':Path(f'/proc/{pid}/stat').read_text().split()[21],'driver_sha256':hashlib.sha256((root/'run.py').read_bytes()).hexdigest(),'source_manifest':str(B/'source_manifest.json')}
 (root/'launch_manifest.json').write_text(json.dumps(entry,indent=2));manifest.append(entry)
 (B/'launch_manifest.json').write_text(json.dumps(manifest,indent=2));print(r['id'],pid,flush=True)
print('LAUNCHED',len(manifest),flush=True)
