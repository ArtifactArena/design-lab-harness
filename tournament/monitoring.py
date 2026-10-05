"""Host-side monitoring; none of this module or its rewards enters model inputs."""
from pathlib import Path
import json,hashlib,math,shutil,datetime

def read(p,default=None):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return default

def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)

def safe_file(p,ws):
 if not p.resolve().is_relative_to(ws.resolve()):raise ValueError('Artifact escaped its workspace')
 return p.is_file()

def score(record):
 n=record['num_steps'];contacts=record['contacts'][:n];contact=any(contacts);first=next((i for i,v in enumerate(contacts) if v),None)
 reason=record['termination_reason'];t=n*record['control_dt'];win=int(contact and record['winner']=='red' and reason=='ring_out')
 edge=record['ring_radius']-math.hypot(*record['initial_blue_pos'][:2]);edges=[record['ring_radius']-math.hypot(*p[:2]) for p in record['blue_positions'][:n]]
 clip=lambda v:max(0.,min(1.,v))
 p=1. if win else clip((edge-min(edges[first:]))/edge) if contact and edge>0 else 0.
 gaps=record.get('red_surface_distances',[])[:n]
 if not gaps or len(gaps)!=n:raise ValueError('Missing full-rate surface-distance recording')
 a=1. if contact else clip((gaps[0]-min(gaps))/gaps[0]) if gaps[0]>0 else 1.
 penalty=None
 if record.get('controller_errors') or reason=='forfeit_crash':penalty=('controller_failure',-1.)
 elif record.get('physics_unstable') or reason=='qacc':penalty=('physical_instability',-.5)
 elif reason=='size_violation' and record['winner']=='blue':penalty=('invalid_morphology',-.8)
 elif record['winner']=='blue' and reason in ('ring_out','inactivity'):penalty=('loss_'+reason,-.5)
 raw=4*win+win*(1-t/20)+1.5*p+.5*a
 return {'seed':record['seed'],'W':win,'t_seconds':t,'P':p,'A':a,'contacted':contact,'raw_reward':raw,'reward':penalty[1] if penalty else raw,'penalty':penalty[0] if penalty else None,'termination_reason':reason}

def capture(root,turn,index,args,config):
 name=args.get('name',args.get('bot_name',''));ws=root/'workspace';folder=ws/'bots'/name
 if not isinstance(name,str) or not name or '/' in name or '\\' in name or '..' in name:return None
 artifact_file=folder/'bot_artifact.json'
 if not safe_file(artifact_file,ws):return None
 art=read(artifact_file,{});xml=folder/'robot.xml';ctrl=folder/'controller.py'
 if not all(safe_file(f,ws) for f in (xml,ctrl)):return None
 digest=hashlib.sha256(xml.read_bytes()+b'\0'+ctrl.read_bytes()).hexdigest();ident=hashlib.sha256((name+'\0'+digest).encode()).hexdigest()[:16]
 dest=root/'monitoring/bots'/ident;meta=dest/'metadata.json'
 if meta.exists():return read(meta)
 dest.mkdir(parents=True,exist_ok=True)
 for p in (artifact_file,xml,ctrl):shutil.copy2(p,dest/p.name)
 work=ws/'_work'/('save_'+name);record_file=work/'refinement/commit_0/qualification/match_data.json';composed=work/'qualification/composed.xml'
 matches=[];error=None;override=None;value=None
 if safe_file(record_file,ws) and safe_file(composed,ws):
  shutil.copy2(record_file,dest/'match_data.json');shutil.copy2(composed,dest/'composed.xml')
  records=read(dest/'match_data.json',{})
  try:
   matches=[score(v) for v in records.values()]
   if any(m['penalty']=='controller_failure' for m in matches):value=-1.;override='controller_failure_across_trials'
   elif any(m['penalty']=='physical_instability' for m in matches):value=-.5;override='physical_instability_across_trials'
   else:value=sum(m['reward'] for m in matches)/len(matches)
  except (ValueError,KeyError,ZeroDivisionError) as e:error=str(e)
 elif not art.get('validation_passed'):
  morph_passed=safe_file(work/'qual_robot.xml',ws) or not xml.read_text().strip() or not ctrl.read_text().strip()
  override='invalid_output_or_controller' if morph_passed else 'invalid_morphology';value=-1. if morph_passed else -.8
 else:error='Qualification recording unavailable'
 payload={'name':name,'asset_id':ident,'turn':turn,'call':index,'source_sha256':digest,'reward':value,'override':override,'matches':matches,'error':error,'monitoring_only':True,'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 save(meta,payload)
 return payload

def freeze_selection(root,turn):
 ws=root/'workspace';candidates=[]
 for i,p in enumerate(sorted((ws/'bots').glob('*/bot_artifact.json'))):
  if not safe_file(p,ws):continue
  row=capture(root,turn,i,{'name':p.parent.name},{})
  if row:candidates.append(row)
 return candidates
