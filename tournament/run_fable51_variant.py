"""Durable 10-call native Responses driver for the updated open-ended tool harness."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parent;WORKSPACE=ROOT/'workspace'
SOURCE=ROOT.parent/'source'
os.environ['ARENA_REPO_ROOT']=str(SOURCE)
sys.path.insert(0,str(SOURCE))
import agent
import linux_sandbox
import provider_adapter
from slots import slot
import monitoring
CONFIG=json.loads((ROOT/'config.json').read_text())

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def status(**data):
 previous=json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {}
 if 'revision' in data:data['turn']=data.pop('revision')
 bots=agent.harness_lib.list_bots(WORKSPACE)
 save(ROOT/'status.json',{**previous,'updated_at':utc(),'model':CONFIG['model'],'total_turns':10,'bots_produced':len(bots),**({'error':None,'error_detail':None} if data.get('status') in ('running','queued','completed') else {}),**data})
def sandbox(request,timeout=3700):
 with slot(ROOT.parent/'tool_slots','tool-',6,lambda:status(status='queued',stage='waiting for lab capacity')):
  status(status='running',stage='tools')
  return linux_sandbox.execute(ROOT/'workspace-runtime',WORKSPACE,request,timeout=timeout)

def build_prompt(turn,journal,last_result,latest_qual,latest_fp):
 import tiktoken
 enc=tiktoken.get_encoding('o200k_base')
 system=agent.build_system_prompt(ROOT/'workspace-runtime/prompt',step=turn,max_steps=10)
 state=agent.build_state(WORKSPACE,journal,last_result,latest_qual,latest_fp,result_budget=160000)
 state=state.replace('== YOUR CURRENT DRAFT (the submitted bot) ==','== YOUR CURRENT DRAFT ==')
 prompt=system+'\n\n'+state
 if len(enc.encode(prompt,disallowed_special=()))>100000:
  state=agent.build_state(WORKSPACE,journal,last_result,latest_qual,latest_fp,result_budget=24000).replace('== YOUR CURRENT DRAFT (the submitted bot) ==','== YOUR CURRENT DRAFT ==')
  prompt=system+'\n\n'+state
 assert len(enc.encode(prompt,disallowed_special=()))<=100000,'Fixed prompt state exceeds input budget; inspect before truncating further'
 return prompt

def run():
 assert json.loads((ROOT/'prepared.json').read_text())['isolation_verified']
 assert linux_sandbox.execute(ROOT/'workspace-runtime',WORKSPACE,{'operation':'probe'},timeout=60)=='isolated'
 client=provider_adapter.client(CONFIG);journal=[];last_result='No previous actions, bot, or feedback.';latest_qual=None;latest_fp=None;finished=False;total_cost=0
 for turn in range(1,11):
  folder=ROOT/'turns'/f'turn_{turn:02d}';folder.mkdir(parents=True,exist_ok=True)
  checkpoint=folder/'checkpoint.json'
  if checkpoint.exists():
   saved=json.loads(checkpoint.read_text());journal=saved['journal'];last_result=saved['last_result'];latest_qual=saved['latest_qual'];latest_fp=saved['latest_fp'];finished=saved['finished'];total_cost=saved['total_cost']
   if finished:break
   continue
  prompt_file=folder/'prompt.txt'
  if prompt_file.exists():prompt=prompt_file.read_text()
  else:prompt=build_prompt(turn,journal,last_result,latest_qual,latest_fp);prompt_file.write_text(prompt)
  status(status='running',stage='model',turn=turn,completed_turns=turn-1,cost_usd=total_cost)
  capacity={'openai':8,'anthropic':4,'google':4,'xai':3,'together':4}[CONFIG['provider']]
  with slot(ROOT.parent.parent/'iterative-23x2-20260924/provider_slots',CONFIG['provider']+'-',capacity,lambda:status(status='queued',stage='waiting for provider slot')):
   status(status='running',stage='model')
   response=provider_adapter.obtain(client,folder,prompt,CONFIG,save,utc,status)
  usage=provider_adapter.usage_summary(response,CONFIG);save(folder/'usage.json',usage);total_cost+=usage['estimated_cost_usd']
  status(cost_usd=total_cost)
  actions,recovery=agent.response_actions(response);blocks=[]
  if recovery:
   notice=recovery['notice']+f"\nFailed turn: {turn}/10. Configured output limit: {CONFIG['max_output_tokens']} tokens."
   save(folder/'response_recovery.json',{'turn':turn,'counted_toward_budget':True,'tools_executed':0,**recovery})
   journal.append({'step':turn,'tool':'parse_error','args':{},'note':'','outcome':recovery['kind'],'reason':recovery['reason']})
   if last_result:blocks.append(last_result)
   blocks.append(notice)
   print(utc(),f"Turn {turn}/10 consumed: {recovery['kind']}; continuing within the 10-call budget",flush=True)
  for index,(tool,args,note) in enumerate(actions,1):
   status(status='running',stage='tools',turn=turn,completed_turns=turn-1,tool=tool,tool_index=index,tool_count=len(actions),cost_usd=total_cost)
   tool_file=folder/f'tool_{index:03d}.json'
   if tool_file.exists():
    record=json.loads(tool_file.read_text());result=record['result']
    if tool=='finish':finished=result.startswith('Finished;')
   elif tool=='finish':
    finished=bool(agent.harness_lib.list_bots(WORKSPACE))
    result='Finished; selection will use the round robin.' if finished else 'ERROR: record at least one complete bot with save_bot before finishing.'
    save(tool_file,{'tool':tool,'args':args,'note':note,'result':result})
   else:
    marker=folder/f'tool_{index:03d}.pending'
    if marker.exists():raise RuntimeError(f'Ambiguous interrupted tool {turn}/{index}; inspect before rerunning')
    marker.write_text(utc())
    started=time.time()
    try:
     result=sandbox({'tool':tool,'args':args})
    except TimeoutError as exc:
     from tool_results import timeout_result
     result=json.dumps(timeout_result(WORKSPACE,{'tool':tool,'args':args},error=str(exc),started_at=started))
    save(tool_file,{'tool':tool,'args':args,'note':note,'result':result,'wall_seconds':time.time()-started})
   if tool=='save_bot':
    try:monitoring.capture(ROOT,turn,index,args,CONFIG)
    except Exception as exc:save(folder/f'monitoring_{index:03d}_error.json',{'error':str(exc)})
   canonical=agent._normalize_args(tool,dict(args));outcome=agent._summarize(tool,result)
   journal.append({'step':turn,'tool':tool,'args':canonical,'note':note,'outcome':outcome})
   if tool=='does_bot_qualify' and canonical.get('ref','draft') in ['draft','current']:
    latest_qual=agent._qual_status(result);latest_fp=agent._draft_fp(WORKSPACE)
   blocks.append(f'[{tool}] → {outcome}\n{result}')
   if tool=='finish' and finished:break
  last_result='\n\n'.join(blocks)
  saved={'journal':journal,'last_result':last_result,'latest_qual':latest_qual,'latest_fp':latest_fp,'finished':finished,'total_cost':total_cost}
  save(checkpoint,saved);save(ROOT/'journal.json',journal)
  # Audit every draft at turn end without adding unsaved drafts to selection.
  for name in ['robot.xml','controller.py','notes.md']:
   (folder/name).write_text(agent._read_if(WORKSPACE/name))
  status(status='running',stage='turn complete',turn=turn,completed_turns=turn,cost_usd=total_cost)
  print(utc(),f'Turn {turn}/10 complete; {len(actions)} tool calls',flush=True)
  if finished:break
 status(status='running',stage='selection tournament',turn=turn,completed_turns=turn,cost_usd=total_cost)
 candidates=monitoring.freeze_selection(ROOT,turn)
 if not candidates:raise RuntimeError('No complete bot saved; selection has no candidates')
 save(ROOT/'generation_complete.json',{'turns':turn,'candidates':candidates,'finished_at':utc()})
 # The external selection coordinator sees only frozen candidates. Its results
 # remain outside the model workspace and never re-enter the prompt.
 while not (ROOT/'selection.json').exists():time.sleep(15)
 final=json.loads((ROOT/'selection.json').read_text())
 if final.get('error'):raise RuntimeError(final['error'])
 status(status='completed',stage='finished',completed_turns=turn,cost_usd=total_cost,selected_bot=final['name'],selection=final)

if __name__=='__main__':
 with (ROOT/'run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:run()
  except Exception as exc:
   old=json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {}
   status(status='failed',stage=old.get('stage'),turn=old.get('turn'),completed_turns=old.get('completed_turns',0),cost_usd=old.get('cost_usd',0),error=str(exc))
   traceback.print_exc();raise
