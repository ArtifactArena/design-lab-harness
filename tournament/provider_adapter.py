"""Native high-effort requests with durable responses and explicit usage accounting."""
import hashlib,json,os,time
from pathlib import Path
import httpx

def client(config):
 key=Path(os.environ['PROVIDER_API_KEY_FILE']).read_text().strip()
 if config['provider']=='openai':
  from openai import OpenAI
  return OpenAI(api_key=key,timeout=120,max_retries=2)
 headers={'x-goog-api-key':key} if config['provider']=='google' else {'x-api-key':key,'anthropic-version':'2023-06-01'} if config['provider']=='anthropic' else {'Authorization':'Bearer '+key}
 headers['User-Agent']='ArtifactArena-iterative/23x2'
 return httpx.Client(headers=headers,timeout=httpx.Timeout(6000,connect=60))

def recover_unavailable_stream(directory,save,utc):
 """Retry only a terminal xAI unavailable error with no final-answer output."""
 stream=directory/'stream.jsonl'
 if not stream.exists():return False
 events=[json.loads(line) for line in stream.read_text().splitlines() if line.strip()]
 if not events:return False
 error=events[-1].get('error') or {}
 if error.get('type')!='server_error' or error.get('code')!='unavailable':return False
 if any(choice.get('delta',{}).get('content') for event in events for choice in event.get('choices',[])):return False
 attempts=directory/'failed_attempts';attempts.mkdir(exist_ok=True)
 number=len(list(attempts.glob('attempt_*')))+1
 dest=attempts/f'attempt_{number:02d}';dest.mkdir()
 for name in ('stream.jsonl','response_pending.json','request_settings.json'):
  source=directory/name
  if source.exists():source.rename(dest/name)
 save(dest/'failure.json',{'recorded_at':utc(),'provider':'xai','error':error,'final_answer_characters':0,'billing_usage_reported':False,'prompt_sha256':hashlib.sha256((directory/'prompt.txt').read_bytes()).hexdigest(),'attempt':number})
 if number>=3:
  save(directory/'retry_exhausted.json',{'at':utc(),'error':error,'attempts':number})
  raise RuntimeError('xAI unavailable after three accepted attempts; failed attempts preserved')
 print(utc(),f'xAI terminal unavailable error archived; retry {number}/2 with identical prompt',flush=True)
 return True

def obtain(client,directory,prompt,config,save,utc,status):
 completed=directory/'response.json'; pending=directory/'response_pending.json'
 if (directory/'retry_exhausted.json').exists():raise RuntimeError('xAI unavailable retry budget exhausted; see retry_exhausted.json')
 if completed.exists():return json.loads(completed.read_text())
 provider=config['provider'];start=time.time()
 if provider=='openai':
  if pending.exists():
   ident=json.loads(pending.read_text())['id']
   if not ident:raise RuntimeError('Ambiguous previous request; inspect before retry')
   response=client.responses.retrieve(ident)
  else:
   req={'model':config['model'],'input':prompt,'reasoning':{'effort':'high'},'max_output_tokens':config['max_output_tokens'],'service_tier':'default','background':True,'store':True}
   save(directory/'request_settings.json',{k:v for k,v in req.items() if k!='input'})
   save(pending,{'id':None,'created_at':utc()})
   response=client.responses.create(**req,extra_headers={'Idempotency-Key':config['run_id']+'-'+config['model']+'-'+directory.name+('-'+json.loads((directory/'request_retry.json').read_text())['id'] if (directory/'request_retry.json').exists() else '')})
   save(pending,{'id':response.id,'created_at':utc()})
  while response.status in ('queued','in_progress'):
   if time.time()-start>6000:client.responses.cancel(response.id);raise TimeoutError('100-minute API deadline')
   time.sleep(10);response=client.responses.retrieve(response.id)
  data=response.model_dump(mode='json');data['output_text']=response.output_text;data['completed_at']=time.time()
  save(completed,data);(directory/'response.txt').write_text(data['output_text']);return data
 if pending.exists():
  if provider!='xai' or not recover_unavailable_stream(directory,save,utc):raise RuntimeError('Interrupted accepted/ambiguous provider stream; inspect before retry to avoid duplicate billing')
 if provider=='google':
  url=f"https://generativelanguage.googleapis.com/v1beta/models/{config['model']}:streamGenerateContent?alt=sse"
  req={'contents':[{'role':'user','parts':[{'text':prompt}]}],'generationConfig':{'temperature':1.0,'maxOutputTokens':config['max_output_tokens'],'thinkingConfig':{'thinkingLevel':'HIGH'}}}
  settings={'model':config['model'],'generationConfig':req['generationConfig']}
 else:
  url={'xai':'https://api.x.ai/v1/chat/completions','together':'https://api.together.xyz/v1/chat/completions','anthropic':'https://api.anthropic.com/v1/messages'}[provider]
  req={'model':config['model'],'messages':[{'role':'user','content':prompt}],'max_tokens':config['max_output_tokens'],'stream':True}
  if provider=='xai':
   req.update(temperature=1.0,stream_options={'include_usage':True})
   if config.get('reasoning_effort_supported',True):req['reasoning_effort']='high'
  elif provider=='together':
   req.update(config.get('sampling_params',{}));req.update(config.get('provider_options',{}))
   req['stream_options']={'include_usage':True}
  else:
   req.update(output_config={'effort':'high'})
   if not config['model'].startswith('claude-fable-'):req['thinking']={'type':'adaptive'}
  settings={k:v for k,v in req.items() if k!='messages'}
 save(directory/'request_settings.json',settings);save(pending,{'id':None,'created_at':utc(),'provider':provider})
 chunks=[];usage={};stop=None;done=False;ident=None;retry_unavailable=False
 for attempt in range(3):
  with client.stream('POST',url,json=req) as response:
   if response.status_code==429 or response.status_code in (500,502,503,504,529):
    response.read()
    if attempt==2:raise RuntimeError(f'Provider HTTP {response.status_code} after three pre-stream attempts')
    time.sleep(20*(attempt+1));continue
   if not response.is_success:
    response.read();save(directory/'request_error.json',{'status':response.status_code,'body':response.text});raise RuntimeError(f'Provider HTTP {response.status_code}; see request_error.json')
   print(utc(),provider,'request accepted; high effort stream',flush=True)
   with (directory/'stream.jsonl').open('a',buffering=1) as log:
    for line in response.iter_lines():
     if time.time()-start>6000:raise TimeoutError('100-minute API deadline')
     if not line.startswith('data:'):continue
     raw=line[5:].strip()
     if raw=='[DONE]':done=True;continue
     event=json.loads(raw);log.write(json.dumps(event)+'\n')
     if event.get('type')=='error' or event.get('error'):
      error=event.get('error') or {}
      if provider=='xai' and not chunks and error.get('type')=='server_error' and error.get('code')=='unavailable':
       retry_unavailable=True;break
      raise RuntimeError('Provider stream error; see stream.jsonl')
     if provider in ('xai','together'):
      ident=event.get('id',ident)
      if event.get('usage'):usage=event['usage']
      for choice in event.get('choices',[]):
       if choice['delta'].get('content'):chunks.append(choice['delta']['content'])
       if choice.get('finish_reason'):stop=choice['finish_reason']
     elif provider=='google':
      ident=event.get('responseId',ident)
      if event.get('usageMetadata'):usage=event['usageMetadata']
      for candidate in event.get('candidates',[]):
       for part in candidate.get('content',{}).get('parts',[]):
        if part.get('text') and not part.get('thought'):chunks.append(part['text'])
       if candidate.get('finishReason'):stop=candidate['finishReason'];done=True
     else:
      typ=event.get('type')
      if typ=='message_start':ident=event['message']['id'];usage.update(event['message'].get('usage',{}))
      elif typ=='content_block_start' and event['content_block'].get('type')=='text':chunks.append(event['content_block'].get('text',''))
      elif typ=='content_block_delta' and event['delta']['type']=='text_delta':chunks.append(event['delta']['text'])
      elif typ=='message_delta':usage.update(event.get('usage',{}));stop=event['delta'].get('stop_reason',stop)
      elif typ=='message_stop':done=True
   break
 if retry_unavailable:
  status(status='running',revision=int(directory.name.rsplit('_',1)[1]),stage='xAI unavailable; retrying in 60 seconds')
  time.sleep(60)
  return obtain(client,directory,prompt,config,save,utc,status)
 if not done:raise RuntimeError('Provider stream ended without completion marker; partial events saved')
 if provider=='google':
  normalized={'input_tokens':usage.get('promptTokenCount',0),'output_tokens':usage.get('candidatesTokenCount',0)+usage.get('thoughtsTokenCount',0),'input_tokens_details':{'cached_tokens':usage.get('cachedContentTokenCount',0)},'output_tokens_details':{'reasoning_tokens':usage.get('thoughtsTokenCount',0)}}
 elif provider in ('xai','together'):
  normalized={'input_tokens':usage.get('prompt_tokens',0),'output_tokens':usage.get('completion_tokens',0)+((usage.get('completion_tokens_details') or {}).get('reasoning_tokens',0) if provider=='xai' else 0),'input_tokens_details':usage.get('prompt_tokens_details',{}),'output_tokens_details':usage.get('completion_tokens_details',{})}
 else:
  read=usage.get('cache_read_input_tokens',0);write=usage.get('cache_creation_input_tokens',0)
  normalized={'input_tokens':usage.get('input_tokens',0)+read+write,'output_tokens':usage.get('output_tokens',0),'input_tokens_details':{'cached_tokens':read,'cache_write_tokens':write}}
 data={'id':ident,'model':config['model'],'provider':provider,'status':'completed' if stop in ('stop','end_turn','STOP') else 'incomplete','stop_reason':stop,'created_at':start,'completed_at':time.time(),'output_text':''.join(chunks),'usage':normalized,'provider_usage':usage}
 save(completed,data);(directory/'response.txt').write_text(data['output_text']);return data

def usage_summary(response,config):
 u=response.get('usage') or {};d=u.get('input_tokens_details') or {}
 i=int(u.get('input_tokens',0));o=int(u.get('output_tokens',0));c=int(d.get('cached_tokens',0));w=int(d.get('cache_write_tokens',0))
 a,b,z=[config['pricing'][k] for k in ('input','cache_read','output')]
 threshold=config.get('long_context_threshold')
 if threshold and i>threshold:
  a*=2;b*=2;z*=2 if config['provider']=='xai' else 1.5
 cost=((i-c-w)*a+c*b+w*a*1.25+o*z)/1e6
 if config['provider']=='xai' and response.get('provider_usage',{}).get('cost_in_usd_ticks') is not None:cost=response['provider_usage']['cost_in_usd_ticks']/1e10
 return {'input_tokens':i,'cached_input_tokens':c,'cache_write_tokens':w,'output_tokens':o,'reasoning_tokens':(u.get('output_tokens_details') or {}).get('reasoning_tokens',0),'estimated_cost_usd':cost,'raw_usage':response.get('provider_usage',u)}
