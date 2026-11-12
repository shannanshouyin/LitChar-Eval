"""Matched stability follow-up; derived from the tested formal runner, isolated outputs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from collections import Counter
from threading import Lock
import argparse
import hashlib
import json
import os
import random
import sys
import time
from dotenv import dotenv_values
from openai import OpenAI
from .core import build_messages, digest

ROOT=Path(__file__).resolve().parents[1]
TERMINAL={'response','refusal','transport_exhausted'}
def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
from .io_utils import resilient_atomic_json as atomic_json

def classify_error(status,code,message):
    text=(str(code)+' '+str(message)).lower()
    if str(code)=='1301' or any(x in text for x in ('content policy','content_filter','sensitive content','不安全','敏感内容')):
        return 'refusal',False,False
    if status==402 or str(code) in ('1113','1120') or any(x in text for x in ('insufficient balance','insufficient_quota','余额不足')):
        return 'billing',False,True
    if status in (401,403) or str(code) in ('1000','1001','1002','1003','1004'):
        return 'authentication_or_permission',False,True
    if status==429 or str(code)=='1302':return 'rate_limit',True,False
    if status is None or status in (408,409) or status>=500:return 'transport_or_server',True,False
    return 'request_configuration',False,True

def make_jobs(variants,api,protocol):
    jobs=[]
    for v in variants:
        for provider in protocol['providers']:
            for condition in v['conditions']:
                body={'model':api['models'][provider]['model'],'messages':build_messages(v['item'],condition),
                      'temperature':protocol['temperature'],'top_p':protocol['top_p'],
                      'max_tokens':protocol['max_tokens'],'response_format':protocol['response_format'],
                      'extra_body':{'thinking':{'type':protocol['thinking']}}}
                jobs.append({'job_id':f"{v['variant_id']}__{provider}__{condition}",'variant_id':v['variant_id'],
                             'sample_id':v['sample_id'],'kind':v['kind'],'replicate':v['replicate'],
                             'provider':provider,'condition':condition,'body':body,'request_sha256':digest(body)})
    return jobs

def reserve_cost(job,protocol):
    rates=protocol['budget_rates_per_million'][job['provider']]
    upper_input=sum(len(m['content'].encode('utf-8'))+128 for m in job['body']['messages'])
    return (upper_input*rates['input']+job['body']['max_tokens']*rates['output'])/1e6

def charge_cost(job,result,protocol,reservation):
    usage=(result.get('response') or {}).get('usage') or {}
    if not all(isinstance(usage.get(k),int) and usage[k]>=0 for k in ('prompt_tokens','completion_tokens')):
        return reservation,False
    rates=protocol['budget_rates_per_million'][job['provider']]
    return (usage['prompt_tokens']*rates['input']+usage['completion_tokens']*rates['output'])/1e6,True

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    parser.add_argument('--max-new-attempts',type=int,default=1600);args=parser.parse_args()
    api=read(ROOT/'config/api_config.json');protocol=read(ROOT/'config/matched_protocol.json')
    assert 1<=args.max_new_attempts<=protocol['maximum_total_attempts']
    variants=read(ROOT/'data/processed/matched_variants.json');jobs=make_jobs(variants,api,protocol)
    assert len({j['job_id'] for j in jobs})==len(jobs)
    output=ROOT/'results/matched_v1';output.mkdir(parents=True,exist_ok=True)
    sources=[str(p.relative_to(ROOT)).replace(chr(92),'/') for p in sorted((ROOT/'src').glob('*.py'))]
    frozen={'protocol':protocol,'api_models':api['models'],'variants_sha256':digest(variants),
            'gold_sha256':digest(read(ROOT/'data/processed/matched120_gold.json')),
            'requests_sha256':digest([(j['job_id'],j['request_sha256']) for j in jobs]),
            'source_sha256':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
            'python':sys.version,'executable':sys.executable}
    manifest=output/'frozen_manifest.json'
    if manifest.exists():
        if read(manifest)['frozen']!=frozen:raise RuntimeError('Frozen formal source/config/input changed')
    else:atomic_json(manifest,{'created_utc':now(),'frozen':frozen})
    log=output/'attempts.jsonl'
    logs=[json.loads(l) for l in log.read_text(encoding='utf-8').splitlines()] if log.exists() else []
    starts={r['attempt_id']:r for r in logs if r['event']=='start'}
    ends={r['attempt_id']:r for r in logs if r['event']=='end'}
    if starts.keys()!=ends.keys():raise RuntimeError('Unresolved request; audit before resume to avoid duplicate charges')
    done={r['job_id'] for r in ends.values() if r['status'] in TERMINAL}
    if len(done)!=sum(r['status'] in TERMINAL for r in ends.values()):raise RuntimeError('Duplicate terminal response')
    counts=Counter(r['job_id'] for r in starts.values())
    spent=sum(r['estimated_charge_cny'] for r in ends.values());used=len(starts)
    pending_jobs=[j for j in jobs if j['job_id'] not in done]
    random.Random(protocol['order_seed']).shuffle(pending_jobs)
    print(json.dumps({'mode':'execute' if args.execute else 'dry_run','planned_jobs':len(jobs),'pending':len(pending_jobs),
                      'estimated_cost_cny':spent,'max_new_attempts':args.max_new_attempts}),flush=True)
    if not args.execute:return
    lock_path=output/'RUNNING.lock'
    with lock_path.open('x',encoding='utf-8') as handle:handle.write(str(os.getpid()))
    mutex=Lock();new_attempts=0;reserved=0.;halt={};clients={};consecutive=Counter();active=Counter()
    env=dotenv_values((ROOT/api['private_env_file']).resolve())
    secrets=[v for k,v in env.items() if v and any(s in k for s in ('KEY','TOKEN','SECRET'))]
    def redact(value):
        text=str(value)
        for secret in secrets:text=text.replace(secret,'[REDACTED]')
        return text
    def append(record):
        with log.open('a',encoding='utf-8') as handle:
            handle.write(redact(json.dumps(record,ensure_ascii=False))+'\n');handle.flush();os.fsync(handle.fileno())
    def status():
        value={'updated_utc':now(),'planned_jobs':len(jobs),'terminal_jobs':len(done),'attempts':used+new_attempts,
               'new_attempts':new_attempts,'estimated_cost_cny':round(spent,6),'reserved_cny':round(reserved,6),
               'estimated_cost_limit_cny':protocol['estimated_cost_stop_cny'],'halts':dict(halt),
               'all_jobs_terminal':len(done)==len(jobs),'pricing_note':protocol['budget_note']}
        atomic_json(output/'progress.json',value);return value
    try:
        for provider in protocol['providers']:
            model=api['models'][provider];url=env[model['base_url_env']] if 'base_url_env' in model else model['base_url']
            if url!=model.get('expected_base_url',model.get('base_url')):raise ValueError('Unexpected endpoint')
            if not env.get(model['key_env']):raise ValueError(f'Missing API key: {provider}')
            clients[provider]=OpenAI(api_key=env[model['key_env']],base_url=url,timeout=protocol['timeout_seconds'],max_retries=0)
        def run(job):
            nonlocal new_attempts,spent,reserved
            provider=job['provider'];jid=job['job_id']
            for _ in range(protocol['max_attempts_per_job']-counts[jid]):
                reserve=reserve_cost(job,protocol)
                with mutex:
                    if halt or new_attempts>=args.max_new_attempts:return
                    if used+new_attempts>=protocol['maximum_total_attempts']:halt['attempt_limit']='reached';return
                    if spent+reserved+reserve>protocol['estimated_cost_stop_cny']:halt['estimated_budget']='reached';return
                    new_attempts+=1;counts[jid]+=1;aid=f'{jid}__{counts[jid]}';reserved+=reserve
                    append({'event':'start','utc':now(),'attempt_id':aid,'reserved_cny':reserve,**job})
                start=time.monotonic();retry=False;fatal=False
                try:
                    response=clients[provider].chat.completions.create(**job['body'])
                    result={'status':'response','response':response.model_dump(mode='json')}
                except Exception as exc:
                    http=getattr(exc,'status_code',None);body=getattr(exc,'body',{}) or {}
                    error=body.get('error',body) if isinstance(body,dict) else {}
                    code=error.get('code') if isinstance(error,dict) else None
                    message=error.get('message','') if isinstance(error,dict) else ''
                    category,retry,fatal=classify_error(http,code,message)
                    result={'status':category,'error_type':type(exc).__name__,'http_status':http,
                            'error_code':code,'error_message':redact(message)[:800]}
                    if retry and counts[jid]>=protocol['max_attempts_per_job']:
                        result['status']='transport_exhausted';result['last_error_category']=category
                charge,known=charge_cost(job,result,protocol,reserve)
                with mutex:
                    reserved-=reserve;spent+=charge
                    append({'event':'end','utc':now(),'attempt_id':aid,**{k:job[k] for k in ('job_id','variant_id','sample_id','kind','replicate','provider','condition')},
                            'elapsed_seconds':round(time.monotonic()-start,3),'estimated_charge_cny':charge,
                            'usage_available':known,**result})
                    if result['status'] in TERMINAL:done.add(jid)
                    if retry:
                        consecutive[provider]+=1
                        if consecutive[provider]>=protocol['max_consecutive_transport_errors']:fatal=True
                    else:consecutive[provider]=0
                    if fatal:halt[provider]=result['status']
                    if spent+reserved>=protocol['estimated_cost_stop_cny']:halt['estimated_budget']='reached'
                    progress=status()
                    if len(done)%20==0 or result['status']!='response':print(json.dumps(progress),flush=True)
                if not retry or fatal:return
                time.sleep(3)
        with ThreadPoolExecutor(max_workers=protocol['max_workers']) as executor:
            in_flight={}
            while pending_jobs or in_flight:
                while pending_jobs and len(in_flight)<protocol['max_workers'] and not halt and new_attempts<args.max_new_attempts:
                    index=next((i for i,j in enumerate(pending_jobs) if active[j['provider']]<protocol['max_in_flight_per_provider']),None)
                    if index is None:break
                    job=pending_jobs.pop(index)
                    if counts[job['job_id']]>=protocol['max_attempts_per_job']:
                        halt['exhausted_nonterminal']=job['job_id'];break
                    active[job['provider']]+=1;in_flight[executor.submit(run,job)]=job['provider']
                if not in_flight:break
                finished,_=wait(in_flight,return_when=FIRST_COMPLETED)
                for future in finished:active[in_flight.pop(future)]-=1;future.result()
        with mutex:
            final=status();atomic_json(output/'last_execution.json',final);print(json.dumps(final),flush=True)
    finally:
        for client in clients.values():client.close()
        lock_path.unlink(missing_ok=True)

if __name__=='__main__':main()
