#!/usr/bin/env python3
"""Single-GPU course evaluation: fixed prompt/scoring/greedy decoding, resumable outputs."""
import argparse, hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def model_identity(path):
 """Content identity: changed weights at the same path must never reuse old results."""
 path=path.resolve()
 if not path.is_dir():raise ValueError(f'Model directory missing: {path}')
 files=[f for f in sorted(path.iterdir()) if f.is_file() and f.suffix in ('.json','.safetensors','.bin','.pt','.model','.txt','.jinja','.tiktoken') and not f.name.startswith(('optimizer','scheduler','rng_state','training_args'))]
 if not files:raise ValueError(f'No model or adapter files found: {path}')
 return {f.name:digest(f) for f in files}

def evaluation_hashes(run):
 return {str(f.relative_to(run)):digest(f) for f in sorted((run/'evaluation').rglob('*')) if f.is_file()}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',required=True,type=Path,help='Qwen2.5-VL-3B-Instruct, or a full/merged model derived from it')
 p.add_argument('--adapter',type=Path,help='Optional PEFT LoRA checkpoint; --model must be its base')
 p.add_argument('--checkpoint-model',action='store_true',help='Optional extra check for full training checkpoints with trainer_state.json; not needed for merged submissions')
 p.add_argument('--games',nargs='+',choices=['sokoban','maze','snake'],default=['sokoban','maze','snake'])
 p.add_argument('--gpu',help='One GPU index or UUID; default: first CUDA_VISIBLE_DEVICES entry, otherwise 0')
 p.add_argument('--batch-size',type=int,default=64)
 p.add_argument('--max-num-seqs',type=int,default=64)
 p.add_argument('--gpu-memory-utilization',type=float,default=.9)
 p.add_argument('--max-model-len',type=int,default=8192)
 p.add_argument('--data-dir',type=Path,default=ROOT/'test_data')
 p.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
 p.add_argument('--run-name',help='Reuse name and identical parameters to resume; otherwise automatic timestamp')
 p.add_argument('--limit',type=int,help='Per-game smoke limit; clearly recorded, not a full benchmark score')
 p.add_argument('--dry-run',action='store_true')
 a=p.parse_args()
 if a.gpu is None:
  visible=os.environ.get('CUDA_VISIBLE_DEVICES')
  a.gpu='0' if visible is None else visible.split(',')[0].strip()
 if ',' in a.gpu or not a.gpu.strip():p.error('--gpu must select exactly one GPU')
 if a.batch_size<1 or a.max_num_seqs<1 or a.max_model_len<=4096 or not 0<a.gpu_memory_utilization<1:p.error('Invalid resource parameters')
 if a.limit is not None and a.limit<1:p.error('--limit must be positive')
 a.games=list(dict.fromkeys(a.games)); a.data_dir=a.data_dir.resolve()
 data=json.loads((a.data_dir/'games_data_test.json').read_text())
 manifest=json.loads((a.data_dir/'manifest.json').read_text())
 # Verify all released bytes (including images/state exclusion list), not just annotation paths.
 for rel,sha in manifest['files'].items():
  f=(a.data_dir/rel).resolve()
  if not f.is_relative_to(a.data_dir) or digest(f)!=sha:raise ValueError(f'Test artifact changed: {rel}')
 counts={g:sum(r['game_name']==g for r in data) for g in a.games}
 if any(n==0 for n in counts.values()):raise ValueError('Selected game missing')
 expected=sum(min(n,a.limit) if a.limit else n for n in counts.values())
 name=a.run_name or ('qwen25vl-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
 if name in ('.','..') or Path(name).name!=name:p.error('run-name must be one directory name')
 run=a.output_dir.resolve()/name
 cmd=[sys.executable,str(ROOT/'evaluation/infer.py'),'infer','--model',str(a.model.resolve()),'--data-dir',str(a.data_dir),'--games',*a.games,'--output-dir',str(a.output_dir.resolve()),'--run-name',name,'--temperature','0','--top-p','1','--top-k','-1','--presence-penalty','0','--repetition-penalty','1','--max-tokens','4096','--seed','0','--no-enable-thinking','--tensor-parallel-size','1','--game-workers','1','--batch-size',str(a.batch_size),'--max-num-seqs',str(a.max_num_seqs),'--max-model-len',str(a.max_model_len),'--gpu-memory-utilization',str(a.gpu_memory_utilization),'--no-analyze-after-game','--no-auto-extract']
 if a.adapter:cmd+=['--adapter',str(a.adapter.resolve())]
 if a.checkpoint_model:cmd+=['--checkpoint-model']
 if a.limit:cmd+=['--limit',str(a.limit)]
 score=[sys.executable,str(ROOT/'evaluation/scripts/evaluate_run.py'),str(run),'--expect-games',str(len(a.games)),'--expect-samples',str(expected)]
 if a.dry_run:
  print(json.dumps({'infer':cmd,'score':score,'CUDA_VISIBLE_DEVICES':a.gpu},indent=2));return
 run.mkdir(parents=True,exist_ok=True)
 import fcntl
 with (run/'.course.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  protocol={'version':'prml-topic2-v1','test_manifest_sha256':digest(a.data_dir/'manifest.json'),'games':a.games,'expected_samples':expected,'full_test':a.limit is None,'max_tokens':4096,'temperature':0,'model':str(a.model.resolve()),'model_files':model_identity(a.model),'adapter':str(a.adapter.resolve()) if a.adapter else None,'adapter_files':model_identity(a.adapter) if a.adapter else None,'checkpoint_model':a.checkpoint_model,'limit':a.limit,'resources':{'batch_size':a.batch_size,'max_num_seqs':a.max_num_seqs,'max_model_len':a.max_model_len,'gpu_memory_utilization':a.gpu_memory_utilization},'evaluator_files':{str(f.relative_to(ROOT)):digest(f) for f in sorted((ROOT/'evaluation').rglob('*.py'))}}
  pp=run/'protocol.json'
  if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Run protocol changed; use a new run name')
  pp.write_text(json.dumps(protocol,indent=2)+'\n')
  marker=run/'completion.json'
  if marker.exists():
   previous=json.loads(marker.read_text())
   paths={g:run/'games'/f'{g}.jsonl' for g in a.games}
   if all(f.is_file() for f in paths.values()) and previous.get('input_hashes')=={g:digest(f) for g,f in paths.items()} and (run/'evaluation/summary.json').is_file() and previous.get('evaluation_hashes')==evaluation_hashes(run):
    print('Already complete:',run);return
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg')
  started=time.time()
  with (run/'launcher.log').open('a') as log:
   log.write('COMMAND '+json.dumps(cmd)+'\n');log.flush()
   subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  # A finished evaluation is reusable only if its exact input files remain unchanged.
  input_hashes={g:digest(run/'games'/f'{g}.jsonl') for g in a.games}
  marker=run/'completion.json'
  if (run/'evaluation').exists():
   (run/'evaluation').rename(run/('evaluation-previous-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')))
  subprocess.run(score,env=env,check=True)
  timings=[json.loads(s) for s in (run/'timing.jsonl').read_text().splitlines()] if (run/'timing.jsonl').exists() else []
  summary={'status':'completed','input_hashes':input_hashes,'evaluation_hashes':evaluation_hashes(run),'full_test':a.limit is None,'samples':expected,'launcher_seconds_this_attempt':time.time()-started,'engine_load_seconds':sum(t['seconds'] for t in timings if t['kind']=='engine_load'),'generation_seconds':sum(t['seconds'] for t in timings if t['kind']=='generation'),'completion_tokens':sum(t.get('completion_tokens',0) for t in timings if t['kind']=='generation')}
  marker.write_text(json.dumps(summary,indent=2)+'\n'); print('Completed:',run)
if __name__=='__main__':main()
