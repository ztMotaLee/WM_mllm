"""Run init + rotation + drawn local edit + revisit on CPU. Not a model."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from wmmllm.config import AppConfig,ChatConfig,ImageConfig,VideoConfig
from wmmllm.budget import BudgetController
from wmmllm.providers.factory import Providers
from wmmllm.providers.video import NoneVideoBackend
from wmmllm.agent.executor import EngineDeps
from wmmllm.session.runner import SessionRunner
from wmmllm.session.io import write_once
from generate_seed import generate
from offline_providers import OfflineProceduralChat,OfflineDrawnImageEdit

def run(directory):
    started=time.monotonic();root=Path(directory).resolve();root.mkdir(parents=True,exist_ok=True)
    print('OFFLINE PROCEDURAL FIXTURE: deterministic drawn providers, not a model.',flush=True)
    seed=generate(root/'procedural_seed.png')
    config=AppConfig(ChatConfig('offline-demo','offline-not-a-model'),ImageConfig('offline-demo','offline-not-a-model'),video=VideoConfig(num_frames=3,fps=3.))
    providers=Providers(OfflineProceduralChat(),OfflineDrawnImageEdit(),NoneVideoBackend())
    budget=BudgetController(config.budget,checkpoint=lambda value: __import__('wmmllm.session.io',fromlist=['write_atomic']).write_atomic(root/'budget/checkpoint.json',value,run_root=root))
    runner=SessionRunner(root,deps=EngineDeps(config,providers,budget,root,None))
    turns=['Look right by 45 degrees.','Paint a yellow square on the middle tile.','Look back at the initial direction.']
    runner.initialize(seed,captions=turns,title='OFFLINE PROCEDURAL FIXTURE - not a model',panorama_width=1024)
    for caption in turns:
        turn=runner.step(caption);print(json.dumps({'turn':turn['index'],'status':'committed' if turn['status']=='accepted' else turn['status'],'video_status':turn['presentation']['status']}),flush=True)
        if turn['status']!='accepted':raise RuntimeError('Offline fixture turn was held')
    elapsed=time.monotonic()-started
    write_once(root/'OFFLINE_FIXTURE.json',{'label':'OFFLINE PROCEDURAL FIXTURE - not a model','source':'generate_seed.py','turns':3,'elapsed_seconds':elapsed,'network_requests':0,'model_calls':0},run_root=root)
    print(json.dumps({'offline_fixture_complete':True,'elapsed_seconds':round(elapsed,3),'model_calls':0}),flush=True)
    providers.close();return runner

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);run(parser.parse_args().run)
