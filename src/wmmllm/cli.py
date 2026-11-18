"""Local engine CLI; CPU inspection requires no provider credentials."""
from __future__ import annotations
import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from . import __version__
from .config import load_config, ConfigError
from .logging_utils import get_logger, safe_error
from .session.io import read_json, write_atomic

class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, "wmmllm: unsupported arguments. API keys belong in WMM_* environment variables or an explicitly selected env file.\n")

def _resolve_config(args, root):
    saved_path=root/'runtime_config.json'
    saved=read_json(saved_path,run_root=root) if saved_path.exists() and not args.config else None
    return load_config(args.config,env_file=args.env_file,overrides=saved)

def _engine(root, config):
    from .budget import BudgetController
    from .providers.factory import create_providers
    from .agent.executor import EngineDeps
    if config.video.size != (1344,768) or config.image.view_output_size != (1344,768):
        raise ConfigError('Core sessions require a 1344x768 viewport and image output; no request was sent.')
    checkpoint=root/'budget'/'checkpoint.json'
    def persist(value):write_atomic(checkpoint,value,run_root=root)
    if checkpoint.exists():
        budget=BudgetController.from_snapshot(config.budget,read_json(checkpoint,run_root=root),checkpoint=persist)
    else:
        if (root/'session.json').exists():raise ValueError('Existing session has no budget checkpoint; no request was sent.')
        budget=BudgetController(config.budget,checkpoint=persist)
        persist(budget.snapshot())
    logger=get_logger()
    return EngineDeps(config,create_providers(config,budget=budget,logger=logger),budget,root,logger)

def main(argv: Sequence[str] | None=None) -> int:
    parser=_Parser(prog='wmmllm',description='Persistent panorama world model')
    parser.add_argument('--version',action='version',version=__version__)
    commands=parser.add_subparsers(dest='command',required=True,parser_class=_Parser)
    init=commands.add_parser('init');init.add_argument('--image',type=Path,required=True);init.add_argument('--run',type=Path,required=True)
    init.add_argument('--title',default='Local panorama session');init.add_argument('--seed',type=int,default=0)
    resume=commands.add_parser('resume');resume.add_argument('--run',type=Path,required=True)
    step=commands.add_parser('step');step.add_argument('--run',type=Path,required=True);step.add_argument('instruction')
    session=commands.add_parser('session');session.add_argument('--run',type=Path,required=True);session.add_argument('--script',type=Path,required=True)
    view=commands.add_parser('view');view.add_argument('--run',type=Path,required=True);view.add_argument('--yaw',type=float,required=True);view.add_argument('--pitch',type=float,required=True);view.add_argument('--out',type=Path,required=True)
    check=commands.add_parser('check-config')
    for sub in (init,resume,step,session,check):sub.add_argument('--config',type=Path);sub.add_argument('--env-file',type=Path)
    args=parser.parse_args(argv);deps=None;runtime_guard=None
    try:
        if args.command=='check-config':
            config=load_config(args.config,env_file=args.env_file)
            print(json.dumps({'chat_key_present':bool(config.chat.api_key),'image_key_present':bool(config.image.api_key),'chat_model_present':bool(config.chat.model),'image_model_present':bool(config.image.model),'configuration_valid':True},sort_keys=True));return 0
        root=args.run.resolve()
        if args.command=='view':
            import math
            from PIL import Image
            from .session.io import validate_session
            from .state.panorama import load_panorama_state,render_view
            if not math.isfinite(args.yaw) or not math.isfinite(args.pitch) or not -80<=args.pitch<=80:raise ValueError('View pose must be finite with pitch between -80 and 80 degrees.')
            record=validate_session(read_json(root/'session.json',run_root=root),run_root=root)
            node=next(n for n in record['nodes'] if n['node_id']==record['current_node_id'])
            state=load_panorama_state(node['panorama_state'],run_root=root)
            pixels=render_view(state,yaw=args.yaw,pitch=args.pitch,hfov=node['pose']['hfov'],width=node['viewport_profile']['width'],height=node['viewport_profile']['height'])
            with args.out.open('xb') as stream:Image.fromarray(pixels).save(stream,format='PNG')
            print(json.dumps({'status':'rendered','provider_requests':0}));return 0
        config=_resolve_config(args,root)
        root.mkdir(parents=True,exist_ok=True)
        from .session.io import file_lock
        runtime_guard=file_lock(root/'runtime.lock',run_root=root)
        runtime_guard.__enter__()
        deps=_engine(root,config)
        from .session.runner import SessionRunner
        runner=SessionRunner(root,deps=deps)
        if args.command=='init':
            record=runner.initialize(args.image,captions=(),seed=args.seed,title=args.title)
            print(json.dumps(_session_summary(record)))
        elif args.command=='resume':print(json.dumps(_session_summary(runner.resume())))
        elif args.command=='step':print(json.dumps(_turn_summary(runner.step(args.instruction)),ensure_ascii=False))
        else:
            script=json.loads(args.script.read_text(encoding='utf-8'));turns=script.get('turns') if isinstance(script,dict) else script
            if not isinstance(turns,list) or not all(isinstance(t,str) or (isinstance(t,dict) and isinstance(t.get('instruction'),str)) for t in turns):raise ValueError('Session script must be an array of instructions or an object with turns.')
            record=runner.resume()
            if record['status'] != 'ready':
                print(json.dumps(_session_summary(record)))
                return 0
            for turn in turns:
                caption=turn if isinstance(turn,str) else turn['instruction']
                print(json.dumps(_turn_summary(runner.step(caption)),ensure_ascii=False),flush=True)
        return 0
    except ConfigError as error:print(str(error),file=sys.stderr);return 2
    except Exception as error:print(json.dumps({'status':'error','error':safe_error(error)}),file=sys.stderr);return 1
    finally:
        if deps is not None:deps.providers.close()
        if runtime_guard is not None:runtime_guard.__exit__(None,None,None)

def _session_summary(record):
    value={'status':record['status'],'current_node_id':record['current_node_id']}
    bootstrap=record.get('extensions',{}).get('bootstrap',{})
    if bootstrap.get('status') == 'BOOTSTRAP_HOLD':
        value.update(bootstrap_status='BOOTSTRAP_HOLD',world_initialized=False,action_completed=False)
    return value

def _turn_summary(turn):
    return {'index':turn['index'],'status':'committed' if turn['status']=='accepted' else turn['status'],
            'current_node_id':turn['to_node_id'],'semantic':turn['semantic']['status'],
            'presentation':turn['presentation']['status']}


if __name__=='__main__':raise SystemExit(main())
