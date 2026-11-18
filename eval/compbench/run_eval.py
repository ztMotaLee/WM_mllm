"""Run a user-supplied, SHA-256 pinned upstream CompBench evaluator."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

# eval_all.py at github.com/comp-bench/CompBench commit cb9165b01d14a321db14611afd9922c43a04665d
EVALUATOR_SHA256 = '04f622a30231c8165188dafb33050233ceb7756c69fb0454f11842373dea0e5e'


def verify_evaluator(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EVALUATOR_SHA256:
        raise ValueError('upstream eval_all.py does not match the pinned SHA-256')
    return digest


def run(args):
    evaluator_path = args.evaluator.resolve(strict=True)
    verify_evaluator(evaluator_path)
    if getattr(args, 'check_inputs', False):
        from .export import check_inputs
        if args.index is None:
            raise ValueError('--check-inputs requires the prepared --index')
        print(json.dumps(check_inputs(args.index, args.data_root, args.results_root, args.model, args.tasks), indent=2))
        return
    model_root = args.results_root / args.model
    if not model_root.is_dir():
        raise ValueError('model result tree is missing')
    if args.model in ('.', '..') or '/' in args.model or '\\' in args.model:
        raise ValueError('model must be a simple result-directory name')
    if args.clip_path is None:
        raise ValueError('metric inference requires --clip-path')
    clip_path = args.clip_path.resolve(strict=True)
    if not (clip_path / 'config.json').is_file():
        raise ValueError('supply the downloaded CLIP ViT-L/14 directory')
    original_path = list(sys.path)
    original_argv = list(sys.argv)
    module_name = '_wmmllm_external_compbench_evaluator'
    prior_module = sys.modules.get(module_name)
    original_clip_loader = None
    tm_clip = None
    original_init = None
    original_defaults = None
    try:
        sys.path.insert(0, str(evaluator_path.parent))
        spec = importlib.util.spec_from_file_location(module_name, evaluator_path)
        if spec is None or spec.loader is None:
            raise ValueError('cannot load external evaluator')
        evaluator = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = evaluator
        spec.loader.exec_module(evaluator)
        import torchmetrics.multimodal.clip_score as tm_clip
        from transformers import CLIPModel, CLIPProcessor
        original_clip_loader = getattr(tm_clip, '_get_clip_model_and_processor', None)

        def local_clip_loader(model_name_or_path):
            if isinstance(model_name_or_path, str) and Path(model_name_or_path).resolve() == clip_path:
                return (CLIPModel.from_pretrained(str(clip_path), local_files_only=True),
                        CLIPProcessor.from_pretrained(str(clip_path), local_files_only=True))
            if original_clip_loader is None:
                raise ValueError('torchmetrics CLIP loading interface is incompatible')
            return original_clip_loader(model_name_or_path)

        if original_clip_loader is not None:
            tm_clip._get_clip_model_and_processor = local_clip_loader
        original_init = evaluator.CompBenchEvaluator.__init__
        original_defaults = original_init.__defaults__
        defaults = list(original_defaults or ())
        if len(defaults) < 2 or Path(str(defaults[1])).name != 'clip-vit-large-patch14':
            raise ValueError('upstream CLIP initializer contract changed')
        defaults[1] = str(clip_path)
        original_init.__defaults__ = tuple(defaults)
        sys.argv = ['eval_all.py', '--model_names', args.model,
                    '--tasks', *args.tasks, '--metric', args.metric,
                    '--data_root', str(args.data_root), '--results_root', str(args.results_root),
                    '--output_dir', str(args.output_dir), '--device', args.device, '--resume']
        evaluator.main()
    finally:
        if original_clip_loader is not None and tm_clip is not None:
            tm_clip._get_clip_model_and_processor = original_clip_loader
        if original_init is not None:
            original_init.__defaults__ = original_defaults
        sys.argv = original_argv
        sys.path[:] = original_path
        if prior_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = prior_module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('--evaluator', type=Path, required=True,
                        help='eval_all.py from a CompBench checkout at the pinned commit')
    parser.add_argument('--clip-path', type=Path)
    parser.add_argument('--check-inputs', action='store_true',
                        help='verify indexed dataset/result paths and exit without metric inference')
    parser.add_argument('--index', type=Path, help='prepared CompBench JSONL index for --check-inputs')
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--metric', default='all', choices=['all', 'LC-T', 'LC-I', 'PSNR', 'SSIM', 'LPIPS'])
    parser.add_argument('--tasks', nargs='+', default=['all'])
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
