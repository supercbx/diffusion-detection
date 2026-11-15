from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from dataset import (Corruption, TriStreamDataset, assert_disjoint, load_manifest,
                     seed_worker, write_manifest)
from gpu_processor_tristream import GPUProcessor_TriStream
from model import FPRAClassifier, TriStreamClassifier, smoke_model

DEFAULT_CONFIG = {
    'model': {'fusion_dim': 768, 'branches': 'both', 'pretrained': True, 'dropout': 0.0,
              'clip_arch': 'vit_base_patch16_clip_224.openai',
              'swin_arch': 'swin_small_patch4_window7_224'},
    'processor': {'image_size': [224, 224], 'amp_strength': 0.02,
                  'phase_strength': 0.05, 'phase_window': 3, 'gaussian_sigma': 1.0},
    'data': {'img_size': 224, 'jpeg_p': 0.4, 'residual_scale': [1.0, 1.0]},
    'training': {'epochs': 50, 'batch_size': 32, 'lr': 1e-5, 'weight_decay': 0.05,
                 'num_workers': 0, 'seed': 42, 'amp': True},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    if device.type not in {'cuda', 'cpu'}:
        raise ValueError('Only CPU and CUDA devices are supported')
    return device


def load_config(args):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if getattr(args, 'config', None):
        with Path(args.config).open(encoding='utf-8') as handle:
            supplied = json.load(handle)
        for section, values in supplied.items():
            if section not in config or not isinstance(values, dict):
                raise ValueError(f'Unknown configuration section: {section}')
            unknown = set(values) - set(config[section]) - {'clip_checkpoint', 'swin_checkpoint'}
            if unknown:
                raise ValueError(f'Unknown configuration fields: {unknown}')
            config[section].update(values)
    for argument, section, key in [('epochs', 'training', 'epochs'), ('batch_size', 'training', 'batch_size'),
                                    ('workers', 'training', 'num_workers'), ('seed', 'training', 'seed'),
                                    ('alpha', 'processor', 'amp_strength'), ('beta', 'processor', 'phase_strength'),
                                    ('branches', 'model', 'branches')]:
        if getattr(args, argument, None) is not None:
            config[section][key] = getattr(args, argument)
    if getattr(args, 'no_pretrained', False):
        config['model']['pretrained'] = False
    training, data, processor = config['training'], config['data'], config['processor']
    if any(training[key] <= 0 for key in ['epochs', 'batch_size', 'lr']):
        raise ValueError('epochs, batch_size and lr must be positive')
    if training['num_workers'] < 0 or training['weight_decay'] < 0:
        raise ValueError('Invalid workers or weight decay')
    if data['img_size'] != 224 or list(processor['image_size']) != [224, 224]:
        raise ValueError('The configured CLIP/Swin backbones require 224x224 inputs')
    return config


def make_loader(records, config, mode, device, use_gpu_processor=False,
                corruption=None, generator=None, workers=None):
    workers = config['training']['num_workers'] if workers is None else workers
    if use_gpu_processor and workers:
        raise ValueError('CuPy processing requires --workers 0; do not initialize CUDA in worker processes')
    processor = GPUProcessor_TriStream(config['processor'], use_gpu=use_gpu_processor)
    data = config['data']
    dataset = TriStreamDataset(records, processor, mode=mode, img_size=data['img_size'],
                               seed=config['training']['seed'], jpeg_p=data['jpeg_p'],
                               residual_scale=data['residual_scale'], corruption=corruption)
    return DataLoader(dataset, batch_size=config['training']['batch_size'],
                      shuffle=mode == 'train', num_workers=workers,
                      pin_memory=device.type == 'cuda', worker_init_fn=seed_worker,
                      generator=generator)


def binary_metrics(labels, probabilities, threshold=0.5):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if not len(labels) or labels.shape != probabilities.shape:
        raise ValueError('Expected nonempty aligned labels and probabilities')
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all():
        raise ValueError('Invalid labels or probabilities')
    if not 0 <= threshold <= 1 or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Probability and threshold values must lie in [0,1]')
    prediction = probabilities >= threshold
    return {'n': len(labels), 'accuracy': float(accuracy_score(labels, prediction)),
            'auc': float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
            'confusion_matrix': confusion_matrix(labels, prediction, labels=[0, 1]).tolist(),
            'threshold': threshold}


@torch.inference_mode()
def evaluate(model, loader, device, threshold=0.5, export_features=False):
    model.eval()
    labels, probabilities, features = [], [], []
    for rgb, amp, phase, target in loader:
        inputs = [x.to(device, non_blocking=True) for x in (rgb, amp, phase)]
        if export_features:
            fused = model.forward_features(*inputs)
            logits = model.classifier(fused).squeeze(-1)
            features.append(fused.cpu().numpy())
        else:
            logits = model(*inputs)
        labels.extend(target.tolist())
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    return binary_metrics(labels, probabilities, threshold), probabilities, (np.concatenate(features) if features else None)


def save_json(data, path):
    with Path(path).open('w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)


def reserve_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def manifest_fingerprint(records):
    import hashlib
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode('utf-8')).hexdigest()


def rng_state(generator):
    numpy = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]],
            'torch': torch.get_rng_state(), 'loader': generator.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    if torch.cuda.is_available() and state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume requires the original number of CUDA devices')
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(checkpoint, path):
    temp = Path(str(path) + '.tmp')
    torch.save(checkpoint, temp)
    temp.replace(path)


def train(args):
    config = load_config(args)
    device = choose_device(args.device)
    cfg = config['training']
    set_seed(cfg['seed'])
    records = load_manifest(args.train_manifest, 'train')
    validation = load_manifest(args.val_manifest, 'val')
    assert_disjoint(records, validation)
    for name, rows in [('train', records), ('val', validation)]:
        if {r['label'] for r in rows} != {0, 1}:
            raise ValueError(f'{name} must contain both real and fake images')
    fingerprints = [manifest_fingerprint(records), manifest_fingerprint(validation)]
    previous = None
    if args.resume:
        previous = torch.load(args.resume, map_location='cpu', weights_only=True)
        if previous['config'] != config or previous['manifest_fingerprints'] != fingerprints:
            raise ValueError('Resume configuration and train/validation manifests must match the checkpoint')
    out = reserve_output(args.output_dir)
    save_json(config, out / 'config.json')
    write_manifest(records, out / 'train_manifest.csv')
    write_manifest(validation, out / 'val_manifest.csv')
    generator = torch.Generator().manual_seed(cfg['seed'])
    training_loader = make_loader(records, config, 'train', device, args.gpu_processor, generator=generator)
    val_loader = make_loader(validation, config, 'val', device, args.gpu_processor)
    model_options = dict(config['model'])
    if previous:
        model_options['pretrained'] = False
        model_options.pop('clip_checkpoint', None)
        model_options.pop('swin_checkpoint', None)
    model = FPRAClassifier(**model_options).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    amp_enabled = bool(cfg['amp'] and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    criterion = nn.BCEWithLogitsLoss()
    start, best = 1, -1.0
    if previous:
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler'])
        scaler.load_state_dict(previous['scaler'])
        if previous.get('selection_metric') != 'accuracy':
            raise ValueError('Resume requires an accuracy-selected checkpoint; old checkpoints remain usable for evaluation')
        start, best = previous['epoch'] + 1, previous['best_accuracy']
        restore_rng(previous['rng'], generator)

        prior_best = Path(args.resume).with_name('best.pt')
        if prior_best.is_file():
            import shutil
            shutil.copy2(prior_best, out / 'best.pt')
    if start > cfg['epochs']:
        raise ValueError('Checkpoint has already completed all configured epochs')
    import platform
    import importlib.metadata
    versions = {}
    for package in ['torch', 'torchvision', 'timm', 'numpy', 'scipy', 'Pillow', 'scikit-learn']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    save_json(dict(python=platform.python_version(), packages=versions, device=str(device),
                   device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                   precision='mixed-float16' if amp_enabled else 'float32',
                   gpu_processor_active=training_loader.dataset.processor.use_gpu,
                   manifest_fingerprints=fingerprints), out / 'environment.json')
    if not config['model']['pretrained']:
        print('WARNING: random backbone initialization is a debugging setting, not the paper protocol.', flush=True)
    print(f'Training on {device}: {len(records)} train / {len(validation)} val images', flush=True)
    with (out / 'history.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'train_loss', 'val_accuracy', 'val_auc', 'lr'])
        writer.writeheader()
        for epoch in range(start, cfg['epochs'] + 1):
            model.train()
            total, count = 0.0, 0
            lr = optimizer.param_groups[0]['lr']
            for rgb, amp, phase, target in training_loader:
                inputs = [x.to(device, non_blocking=True) for x in (rgb, amp, phase)]
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    loss = criterion(model(*inputs), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss; inspect input residual scales and gradients')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += loss.item() * target.numel()
                count += target.numel()
            metrics, _, _ = evaluate(model, val_loader, device)
            scheduler.step()
            improved = metrics['accuracy'] > best
            best = max(best, metrics['accuracy'])
            row = dict(epoch=epoch, train_loss=total / count, val_accuracy=metrics['accuracy'],
                       val_auc=metrics['auc'], lr=lr)
            writer.writerow(row)
            handle.flush()
            checkpoint = dict(format_version=1, config=config, model=model.state_dict(),
                              optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                              scaler=scaler.state_dict(), epoch=epoch, best_accuracy=best,
                              selection_metric='accuracy',
                              manifest_fingerprints=fingerprints,
                              source_paths=[r['path'].casefold() for r in records + validation],
                              rng=rng_state(generator), validation=metrics)
            atomic_save(checkpoint, out / 'last.pt')
            if improved:
                atomic_save(checkpoint, out / 'best.pt')
            print(json.dumps(row), flush=True)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1:
        raise ValueError('Not a version-1 FPRA checkpoint; legacy architecture weights are not compatible')
    config = checkpoint['config']
    options = dict(config['model'], pretrained=False)
    options.pop('clip_checkpoint', None)
    options.pop('swin_checkpoint', None)
    model = FPRAClassifier(**options).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.source_paths = set(checkpoint.get('source_paths', []))
    model.eval()
    return model, config


def evaluation_report(model, records, config, device, output, args, corruption=None):
    loader = make_loader(records, config, 'test', device, args.gpu_processor,
                         corruption=corruption, workers=args.workers)
    overall, probabilities, features = evaluate(model, loader, device, args.threshold,
                                                getattr(args, 'export_features', False))
    subsets = {}
    for subset in sorted({r['subset'] for r in records}):
        indices = [i for i, r in enumerate(records) if r['subset'] == subset]
        subsets[subset] = binary_metrics([records[i]['label'] for i in indices],
                                         [probabilities[i] for i in indices], args.threshold)
    aucs = [m['auc'] for m in subsets.values()]
    report = {'overall': overall, 'subsets': subsets,
              'macro_accuracy': float(np.mean([m['accuracy'] for m in subsets.values()])),
              'macro_auc': float(np.mean(aucs)) if all(a is not None for a in aucs) else None}
    save_json(report, output / 'metrics.json')
    with (output / 'predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'label', 'subset', 'split', 'probability_fake', 'prediction'])
        writer.writeheader()
        for row, probability in zip(records, probabilities):
            writer.writerow(dict(row, probability_fake=probability, prediction=int(probability >= args.threshold)))
    if features is not None:
        np.savez_compressed(output / 'features.npz', features=features,
                            labels=np.array([r['label'] for r in records]),
                            subsets=np.array([r['subset'] for r in records]),
                            paths=np.array([r['path'] for r in records]))
    return report


def test(args):
    device = choose_device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)
    config['training']['batch_size'] = args.batch_size
    set_seed(config['training']['seed'])
    records = load_manifest(args.manifest, getattr(args, 'split', 'test'))
    overlap = model.source_paths & {r['path'].casefold() for r in records}
    if overlap and getattr(args, 'split', 'test') == 'test':
        raise ValueError(f'Test image overlaps checkpoint training/validation data: {next(iter(overlap))}')
    output = reserve_output(args.output_dir)
    save_json(config, output / 'config.json')
    report = evaluation_report(model, records, config, device, output, args)
    report['evaluation_split'] = getattr(args, 'split', 'test')
    report['training_or_selection_overlap'] = len(overlap)
    save_json(report, output / 'metrics.json')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def robustness(args):
    device = choose_device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)
    config['training']['batch_size'] = args.batch_size
    records = load_manifest(args.manifest, getattr(args, 'split', 'test'))
    overlap = model.source_paths & {r['path'].casefold() for r in records}
    if overlap and getattr(args, 'split', 'test') == 'test':
        raise ValueError(f'Test image overlaps checkpoint training/validation data: {next(iter(overlap))}')
    output = reserve_output(args.output_dir)
    results = {}
    grid = {'clean': [0], 'jpeg': [100, 90, 80, 70, 60, 50],
            'crop': [1, .9, .8, .7, .6, .5], 'blur': [0, .5, 1, 1.5, 2, 3],
            'noise': [0, 5, 10, 15, 20, 25]}
    for kind, values in grid.items():
        for value in values:
            key = f'{kind}_{value}'
            destination = reserve_output(output / key)
            results[key] = evaluation_report(model, records, config, device, destination, args,
                                              Corruption(kind, value, config['training']['seed']))
            print(key, results[key]['overall'], flush=True)
    save_json({'protocol': 'corrupt RGB before resize and probing', 'config': config,
               'evaluation_split': getattr(args, 'split', 'test'),
               'training_or_selection_overlap': len(overlap),
               'results': results}, output / 'robustness.json')


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark(args):
    device = choose_device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)
    processor = GPUProcessor_TriStream(config['processor'], use_gpu=args.gpu_processor)
    with Image.open(args.image) as source:
        image = source.convert('RGB').copy()
    record = dict(path=str(Path(args.image).resolve()), label=0, subset='benchmark', split='test')
    dataset = TriStreamDataset([record], processor, mode='test', **config['data'])
    rgb, amp, phase, _ = dataset[0]
    fixed = [x.unsqueeze(0).to(device) for x in (rgb, amp, phase)]
    def forward():
        model(*fixed)
    def end_to_end():

        rgb, amp, phase = processor.process_single_image(np.asarray(image), is_train_mode=False, seed=42)
        from dataset import rgb_tensor
        inputs = [rgb_tensor(rgb), torch.from_numpy(amp).unsqueeze(0) / config['data']['residual_scale'][0],
                  torch.from_numpy(phase).unsqueeze(0) / config['data']['residual_scale'][1]]
        model(*[x.unsqueeze(0).to(device) for x in inputs])
    def measure(function):
        for _ in range(args.warmup):
            function()
        synchronize(device)
        timings = []
        for _ in range(args.iterations):
            synchronize(device)
            start = time.perf_counter()
            function()
            synchronize(device)
            timings.append((time.perf_counter() - start) * 1000)
        return dict(mean_ms=float(np.mean(timings)), median_ms=float(np.median(timings)),
                    p95_ms=float(np.percentile(timings, 95)), fps=1000 / float(np.mean(timings)))
    result = {'device': str(device), 'batch_size': 1, 'precision': 'float32',
              'parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'forward_only': measure(forward), 'decoded_image_to_prediction': measure(end_to_end),
              'includes_file_io': False, 'gpu_processor_requested': args.gpu_processor,
              'torch_version': str(torch.__version__), 'gflops': None}
    print(json.dumps(result, indent=2))
    with Path(args.output).open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)


def smoke(args):
    set_seed(42)
    device = choose_device(args.device)
    with tempfile.TemporaryDirectory(prefix='fpra_smoke_') as folder:
        root = Path(folder)
        records = []
        rng = np.random.default_rng(42)
        for index in range(4):
            path = root / f'{index}.png'
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
            records.append(dict(path=str(path), label=index % 2, subset='synthetic', split='train'))
        config = copy.deepcopy(DEFAULT_CONFIG)
        config['training']['batch_size'] = 2
        config['processor']['image_size'] = [32, 32]
        config['data']['img_size'] = 32
        loader = make_loader(records, config, 'train', device)
        model = smoke_model().to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        for rgb, amp, phase, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(*[x.to(device) for x in (rgb, amp, phase)])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        val = make_loader(records, config, 'test', device)
        metrics, _, _ = evaluate(model, val, device)
        torch.save({'model': model.state_dict()}, root / 'smoke.pt')
        restored = smoke_model().to(device)
        restored.load_state_dict(torch.load(root / 'smoke.pt', map_location='cpu', weights_only=True)['model'])
        restored_metrics, _, _ = evaluate(restored, val, device)
        assert metrics == restored_metrics
        assert all(p.grad is None for p in model.encoder_rgb.parameters())
        print(json.dumps({'status': 'passed', 'device': str(device), 'last_loss': loss.item(),
                          'checkpoint_roundtrip': True,
                          'note': 'Synthetic pipeline check with tiny test encoders, NOT a paper result.'}, indent=2))


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='command', required=True)
    train_parser = commands.add_parser('train', help='Train using explicit train/validation manifests')
    train_parser.add_argument('--train-manifest', required=True)
    train_parser.add_argument('--val-manifest', required=True)
    train_parser.add_argument('--output-dir', required=True)
    train_parser.add_argument('--config')
    train_parser.add_argument('--resume')
    train_parser.add_argument('--no-pretrained', action='store_true')
    for flag in ['epochs', 'batch-size', 'workers', 'seed']:
        train_parser.add_argument('--' + flag, type=int)
    train_parser.add_argument('--alpha', type=float)
    train_parser.add_argument('--beta', type=float)
    train_parser.add_argument('--branches', choices=['rgb', 'amp', 'phase', 'both'])
    train_parser.set_defaults(function=train)
    for name, function in [('evaluate', test), ('robustness', robustness)]:
        command = commands.add_parser(name)
        command.add_argument('--checkpoint', required=True)
        command.add_argument('--manifest', required=True)
        command.add_argument('--output-dir', required=True)
        command.add_argument('--batch-size', type=int, default=32)
        command.add_argument('--workers', type=int, default=0)
        command.add_argument('--threshold', type=float, default=0.5)
        command.add_argument('--split', choices=['train', 'val', 'test'], default='test')
        if name == 'evaluate':
            command.add_argument('--export-features', action='store_true')
        command.set_defaults(function=function)
    command = commands.add_parser('benchmark', help='Batch-1 timings; no fabricated FLOPs')
    command.add_argument('--checkpoint', required=True)
    command.add_argument('--image', required=True)
    command.add_argument('--output', required=True)
    command.add_argument('--warmup', type=int, default=10)
    command.add_argument('--iterations', type=int, default=100)
    command.set_defaults(function=benchmark)
    command = commands.add_parser('smoke-test', help='Offline tiny-encoder pipeline check')
    command.set_defaults(function=smoke)
    for command in commands.choices.values():
        command.add_argument('--device', default='auto')
        command.add_argument('--gpu-processor', action='store_true')
    return root


def main():
    args = parser().parse_args()
    if getattr(args, 'iterations', 1) < 1 or getattr(args, 'warmup', 0) < 0:
        raise ValueError('Invalid benchmark iteration count')
    args.function(args)


if __name__ == '__main__':
    main()
