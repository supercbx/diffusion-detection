import argparse
import csv
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from analysis_tools import _Moments, frequency_template, frequency_regularize
from dataset import load_manifest, rgb_tensor, stable_seed, write_manifest
from gpu_processor_tristream import GPUProcessor_TriStream
from main import binary_metrics, choose_device, load_checkpoint, reserve_output, save_json, synchronize


def prepare_inputs(image, processor, seed, config, device):
    rgb, amp, phase = processor.process_single_image(np.asarray(image), is_train_mode=False, seed=seed)
    tensors = [rgb_tensor(rgb),
               torch.from_numpy(amp).unsqueeze(0) / config['data']['residual_scale'][0],
               torch.from_numpy(phase).unsqueeze(0) / config['data']['residual_scale'][1]]
    return [value.unsqueeze(0).to(device) for value in tensors], amp, phase


def decoded_images(records, seed):
    images = []
    for row in records:
        with Image.open(row['path']) as source:
            image = ImageOps.exif_transpose(source).convert('RGB').copy()
        images.append((image, stable_seed(row['path'], seed)))
    return images


def synchronize_backends(device, processor):
    synchronize(device)
    if processor.use_gpu:
        processor.xp.cuda.runtime.deviceSynchronize()


@torch.inference_mode()
def benchmark(args):
    device = choose_device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)
    records = load_manifest(args.manifest)
    if len(records) != 50:
        raise ValueError('The paper benchmark requires a manifest of exactly 50 images')
    processor = GPUProcessor_TriStream(config['processor'], use_gpu=args.gpu_processor)
    images = decoded_images(records, config['training']['seed'])
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timings = []
    try:
        with torch.autocast(device_type=device.type, enabled=False):
            for pass_index in range(6):
                for image, seed in images:
                    synchronize_backends(device, processor)
                    start = time.perf_counter()
                    resized = image.resize((224, 224), Image.Resampling.BICUBIC)
                    inputs, _, _ = prepare_inputs(resized, processor, seed, config, device)
                    torch.sigmoid(model(*inputs))
                    synchronize_backends(device, processor)
                    if pass_index == 5:
                        timings.append((time.perf_counter() - start) * 1000)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn
    result = dict(protocol='paper_table_12', image_count=50, warmup_passes=5,
                  batch_size=1, precision='float32', tf32=False, includes_decode=False,
                  includes_resize=True, device=str(device), gpu_processor=processor.use_gpu,
                  latency_ms=float(np.mean(timings)), fps=1000 / float(np.mean(timings)),
                  per_image_ms=timings, parameters=sum(p.numel() for p in model.parameters()),
                  manifest=str(Path(args.manifest).resolve()), gflops=None)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


@torch.inference_mode()
def matched_preprocessing(args):
    device = choose_device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)
    records = load_manifest(args.manifest)
    if {row['label'] for row in records} != {0, 1}:
        raise ValueError('Matched preprocessing requires both classes')
    processor = GPUProcessor_TriStream(config['processor'], use_gpu=args.gpu_processor)
    images = decoded_images(records, config['training']['seed'])
    output = reserve_output(args.output_dir)
    results = {}
    for condition, quality in [('PNG', None), ('JPEG95', 95), ('JPEG75', 75)]:
        moments = {0: _Moments(), 1: _Moments()}
        probabilities = []
        with (output / f'{condition}.csv').open('x', encoding='utf-8', newline='') as stream:
            fields = ['path', 'label', 'subset', 'seed', 'amplitude_mae', 'phase_mae', 'probability_fake']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row, (image, seed) in zip(records, images):
                resized = image.resize((224, 224), Image.Resampling.BILINEAR)
                buffer = io.BytesIO()
                options = dict(format='PNG') if quality is None else dict(format='JPEG', quality=quality, subsampling=0)
                resized.save(buffer, **options)
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    inputs, amp, phase = prepare_inputs(decoded.convert('RGB'), processor, seed, config, device)
                probability = float(torch.sigmoid(model(*inputs)).item())
                probabilities.append(probability)
                amp_mean = float(amp.mean(dtype=np.float64))
                phase_mean = float(phase.mean(dtype=np.float64))
                moments[row['label']].add(amp_mean, phase_mean)
                writer.writerow(dict(path=row['path'], label=row['label'], subset=row['subset'], seed=seed,
                                     amplitude_mae=amp_mean, phase_mae=phase_mean, probability_fake=probability))
        results[condition] = dict(metrics=binary_metrics([r['label'] for r in records], probabilities),
                                  real=moments[0].as_dict(), generated=moments[1].as_dict())
    save_json(dict(protocol='paper_table_11', checkpoint=str(Path(args.checkpoint).resolve()),
                   seed=config['training']['seed'], image_count=len(records),
                   class_counts={str(label): sum(r['label'] == label for r in records) for label in [0, 1]},
                   results=results), output / 'results.json')


def training_template(args):
    rows = load_manifest(args.manifest, 'train')
    real = {r['path']: r for r in rows if r['label'] == 0}
    if len(real) < 1000:
        raise ValueError('The paper template requires 1000 distinct real training images')
    rng = np.random.default_rng(args.seed)
    candidates = sorted(real)
    selected = rng.choice(len(candidates), size=1000, replace=False)
    output = reserve_output(args.output_dir)
    template_manifest = output / 'template_sources.csv'
    write_manifest([real[candidates[i]] for i in selected], template_manifest)
    template = output / 'template.npy'
    frequency_template(template_manifest, template)
    frequency_regularize(args.manifest, template, output / 'manipulated', training_sensitivity=True)
    save_json(dict(seed=args.seed, template_images=1000, evaluation='training-data sensitivity'), output / 'sampling.json')


def summarize_runs(args):
    with Path(args.input).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        value = float(row['accuracy'])
        if not 0 <= value <= 1:
            raise ValueError('Accuracy must be a fraction in [0,1]')
        group = groups.setdefault(row['setting'], {})
        run = group.setdefault(row['run'], {})
        if row['subset'] in run:
            raise ValueError('Duplicate setting/run/subset')
        run[row['subset']] = value
    if not groups:
        raise ValueError('No runs supplied')
    output = {}
    reference_subsets = set(next(iter(next(iter(groups.values())).values())))
    for setting, runs in groups.items():
        if len(runs) != 5 or any(set(run) != reference_subsets for run in runs.values()):
            raise ValueError('Each setting requires five complete runs with identical subsets')
        values = np.array([[run[name] for name in sorted(reference_subsets)] for run in runs.values()])
        output[setting] = dict(subsets={name: dict(mean=float(values[:, i].mean()), sd=float(values[:, i].std(ddof=1)))
                                       for i, name in enumerate(sorted(reference_subsets))},
                               macro=dict(mean=float(values.mean(axis=1).mean()), sd=float(values.mean(axis=1).std(ddof=1))))
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(dict(runs=5, ddof=1, results=output), stream, indent=2)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    for name, function in [('benchmark', benchmark), ('matched-preprocessing', matched_preprocessing)]:
        command = commands.add_parser(name)
        command.add_argument('--checkpoint', required=True)
        command.add_argument('--manifest', required=True)
        command.add_argument('--device', default='auto')
        command.add_argument('--gpu-processor', action='store_true')
        command.add_argument('--output' if name == 'benchmark' else '--output-dir', required=True)
        command.set_defaults(function=function)
    command = commands.add_parser('training-template')
    command.add_argument('--manifest', required=True)
    command.add_argument('--output-dir', required=True)
    command.add_argument('--seed', type=int, default=42)
    command.set_defaults(function=training_template)
    command = commands.add_parser('summarize-runs')
    command.add_argument('--input', required=True)
    command.add_argument('--output', required=True)
    command.set_defaults(function=summarize_runs)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
