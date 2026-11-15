from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from dataset import IMAGE_EXTENSIONS, assert_disjoint, write_manifest


def scan_directories(directories, base):
    if isinstance(directories, str):
        directories = [directories]
    images = set()
    for name in directories:
        directory = Path(name)
        if not directory.is_absolute():
            directory = base / directory
        if not directory.is_dir():
            raise FileNotFoundError(f'Image directory not found: {directory}')
        images.update(str(p.resolve()) for p in directory.rglob('*')
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f'No images in {directories}')
    return sorted(images)


def prepare(spec, base, train_per_class=None, val_fraction=0.1, seed=42):
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must lie strictly between 0 and 1')
    if not spec or set(spec) - {'train', 'val', 'test', 'calibration'}:
        raise ValueError('Specification sections must be train, val, test, or calibration')
    if train_per_class is not None and ('train' not in spec or train_per_class < 1):
        raise ValueError('train_per_class requires a train section and a positive count')
    rng = random.Random(seed)
    result = {'train': [], 'val': [], 'test': [], 'calibration': []}
    for split, groups in spec.items():
        if not groups:
            raise ValueError(f'Empty {split} section')
        names = [group['subset'] for group in groups]
        if len(set(names)) != len(names):
            raise ValueError(f'Duplicate subset names in {split}')

        used = {0: set(), 1: set()}
        for index, group in enumerate(groups):
            for label, field in [(0, 'real'), (1, 'fake')]:
                paths = scan_directories(group[field], base)
                if split == 'train':
                    paths = [p for p in paths if p.casefold() not in used[label]]
                    rng.shuffle(paths)
                    if 'val' in spec:
                        if train_per_class is not None:
                            quota = train_per_class // len(groups) + (index < train_per_class % len(groups))
                            if quota < 1 or len(paths) < quota:
                                raise ValueError(f'{group["subset"]}/{field}: insufficient training images')
                            paths = paths[:quota]
                        used[label].update(p.casefold() for p in paths)
                        result['train'].extend(dict(path=p, label=label, subset=group['subset'], split='train') for p in paths)
                        continue
                    if train_per_class is not None:
                        quota = train_per_class // len(groups) + (index < train_per_class % len(groups))
                        if quota < 1:
                            raise ValueError('train_per_class must provide at least one image per subset')
                        n_val = max(1, round(quota * val_fraction / (1 - val_fraction)))
                        needed = quota + n_val
                        if len(paths) < needed:
                            raise ValueError(f'{group["subset"]}/{field}: need {needed} unique images '
                                             f'({quota} train + {n_val} val), have {len(paths)}')
                        paths = paths[:needed]
                    else:
                        n_val = max(1, min(len(paths) - 1, round(len(paths) * val_fraction)))
                    if len(paths) < 2:
                        raise ValueError(f'{group["subset"]}/{field}: need at least two images for train/validation')
                    used[label].update(p.casefold() for p in paths)
                    for position, path in enumerate(paths):
                        target = 'val' if position < n_val else 'train'
                        result[target].append(dict(path=path, label=label, subset=group['subset'], split=target))
                else:
                    result[split].extend(dict(path=p, label=label, subset=group['subset'], split=split) for p in paths)
    assert_disjoint(*(rows for rows in result.values() if rows))
    label_by_path = {}
    for rows in result.values():
        for row in rows:
            key = row['path'].casefold()
            if key in label_by_path and label_by_path[key] != row['label']:
                raise ValueError(f'Conflicting labels: {row["path"]}')
            label_by_path[key] = row['label']
    return result


def check_content_leakage(result):
    split_for_digest = {}
    for split, rows in result.items():
        for path in {row['path'] for row in rows}:
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            previous = split_for_digest.setdefault(digest, (split, path))
            if previous[0] != split:
                raise ValueError(f'Identical file content across {previous[0]} and {split}: {previous[1]} / {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, help='JSON mapping source splits to subset/real/fake directory lists')
    parser.add_argument('--output-dir', required=True, help='Must not already exist')
    parser.add_argument('--train-per-class', type=int, help='Final training count per class; explicit val preserves the full official validation split')
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check-content-leakage', action='store_true', help='Also detect byte-identical files across splits')
    args = parser.parse_args()
    path = Path(args.spec).resolve()
    with path.open(encoding='utf-8') as handle:
        spec = json.load(handle)
    result = prepare(spec, path.parent, args.train_per_class, args.val_fraction, args.seed)
    if args.check_content_leakage:
        check_content_leakage(result)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    summary = {}
    for split, rows in result.items():
        if rows:
            write_manifest(rows, output / f'{split}.csv')
            summary[split] = {str(label): sum(r['label'] == label for r in rows) for label in [0, 1]}
    with (output / 'sampling.json').open('w', encoding='utf-8') as handle:
        json.dump(dict(spec=spec, seed=args.seed, train_per_class=args.train_per_class,
                       val_fraction=args.val_fraction, check_content_leakage=args.check_content_leakage,
                       counts=summary), handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
