from __future__ import annotations

import csv
import hashlib
import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def load_manifest(path, required_split=None):
    path = Path(path).resolve()
    records, seen = [], set()
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'path', 'label', 'subset', 'split'} <= set(reader.fieldnames or []):
            raise ValueError('Manifest must contain path,label,subset,split columns')
        for line, row in enumerate(reader, 2):
            image = Path(row['path'])
            if not image.is_absolute():
                image = path.parent / image
            image = image.resolve()
            if row['label'] not in {'0', '1'}:
                raise ValueError(f'{path}:{line}: label must be 0 (real) or 1 (fake)')
            split = row['split'].strip()
            if split not in {'train', 'val', 'test', 'calibration'}:
                raise ValueError(f'{path}:{line}: invalid split {split!r}')
            if required_split and split != required_split:
                raise ValueError(f'{path}:{line}: expected {required_split}, found {split}')
            if not row['subset'].strip():
                raise ValueError(f'{path}:{line}: empty subset')
            if not image.is_file():
                raise FileNotFoundError(f'{path}:{line}: {image}')
            key = (str(image).casefold(), row['subset'])
            if key in seen:
                raise ValueError(f'{path}:{line}: duplicate path within subset: {image}')
            seen.add(key)
            records.append(dict(path=str(image), label=int(row['label']),
                                subset=row['subset'].strip(), split=split))
    if not records:
        raise ValueError(f'Empty manifest: {path}')
    labels_by_path = {}
    for row in records:
        key = row['path'].casefold()
        if key in labels_by_path and labels_by_path[key] != row['label']:
            raise ValueError(f'Conflicting labels: {row["path"]}')
        labels_by_path[key] = row['label']
    return records


def write_manifest(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'label', 'subset', 'split'])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in records)


def assert_disjoint(*record_sets):
    seen = set()
    for records in record_sets:
        current = {r['path'].casefold() for r in records}
        overlap = seen & current
        if overlap:
            raise ValueError(f'Data split leakage: {next(iter(overlap))}')
        seen |= current


def stable_seed(path, seed=42):

    digest = hashlib.blake2b(Path(path).read_bytes(), digest_size=8).digest()
    return (int.from_bytes(digest, 'little') + int(seed)) % (2 ** 32)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


class RandomJPEGCompression:
    def __init__(self, quality_min=60, quality_max=100, p=0.4):
        if not 1 <= quality_min <= quality_max <= 100 or not 0 <= p <= 1:
            raise ValueError('Invalid JPEG quality or probability')
        self.quality_min, self.quality_max, self.p = quality_min, quality_max, p

    def __call__(self, image):
        if random.random() >= self.p:
            return image
        buffer = io.BytesIO()
        image.convert('RGB').save(buffer, format='JPEG',
                                  quality=random.randint(self.quality_min, self.quality_max))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert('RGB').copy()


class ImageAugmentation:
    def __init__(self, mode='train', img_size=224, jpeg_p=0.4):
        self.mode, self.img_size = mode, img_size
        self.jpeg = RandomJPEGCompression(p=jpeg_p)

    def __call__(self, image):
        image = image.convert('RGB').resize((self.img_size, self.img_size), Image.Resampling.BICUBIC)
        if self.mode == 'train':
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            image = self.jpeg(image)
        return image


def rgb_tensor(rgb):
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
    return (tensor - torch.tensor(CLIP_MEAN)[:, None, None]) / torch.tensor(CLIP_STD)[:, None, None]


class Corruption:

    def __init__(self, kind='clean', value=0, seed=42):
        if kind not in {'clean', 'jpeg', 'crop', 'blur', 'noise'}:
            raise ValueError(f'Unknown corruption: {kind}')
        if (kind == 'jpeg' and not 1 <= value <= 100 or
                kind == 'crop' and not 0 < value <= 1 or
                kind in {'blur', 'noise'} and value < 0):
            raise ValueError('Invalid corruption strength')
        self.kind, self.value, self.seed = kind, value, seed

    def __call__(self, image, sample_seed):
        if self.kind == 'jpeg':
            return RandomJPEGCompression(int(self.value), int(self.value), p=1)(image)
        if self.kind == 'crop':
            w, h = image.size
            cw, ch = max(1, round(w * self.value)), max(1, round(h * self.value))
            left, top = (w - cw) // 2, (h - ch) // 2
            return image.crop((left, top, left + cw, top + ch))
        if self.kind == 'blur' and self.value:
            return image.filter(ImageFilter.GaussianBlur(self.value))
        if self.kind == 'noise' and self.value:
            rng = np.random.default_rng((sample_seed + self.seed) % (2 ** 32))
            arr = np.asarray(image, dtype=np.float32)
            arr = np.clip(arr + rng.normal(0, self.value, arr.shape), 0, 255).round().astype(np.uint8)
            return Image.fromarray(arr)
        return image


class TriStreamDataset(Dataset):
    def __init__(self, records, processor, mode='train', img_size=224, seed=42,
                 jpeg_p=0.4, corruption=None, residual_scale=(1.0, 1.0)):
        self.records = records
        self.processor = processor
        self.mode, self.seed = mode, seed
        self.augment = ImageAugmentation(mode, img_size, jpeg_p)
        self.corruption = corruption or Corruption()
        if len(residual_scale) != 2 or any(x <= 0 for x in residual_scale):
            raise ValueError('Residual scales must be positive')
        self.residual_scale = residual_scale

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        try:
            sample_seed = random.randrange(2 ** 32) if self.mode == 'train' else stable_seed(row['path'], self.seed)
            with Image.open(row['path']) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
            image = self.corruption(image, sample_seed)
            image = self.augment(image)
            rgb, amp, phase = self.processor.process_single_image(
                np.asarray(image), is_train_mode=self.mode == 'train', seed=sample_seed)
            amp = torch.from_numpy(np.ascontiguousarray(amp, dtype=np.float32)).unsqueeze(0)
            phase = torch.from_numpy(np.ascontiguousarray(phase, dtype=np.float32)).unsqueeze(0)
            return (rgb_tensor(rgb), amp / self.residual_scale[0],
                    phase / self.residual_scale[1], torch.tensor(row['label'], dtype=torch.float32))
        except Exception as exc:
            raise RuntimeError(f'Failed to process image: {row["path"]}') from exc


def get_transforms(mode='train', img_size=224, **unused):

    return ImageAugmentation(mode, img_size), rgb_tensor
