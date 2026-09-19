"""Indexed uncompressed WebDataset tar: one sharding owner (Accelerator)."""
import hashlib
import json
import os
import random
import sqlite3
import tarfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch

FIELDS = {'image', 'caption', 'height', 'width', 'segmentation'}


def tar_files(directory):
    paths = sorted(Path(directory).resolve().glob('*.tar'))
    if not paths:
        raise ValueError(f'No uncompressed .tar files in {directory}')
    return paths


def fingerprint(paths):
    return hashlib.sha256(json.dumps([(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                                     for p in paths]).encode()).hexdigest()


def scan_tar(path):
    grouped = OrderedDict()
    with tarfile.open(path, 'r:') as archive:
        for member in archive:
            if not member.isfile() or '.' not in member.name:
                continue
            key, field = member.name.rsplit('.', 1)
            field = 'image' if field.lower() in ('jpg', 'jpeg', 'png', 'webp') else field
            if field in FIELDS:
                item = grouped.setdefault(key, {})
                if field in item:
                    raise ValueError(f'Duplicate field: {path}:{member.name}')
                item[field] = [member.offset_data, member.size]
    rows = [(key, fields) for key, fields in grouped.items() if FIELDS <= fields.keys()]
    if not rows:
        raise ValueError(f'No complete image/caption/height/width/segmentation samples: {path}')
    return rows


def build_index(directory, index, workers=4):
    paths = tar_files(directory)
    signature = fingerprint(paths)
    index = Path(index)
    if index.exists():
        with sqlite3.connect(index) as db:
            old = db.execute('SELECT value FROM meta WHERE name="fingerprint"').fetchone()[0]
        if old != signature:
            raise ValueError('Tar index is stale; use a NEW --tar_index path after changing data')
        return signature
    index.parent.mkdir(parents=True, exist_ok=True)
    print(f'Building tar index: {len(paths)} archives -> {index}', flush=True)
    temporary = index.with_name(index.name + f'.{os.getpid()}.tmp')
    try:
        with sqlite3.connect(temporary) as db, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            db.execute('CREATE TABLE meta (name TEXT PRIMARY KEY, value TEXT)')
            db.execute('CREATE TABLE samples (id INTEGER PRIMARY KEY, path TEXT, tar_id INT, key TEXT, fields TEXT)')
            db.execute('INSERT INTO meta VALUES (?,?)', ('fingerprint', signature))
            # Bounded concurrent scans: do not hold all tar metadata in memory.
            sample_id = 0
            for start in range(0, len(paths), max(1, workers)):
                chunk = paths[start:start + max(1, workers)]
                for tar_id, (path, rows) in enumerate(zip(chunk, pool.map(scan_tar, chunk)), start):
                    for key, fields in rows:
                        db.execute('INSERT INTO samples VALUES (?,?,?,?,?)',
                                   (sample_id, str(path), tar_id, key, json.dumps(fields)))
                        sample_id += 1
                print(f'Indexed {min(start+max(1,workers),len(paths))}/{len(paths)} archives, {sample_id} samples', flush=True)
            db.commit()
        os.replace(temporary, index)
    finally:
        if temporary.exists():
            temporary.unlink()
    return signature


class IndexedTarDataset(torch.utils.data.Dataset):
    def __init__(self, index, limit=None):
        self.index = str(Path(index).resolve())
        with sqlite3.connect(self.index) as db:
            count = db.execute('SELECT count(*) FROM samples').fetchone()[0]
            self.fingerprint = db.execute('SELECT value FROM meta WHERE name="fingerprint"').fetchone()[0]
        self.length = count if limit is None else min(count, int(limit))
        self.epoch = 0
        self._db = None
        self._handles = OrderedDict()

    def __len__(self):
        return self.length

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_db'], state['_handles'] = None, OrderedDict()
        return state

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._db is None:
            self._db = sqlite3.connect(f'file:{self.index}?mode=ro', uri=True)
        path, tar_id, key, fields = self._db.execute(
            'SELECT path,tar_id,key,fields FROM samples WHERE id=?', (int(index),)).fetchone()
        handle = self._handles.pop(path, None)
        if handle is None:
            handle = open(path, 'rb')
        self._handles[path] = handle
        while len(self._handles) > 8:
            self._handles.popitem(last=False)[1].close()
        row = {'__url__': path, '__key__': key, '_sample_id': index, '_tar_id': tar_id, '_epoch': self.epoch}
        for field, (offset, size) in json.loads(fields).items():
            handle.seek(offset)
            row[field] = handle.read(size)
            if len(row[field]) != size:
                raise IOError(f'Truncated tar member {path}:{key}.{field}')
        return row

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


class SeededCollator:
    """Augmentation is a pure function of sample id, epoch and run seed; resume-safe."""
    def __init__(self, collate, seed):
        self.collate, self.seed = collate, int(seed or 0)

    def __call__(self, examples):
        py_state, np_state = random.getstate(), np.random.get_state()
        batches = []
        try:
            for example in examples:
                token = f'{self.seed}:{example["_epoch"]}:{example["_sample_id"]}'
                seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], 'little')
                random.seed(seed)
                np.random.seed(seed)
                batches.append(self.collate([example]))
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
        return {key: torch.cat([batch[key] for batch in batches], 0) for key in batches[0]}
