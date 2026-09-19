import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def load_records(path):
    path=Path(path)
    if path.suffix=='.jsonl':
        records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        data=json.loads(path.read_text())
        records=[dict(v,id=str(k)) for k,v in data.items()] if isinstance(data,dict) else data
    seen=set()
    for i,row in enumerate(records):
        row.setdefault('id',str(i)); row['id']=str(row['id'])
        if row['id'] in seen:
            raise ValueError('Duplicate record id: '+row['id'])
        seen.add(row['id'])
    return records


def rle_mask(rle,shape):
    values=np.asarray(rle,dtype=np.int64)
    if len(values)%2:
        raise ValueError('RLE requires (1-based start,length) pairs in C row-major order')
    out=np.zeros(int(np.prod(shape)),np.uint8)
    for start,length in values.reshape(-1,2):
        start-=1
        if start<0 or length<0 or start+length>out.size:
            raise ValueError('Invalid RLE extent')
        out[start:start+length]=1
    return out.reshape(shape)


def load_pair(row,root,mask_key='inpainting_mask',size=512):
    image=Image.open(Path(root)/row['image']).convert('RGB')
    if 'mask' in row and isinstance(row['mask'],str):
        mask=Image.open(Path(root)/row['mask']).convert('L')
        if mask.size!=image.size:
            raise ValueError('Mask/image native size mismatch')
    else:
        shape=tuple(row.get('mask_shape',[image.height,image.width]))
        mask=Image.fromarray(rle_mask(row[mask_key],shape)*255)
        if mask.size!=image.size:
            raise ValueError('Set mask_shape correctly; image and mask native sizes must agree')
    image=image.resize((size,size),Image.Resampling.BICUBIC)
    mask=mask.resize((size,size),Image.Resampling.NEAREST)
    mask=Image.fromarray((np.asarray(mask)>127).astype(np.uint8)*255)
    return image,mask


def sample_seed(seed,identity):
    return (seed+int(hashlib.sha256(str(identity).encode()).hexdigest()[:8],16))%(2**32)


def sample_name(identity):
    return hashlib.sha256(str(identity).encode()).hexdigest()[:24]+'.png'


class ManifestDataset(Dataset):
    def __init__(self,path,root,tokenizer,mask_key='inpainting_mask',size=512):
        self.records=load_records(path)
        self.root,self.tokenizer,self.mask_key,self.size=root,tokenizer,mask_key,size

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        row=self.records[index]
        image,mask=load_pair(row,self.root,self.mask_key,self.size)
        raw=np.array(image,copy=True)
        known=1-(np.array(mask)>127).astype(np.float32)
        x=torch.from_numpy(raw).permute(2,0,1).float()/127.5-1
        # BrushNet's PIL black fill is -1 after normalization; wave path explicitly masks to 0 itself.
        cond=torch.from_numpy(raw*known[:,:,None]).permute(2,0,1).float()/127.5-1
        ids=self.tokenizer(row.get('caption',''),max_length=self.tokenizer.model_max_length,padding='max_length',truncation=True,return_tensors='pt').input_ids[0]
        return dict(pixel_values=x,conditioning_pixel_values=cond,masks=torch.from_numpy(known)[None],input_ids=ids)
