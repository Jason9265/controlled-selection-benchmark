"""Freeze LoveDA v2 names offline; --download is an explicit network opt-in.
No target label statistics or target-dependent sample selection. Original cache is read-only.
"""
import argparse
import hashlib
import io
import json
import os
import struct
import time
import zlib
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'baseline'
OUT = ROOT / 'prepared_loveda'
SOURCES = ROOT / 'sources'
REPORTS = ROOT / 'reports'
SALT = 'loveda-v2-source-support-independent-urban-20260929'
SPEC = {'source_train': ('Train', 'Rural', 320), 'source_val': ('Val', 'Rural', 64),
        'pool': ('Train', 'Urban', 60), 'test': ('Val', 'Urban', 24)}
CLASSES = ['background', 'building', 'road', 'water', 'barren', 'forest', 'agriculture']

def sha(data): return hashlib.sha256(data).hexdigest()
def readjson(path): return json.loads(path.read_text())
def save(path, value, frozen=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists():
        if readjson(path) != value: raise RuntimeError('frozen output changed: '+str(path))
        return
    tmp = path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)

def raw_ok(path, m):
    if not path.exists(): return False
    b = path.read_bytes()
    return len(b) == m['size'] and zlib.crc32(b) == m['crc32']

def freeze():
    inventories = {s: readjson(OLD/'sources'/f'loveda_{s}_zip_manifest.json') for s in ('Train','Val')}
    old = readjson(OLD/'prepared_loveda/manifest.json')
    excluded = sorted({r['original_member'] for t in old['targets'].values() for k in ('pool','test') for r in t[k]})
    byname = {m['filename']: m for inv in inventories.values() for m in inv['members']}
    orders = {}
    for role, (split, domain, count) in SPEC.items():
        names = [n for n in byname if n.startswith(f'{split}/{domain}/images_png/') and n.endswith('.png')]
        if domain == 'Urban': names = [n for n in names if n not in excluded]
        orders[role] = sorted(names, key=lambda n: (sha((SALT+'|'+n).encode()), n))
        if len(orders[role]) < count: raise RuntimeError('insufficient candidates '+role)
    plan = {'salt':SALT,'rule':'ascending sha256(salt|full ZIP member filename), filename tie break',
            'initial_counts':{r:s[2] for r,s in SPEC.items()}, 'source_extension_batch':32,
            'ordered_candidates':orders,'excluded_old_urban_originals':excluded,
            'old_manifest_sha256':sha((OLD/'prepared_loveda/manifest.json').read_bytes()),
            'inventory_sha256':{s:sha((OLD/'sources'/f'loveda_{s}_zip_manifest.json').read_bytes()) for s in inventories},
            'source_support':{'train_images_per_class':20,'val_images_per_class':5,
                              'pixels_per_supporting_image':1024,'train_pixels_per_class':65536,'val_pixels_per_class':16384},
            'source_retention':'retain non-all-ignore 256 crops; support aggregated over retained crops per original',
            'target_retention':'all 16 crops; no target class histogram or label-dependent selection'}
    save(OUT/'selection_preregistration.json',plan,frozen=True)
    estimates={}
    for role, (_,_,count) in SPEC.items():
        members=[byname[n] for image in orders[role][:count] for n in (image,image.replace('/images_png/','/masks_png/'))]
        cache=sum(m['compressed_size'] for m in members if raw_ok(OLD/'data/loveda'/m['filename'],m) or raw_ok(OUT/'originals'/m['filename'],m))
        estimates[role]={'originals':count,'max_patches':16*count,'compressed_bytes':sum(m['compressed_size'] for m in members),
                         'raw_bytes':sum(m['size'] for m in members),'compressed_bytes_not_cached':sum(m['compressed_size'] for m in members)-cache}
    save(REPORTS/'loveda_data_plan.json',{'selection_sha256':sha((OUT/'selection_preregistration.json').read_bytes()),
             'initial_estimates':estimates,'network_started':False,'support_status':'requires source mask decoding; not inferred from filenames'})
    return plan,inventories,byname

class Fetcher:
    def __init__(self, inventories, allow_network, interval):
        self.inventories=inventories; self.allow_network=allow_network; self.interval=interval
        self.session=requests.Session(); self.session.trust_env=False; self.last=0
        self.state_path=SOURCES/'loveda_download_state.json'
        self.state=readjson(self.state_path) if self.state_path.exists() else {'files':{},'offsets':{}}
        receipt=OLD/'prepared_loveda/download_records.json'
        self.old_sha={r['filename']:r['sha256'] for r in readjson(receipt)} if receipt.exists() else {}
    def range(self,url,a,b):
        if not self.allow_network: raise RuntimeError('cache miss; download requires --download after shared-host clearance')
        stop=SOURCES/'loveda_rate_limit.json'
        if stop.exists() and time.time()<readjson(stop)['retry_not_before']:
            raise RuntimeError('server Retry-After cooldown still active')
        for attempt in range(4):
            time.sleep(max(0,self.interval-(time.monotonic()-self.last)))
            try:
                response=self.session.get(url,headers={'Range':f'bytes={a}-{b}'},timeout=90);self.last=time.monotonic()
                if response.status_code==429:
                    raw=response.headers.get('Retry-After'); delay=60
                    if raw:
                        try: delay=max(0,float(raw))
                        except ValueError:
                            try: delay=max(0,parsedate_to_datetime(raw).timestamp()-time.time())
                            except (ValueError,TypeError): pass
                    save(stop,{'status':429,'retry_after':raw,'retry_not_before':time.time()+delay,
                               'action':'exit immediately; resume serially after cooldown','url':url})
                    raise RuntimeError('HTTP 429; stopped, Retry-After='+str(raw))
                response.raise_for_status()
                if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {a}-{b}/') or len(response.content)!=b-a+1:
                    raise RuntimeError('Range response mismatch; refusing full archive')
                return response.content
            except requests.RequestException:
                if attempt==3: raise
                time.sleep(2**(attempt+1))
    def get(self,m):
        name=m['filename']
        for path in (OLD/'data/loveda'/name,OUT/'originals'/name):
            if raw_ok(path,m):
                digest=sha(path.read_bytes())
                if path.is_relative_to(OLD) and name in self.old_sha and digest!=self.old_sha[name]:
                    raise RuntimeError('old download receipt SHA mismatch '+name)
                previous=self.state['files'].get(name)
                if previous and previous['sha256']!=digest: raise RuntimeError('cached original SHA mismatch '+name)
                self.state['files'][name]={'sha256':digest,'crc32':m['crc32'],'path':os.path.relpath(path,ROOT)}
                save(self.state_path,self.state)
                return path,digest
        dst=OUT/'originals'/name
        if dst.exists(): raise RuntimeError('invalid existing original; refusing overwrite '+str(dst))
        url=self.inventories[name.split('/')[0]]['url']
        offset=self.state['offsets'].get(name)
        if offset is None:
            header=self.range(url,m['header_offset'],m['header_offset']+29)
            if header[:4]!=b'PK\x03\x04': raise RuntimeError('bad local ZIP signature')
            n,e=struct.unpack_from('<HH',header,26);offset=m['header_offset']+30+n+e
            self.state['offsets'][name]=offset;save(self.state_path,self.state)
        raw=self.range(url,offset,offset+m['compressed_size']-1)
        if m['compression'] not in (0,8):raise RuntimeError('unsupported ZIP compression')
        payload=zlib.decompress(raw,-15) if m['compression']==8 else raw
        if len(payload)!=m['size'] or zlib.crc32(payload)!=m['crc32']:raise RuntimeError('CRC mismatch '+name)
        dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix('.png.part');tmp.write_bytes(payload);tmp.replace(dst)
        digest=sha(payload);self.state['files'][name]={'sha256':digest,'crc32':m['crc32'],'path':os.path.relpath(dst,ROOT)};save(self.state_path,self.state)
        return dst,digest

def png(path,array):
    buffer=io.BytesIO();Image.fromarray(array).save(buffer,format='PNG');payload=buffer.getvalue()
    if path.exists():
        if sha(path.read_bytes())!=sha(payload):raise RuntimeError('existing PNG differs '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.png.part');tmp.write_bytes(payload);tmp.replace(path)
    return sha(payload)

def prepare_original(name,role,fetcher,byname,seen):
    ip,image_sha=fetcher.get(byname[name]); maskname=name.replace('/images_png/','/masks_png/')
    mp,mask_sha=fetcher.get(byname[maskname])
    with Image.open(ip) as im: rgb=np.asarray(im.convert('RGB'))
    with Image.open(mp) as im: raw=np.asarray(im)
    if rgb.shape!=(1024,1024,3) or raw.shape!=(1024,1024) or not np.isin(raw,np.arange(8)).all():
        raise RuntimeError('format/label-map error '+name)
    rgb_sha=sha(rgb.tobytes())
    if rgb_sha in seen and seen[rgb_sha]!=name:raise RuntimeError('duplicate original RGB '+name+' / '+seen[rgb_sha])
    seen[rgb_sha]=name
    label=np.where(raw==0,255,raw.astype(np.int16)-1).astype(np.uint8)
    group=name.replace('/images_png/','_').replace('/','_').removesuffix('.png')
    rows=[];counts=np.zeros(7,dtype=np.int64) if role.startswith('source_') else None
    for y in range(0,1024,256):
        for x in range(0,1024,256):
            crop=label[y:y+256,x:x+256]
            if counts is not None:
                valid=crop!=255
                if not valid.any():continue
                counts+=np.bincount(crop[valid],minlength=7)
            identity=f'{group}_r{y}_c{x}';image=OUT/'images'/f'{identity}.png';mask=OUT/'labels'/f'{identity}.png'
            ih=png(image,rgb[y:y+256,x:x+256]);mh=png(mask,crop)
            row={'id':identity,'group':group,'image':str(image.relative_to(ROOT)),'mask':str(mask.relative_to(ROOT)),
                 'area_pixels':65536,'area_m2':None,'row':y,'col':x,'image_sha256':ih,'mask_sha256':mh,
                 'original_image_sha256':image_sha,'original_mask_sha256':mask_sha,'original_rgb_sha256':rgb_sha,'original_member':name}
            if counts is not None:row.update(valid_pixels=int(valid.sum()),has_valid_pixels=True)
            rows.append(row)
    return rows,counts

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');parser.add_argument('--prepare-cache',action='store_true')
    parser.add_argument('--scope',choices=('source','all'),default='all');parser.add_argument('--interval',type=float,default=0.75)
    args=parser.parse_args()
    if args.interval<0.5:parser.error('interval must be >= 0.5 seconds')
    plan,inventories,byname=freeze()
    print(json.dumps(readjson(REPORTS/'loveda_data_plan.json')['initial_estimates']),flush=True)
    if not (args.download or args.prepare_cache):print('OFFLINE PLAN READY; no network performed');return
    fetcher=Fetcher(inventories,args.download,args.interval);seen={}
    manifest={'schema':'loveda-v2-source-support','num_classes':7,'classes':CLASSES,'crop':256,
              'raw_label_map':'0->255(ignore);1..7->0..6','split_seed':SALT,'source_train':[],'source_val':[],'targets':{},
              'spatial_claim':'official split and original-file isolation only; no reliable coordinates or spatial buffer claim',
              'target_selection_policy':'filename hash, exclude all old Urban pool/test originals; no target label statistics',
              'source_selection_policy':'fixed hash prefix, extend by 32 only for prespecified source support failure',
              'selection_preregistration_sha256':sha((OUT/'selection_preregistration.json').read_bytes())}
    audit={'source':{},'target_histograms_computed':False,'created_utc':datetime.now(timezone.utc).isoformat()}
    for role in ('source_train','source_val'):
        names=plan['ordered_candidates'][role];n=SPEC[role][2];processed=0;per_image=[]
        while True:
            for name in names[processed:n]:
                rows,counts=prepare_original(name,role,fetcher,byname,seen);manifest[role].extend(rows)
                per_image.append({'original':name,'retained_patches':len(rows),'pixels_per_class':counts.tolist()})
                print(role,len(per_image),name,flush=True)
            processed=n;matrix=np.array([r['pixels_per_class'] for r in per_image]);support=(matrix>=1024).sum(0);totals=matrix.sum(0)
            minimum_images=20 if role=='source_train' else 5;minimum_pixels=65536 if role=='source_train' else 16384
            passed=bool(np.all(support>=minimum_images) and np.all(totals>=minimum_pixels))
            audit['source'][role]={'originals':n,'retained_patches':len(manifest[role]),'supporting_originals_per_class':support.tolist(),
                                   'pixels_per_class':totals.tolist(),'passed':passed,'per_original':per_image}
            save(REPORTS/'loveda_data_source_audit.json',audit)
            if passed:break
            if n==len(names):raise RuntimeError('source support failed after exhausting fixed order '+role)
            n=min(n+32,len(names))
        manifest.setdefault('selected_source_originals',{})[role]=names[:n]
    save(OUT/'source_manifest.json',manifest,frozen=True)
    print('SOURCE READY',OUT/'source_manifest.json',flush=True)
    if args.scope=='source':return
    manifest['targets']={'urban':{'pool':[],'test':[],'role_confirmation':'official Train/Urban pool; official Val/Urban evaluation only; all previous Urban originals excluded'}}
    for role in ('pool','test'):
        names=plan['ordered_candidates'][role][:SPEC[role][2]]
        for i,name in enumerate(names):
            rows,_=prepare_original(name,role,fetcher,byname,seen);manifest['targets']['urban'][role].extend(rows)
            print(role,i+1,name,flush=True)
    groups=[{r['group'] for r in rows} for rows in (manifest['source_train'],manifest['source_val'],manifest['targets']['urban']['pool'],manifest['targets']['urban']['test'])]
    if any(groups[i]&groups[j] for i in range(4) for j in range(i)):raise RuntimeError('split overlap')
    if any(r['original_member'] in set(plan['excluded_old_urban_originals']) for k in ('pool','test') for r in manifest['targets']['urban'][k]):raise RuntimeError('old target overlap')
    save(OUT/'manifest.json',manifest,frozen=True)
    audit['target_original_counts']={'pool':60,'test':24};audit['manifest_sha256']=sha((OUT/'manifest.json').read_bytes())
    save(REPORTS/'loveda_data_audit.json',audit);print('DONE',OUT/'manifest.json',flush=True)

if __name__=='__main__':main()
