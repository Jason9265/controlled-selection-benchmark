"""Expanded, source-support-audited OEM data with label-blind new targets.

Single persistent HTTP-range session, conservative request pacing, CRC/SHA checks,
atomic publication and resumable local caches. No old baseline file is modified.
Source expansion may inspect SOURCE labels only, within the retained full crop grid.
Target images are chosen using filename hashes and spatial metadata only.
"""
import sys
sys.dont_write_bytecode = True
import argparse, collections, email.utils, functools, hashlib, io, json, math, os, time, zipfile, zlib
from pathlib import Path
import numpy as np
from PIL import Image
from pyproj import CRS, Transformer
import requests
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'baseline'
PILOT=ROOT.parent/'pilot'
URL='https://zenodo.org/records/7223446/files/OpenEarthMap.zip'
SIZE=9099481727
PREFIX='OpenEarthMap_wo_xBD/'
SALT='oem-expanded-v2-20260930:'
PRIMARY=['bielefeld','koeln','vienna','christchurch','aachen','muenster','adelaide','melbourne','al_qurnah','khartoum']
BACKUP=['buenos_aires','dortmund','accra','dhaka','santiago','shanghai',
        'leilane_estates','svaneti','mahe','tonga']
TARGETS={'daressalaam':('confirmation',30,8),'zanzibar':('confirmation',30,8),'coxsbazar':('confirmation',30,8),'austin':('development',20,7)}
CLASSES=['bareland','rangeland','developed space','road','tree','water','agriculture','building']
DEADLINE=float('inf')
STATE={}

def utc(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def sha(data): return hashlib.sha256(data).hexdigest()
def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.part')
    with temp.open('wb') as f: f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def save(path,value): atomic(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
def progress(event=None):
    STATE['recorded_network_GB_all_runs']=(STATE['previous_network_bytes']+STATE['network_bytes_this_run'])/1e9
    STATE['last_update_utc']=utc()
    if event: STATE['last_event']=event
    save(ROOT/'reports/data_download.json',STATE)
    if event: print(json.dumps({'utc':utc(),'event':event,'network_GB':STATE['recorded_network_GB_all_runs']}),flush=True)
def freeze(path,value):
    if path.exists():
        old=json.loads(path.read_text())
        if old['content']!=value: raise RuntimeError('immutable frozen selection changed: '+str(path))
        return old
    result={'frozen_at_utc':utc(),'content':value};save(path,result);return result

class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.pos=0;self.cache=[];self.session=requests.Session();self.last_request=0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else SIZE+offset
        return self.pos
    def close(self): self.session.close();super().close()
    def read(self,n=-1):
        if n<0:n=SIZE-self.pos
        if not n:return b''
        start=self.pos;end=min(SIZE,start+n)
        for a,b,data in self.cache:
            if a<=start and end<=b:self.pos=end;return data[start-a:end-a]
        stop=min(SIZE,max(end,start+65536))
        for attempt in range(5):
            if time.monotonic()>=DEADLINE:raise TimeoutError('bounded runtime ended; safely resume')
            time.sleep(max(0,.55-(time.monotonic()-self.last_request)))
            self.last_request=time.monotonic()
            try:
                response=self.session.get(URL,headers={'Range':f'bytes={start}-{stop-1}'},timeout=90)
                STATE['network_bytes_this_run']+=len(response.content)
                if response.status_code in (429,503):
                    raw=response.headers.get('Retry-After','')
                    try:delay=float(raw)
                    except ValueError:
                        try:delay=email.utils.parsedate_to_datetime(raw).timestamp()-time.time()
                        except (ValueError,TypeError):delay=60*2**attempt
                    STATE['rate_limit_events'].append({'utc':utc(),'http_status':response.status_code,'retry_after':raw,'minimum_delay_seconds':max(delay,60),'action':'stop serial downloader; require cooldown before resume'})
                    STATE['resume_not_before_unix']=time.time()+max(delay,60)
                    progress('rate limit: stopped with no further requests')
                    raise RuntimeError('rate-limited; resume only after recorded cooldown')
                response.raise_for_status()
                if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{stop-1}/{SIZE}':raise ValueError('exact range not honored')
                data=response.content
                if len(data)!=stop-start:raise ValueError('truncated range')
                self.cache.append((start,stop,data));self.cache=self.cache[-4:]
                self.pos=end;return data[:end-start]
            except (requests.RequestException,ValueError):
                if attempt==4:raise
                time.sleep(min(2**(attempt+1),16))
        raise RuntimeError('range retries exhausted')

class Archive:
    def __init__(self):self.remote=RemoteZip();self.zip=zipfile.ZipFile(self.remote);self.names=set(self.zip.namelist())
    def close(self):self.zip.close();self.remote.close()
    def fetch(self,member):
        if '..' in Path(member).parts or Path(member).is_absolute():raise ValueError('unsafe member')
        info=self.zip.getinfo(member);dest=ROOT/'data/oem'/member;found=None;origin=None
        for candidate,kind in [(dest,'v2_crc_verified_cache'),(BASE/'data/oem'/member,'old_baseline_crc_verified_copy'),(PILOT/'data/oem'/member,'pilot_crc_verified_copy')]:
            if candidate.is_file() and candidate.stat().st_size==info.file_size:
                data=candidate.read_bytes()
                if zlib.crc32(data)&0xffffffff==info.CRC:found=data;origin=kind;break
        if found is None:
            found=self.zip.read(info);origin='network'
        if len(found)!=info.file_size or zlib.crc32(found)&0xffffffff!=info.CRC:raise RuntimeError('CRC/length mismatch '+member)
        if origin!='v2_crc_verified_cache':atomic(dest,found)
        STATE['files'][member]={'bytes':len(found),'crc32':info.CRC,'sha256':sha(found),'origin':origin,'verified_at_utc':utc()}
        progress('verified '+member)
        return dest


@functools.lru_cache(maxsize=128)
def transformer_for(source,destination):
    return Transformer.from_crs(source,destination,always_xy=True)

def point_xy(m,row,col):
    if 'native_transform' not in m:return [m['bbox'][0]+col*m['gsd'][0],m['bbox'][3]-row*m['gsd'][1]]
    x,sx,_,y,_,minus_sy=m['native_transform']
    transformer=transformer_for(m['native_epsg'],m['epsg'])
    return list(transformer.transform(x+col*sx,y+row*minus_sy))

def geometry(m,row,col):
    corners=np.array([point_xy(m,row,col),point_xy(m,row,col+256),point_xy(m,row+256,col+256),point_xy(m,row+256,col)])
    shifted=corners-corners[0]
    area=abs(float(np.dot(shifted[:,0],np.roll(shifted[:,1],-1))-np.dot(shifted[:,1],np.roll(shifted[:,0],-1))))/2
    sides=[]
    for a,b in [((row,col),(row,col+256)),((row,col+256),(row+256,col+256)),((row+256,col+256),(row+256,col)),((row+256,col),(row,col))]:
        points=np.array([point_xy(m,a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f) for f in np.linspace(0,1,5 if 'native_transform' in m else 2)])
        sides.append(float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()))
    center=point_xy(m,row+128,col+128)
    if not np.isfinite([*corners.ravel(),*center,area,*sides]).all() or area<=0:raise ValueError('nonfinite metric geometry')
    return {'center_xy':center,'corners_xy':corners.tolist(),'area_m2':area,'edge_lengths_m':sides}

METRIC_BY_REGION={}
def metadata(member):
    path=ROOT/'data/oem'/member;region=Path(member).parts[1]
    with Image.open(path) as image:
        tags=image.tag_v2;scale=tags.get(33550);tie=tags.get(33922);keys=tags.get(34735,())
        direct={keys[k]:keys[k+3] for k in range(4,len(keys),4) if keys[k+1]==0 and keys[k+2]==1}
        if not scale or not tie or 34264 in tags or direct.get(1025,1)!=1 or tags.get(274,1)!=1:raise RuntimeError('unsupported GeoTIFF transform '+member)
        native=direct.get(3072) if direct.get(1024)==1 else direct.get(2048) if direct.get(1024)==2 else None
        if not native or native==32767:raise RuntimeError('unknown authoritative CRS '+member)
        crs=CRS.from_epsg(native);sx,sy=map(float,scale[:2]);w,h=image.size
        x=float(tie[3]-tie[0]*sx);y=float(tie[4]+tie[1]*sy)
        if not np.isfinite([sx,sy,x,y]).all() or min(sx,sy)<=0:raise RuntimeError('invalid affine '+member)
        if image.mode!='RGB':raise RuntimeError('non RGB imagery '+member)
    result={'member':member,'region':region,'group':path.stem,'width':w,'height':h,'native_epsg':native,'epsg':native,'units':'metre','gsd':[sx,sy],'bbox':[x,y-h*sy,x+w*sx,y],'sha256':STATE['files'][member]['sha256']}
    if crs.is_projected and native!=3857:
        if direct.get(3076) not in (None,9001) or not all(a.unit_name in ('metre','meter') and abs(a.unit_conversion_factor-1)<1e-12 for a in crs.axis_info[:2]):raise RuntimeError('nonmetric projected CRS '+member)
        result['buffer_width_m']=256*max(sx,sy)
    elif (crs.is_geographic and native==4326 and direct.get(2054)==9102) or native==3857:
        to_geographic=transformer_for(native,4326)
        lon,lat=to_geographic.transform(x+w*sx/2,y-h*sy/2)
        computed=(32600 if lat>=0 else 32700)+int((lon+180)//6)+1
        epsg=METRIC_BY_REGION.setdefault(region,computed)
        metric=CRS.from_epsg(epsg)
        if not metric.is_projected or not all(a.unit_name=='metre' for a in metric.axis_info[:2]):raise RuntimeError('cannot establish metric destination '+member)
        transform=Transformer.from_crs(native,epsg,always_xy=True)
        nativebox=result['bbox'];geographicbox=to_geographic.transform_bounds(*nativebox,densify_pts=21);area=metric.area_of_use
        if not (area.west<=geographicbox[0]<=geographicbox[2]<=area.east and area.south<=geographicbox[1]<=geographicbox[3]<=area.north):raise RuntimeError('region outside fixed UTM zone '+member)
        box=list(transform.transform_bounds(*nativebox,densify_pts=21))
        result.update(epsg=epsg,metric_epsg=epsg,native_units='degree' if native==4326 else 'Web Mercator map metre; not ground metric',native_transform=[x,sx,0.,y,0.,-sy],native_bbox=nativebox,bbox=[box[0]-.001,box[1]-.001,box[2]+.001,box[3]+.001],projection_note=f'explicit pyproj EPSG{native} to fixed region UTM; 21 edge subdivisions; 1mm outward bbox; original pixels not resampled')
        patches=[geometry(result,r,c) for r in range(0,h-255,256) for c in range(0,w-255,256)]
        if not patches:raise RuntimeError('no full patches '+member)
        edge=max(max(g['edge_lengths_m']) for g in patches)*1.000001
        result['gsd']=[edge/256,edge/256];result['buffer_width_m']=edge
    else:raise RuntimeError('unsupported geographic/projected CRS '+member)
    old=METRIC_BY_REGION.setdefault(region,result['epsg'])
    if old!=result['epsg']:raise RuntimeError('region mixes metric CRS '+region)
    return result

def gap(a,b):
    if a['epsg']!=b['epsg']:raise ValueError('cross CRS distance not permitted')
    x,y=a['bbox'],b['bbox']
    return math.hypot(max(x[0]-y[2],y[0]-x[2],0),max(x[1]-y[3],y[1]-x[3],0))

def ranked(archive,split,region):
    allowed=OFFICIAL[split]
    names=[n for n in archive.names if n.startswith(f'{PREFIX}{region}/images/') and n.endswith('.tif') and Path(n).name in allowed and n.replace('/images/','/labels/') in archive.names]
    return sorted(names,key=lambda n:sha((SALT+n).encode()))

def spatial_region(archive,region,train_limit=None,val_limit=None):
    train_names=ranked(archive,'train',region);val_names=ranked(archive,'val',region)
    selected_val=val_names if val_limit is None else val_names[:val_limit]
    if not train_names or not selected_val:raise RuntimeError('official split missing '+region)
    val=[]
    for n in selected_val:archive.fetch(n);val.append(metadata(n))
    kept=[];excluded=[];scanned=[]
    for n in train_names:
        archive.fetch(n);m=metadata(n);scanned.append(n)
        distances=[gap(m,h) for h in val]
        if all(d>=max(m['buffer_width_m'],h['buffer_width_m']) for d,h in zip(distances,val)):kept.append(m)
        else:excluded.append({'member':n,'minimum_bbox_gap_m':min(distances),'reason':'256-patch-side buffer from fixed official val'})
        if train_limit is not None and len(kept)>=train_limit:break
    if not kept:raise RuntimeError('no buffered training/pool originals '+region)
    return {'region':region,'train':kept,'val':val,'excluded_buffer':excluded,'scanned_train_members':scanned,'available_official_train':len(train_names),'available_official_val':len(val_names),'requested_train':train_limit,'requested_val':val_limit,'minimum_bbox_gap_m':min(gap(a,b) for a in kept for b in val),'label_used_for_spatial_selection':False}

def source_diagnostic(archive,region):
    output={}
    for split in ('train','val'):
        support=np.zeros(8,dtype=np.int64);robust=np.zeros(8,dtype=np.int64);substantial=np.zeros(8,dtype=np.int64);pixels=np.zeros(8,dtype=np.int64);perimage=[]
        for m in region[split]:
            n=m['member'].replace('/images/','/labels/');path=archive.fetch(n)
            with Image.open(path) as image:y=np.asarray(image)
            if y.shape!=(m['height'],m['width']) or not np.isin(y,np.arange(9)).all():raise RuntimeError('source label format mismatch '+n)
            retained=y[:(m['height']//256)*256,:(m['width']//256)*256]
            hist=np.bincount(retained.ravel(),minlength=9)[1:9]
            support+=(hist>0);robust+=(hist>=256);substantial+=(hist>=1024);pixels+=hist
            perimage.append({'group':m['group'],'class_pixels_in_retained_crops':hist.tolist(),'retained_pixel_count':int(retained.size),'raw_label_sha256':STATE['files'][n]['sha256']})
        output[split]={'original_images':len(region[split]),'class_image_support_ge1pixel':support.tolist(),'class_image_support_ge256pixels':robust.tolist(),'class_image_support_ge1024pixels':substantial.tolist(),'retained_class_pixel_counts':pixels.tolist(),'images':perimage}
    return output

def aggregate(diagnostics):
    output={}
    for split in ('train','val'):
        output[split]={'original_images':sum(v[split]['original_images'] for v in diagnostics.values()),'class_image_support_ge1pixel':np.sum([v[split]['class_image_support_ge1pixel'] for v in diagnostics.values()],axis=0).tolist(),'class_image_support_ge256pixels':np.sum([v[split]['class_image_support_ge256pixels'] for v in diagnostics.values()],axis=0).tolist(),'class_image_support_ge1024pixels':np.sum([v[split]['class_image_support_ge1024pixels'] for v in diagnostics.values()],axis=0).tolist(),'retained_class_pixel_counts':np.sum([v[split]['retained_class_pixel_counts'] for v in diagnostics.values()],axis=0).tolist()}
    output['meets_gate']=(output['train']['original_images']>=300 and output['val']['original_images']>=40 and min(output['train']['class_image_support_ge1024pixels'])>=20 and min(output['val']['class_image_support_ge1024pixels'])>=5 and min(output['train']['retained_class_pixel_counts'])>=65536 and min(output['val']['retained_class_pixel_counts'])>=16384)
    return output

def save_png(path,pixels):
    buffer=io.BytesIO();Image.fromarray(pixels).save(buffer,format='PNG');data=buffer.getvalue()
    if not path.exists() or sha(path.read_bytes())!=sha(data):atomic(path,data)
    return sha(data)

def prepare_region(archive,region,seen):
    output={};image_audit=[]
    for split in ('train','val'):
        records=[]
        for m in region[split]:
            if time.monotonic()>=DEADLINE:raise TimeoutError('preparation deadline; resume')
            label_member=m['member'].replace('/images/','/labels/');labelpath=archive.fetch(label_member)
            with Image.open(ROOT/'data/oem'/m['member']) as image:x=np.asarray(image)
            with Image.open(labelpath) as label:y=np.asarray(label)
            if x.shape[:2]!=y.shape or not np.isin(y,np.arange(9)).all():raise RuntimeError('alignment/raw range failure '+m['member'])
            digest=sha(x.tobytes())
            if digest in seen:raise RuntimeError('exact original RGB duplicate '+m['member']+' / '+seen[digest])
            seen[digest]=m['member'];mapped=np.where(y==0,255,y.astype(np.int16)-1).astype(np.uint8)
            count=0
            for row in range(0,m['height']-255,256):
                for col in range(0,m['width']-255,256):
                    identity=f'{m["group"]}_r{row}_c{col}';ip=ROOT/'prepared/images'/f'{identity}.png';mp=ROOT/'prepared/labels'/f'{identity}.png'
                    ih=save_png(ip,x[row:row+256,col:col+256]);mh=save_png(mp,mapped[row:row+256,col:col+256]);g=geometry(m,row,col)
                    records.append({'id':identity,'group':m['group'],'region':m['region'],'image':str(ip.relative_to(ROOT)),'mask':str(mp.relative_to(ROOT)),'area_pixels':65536,'row':row,'col':col,'epsg':m['epsg'],'native_epsg':m['native_epsg'],'image_sha256':ih,'mask_sha256':mh,'original_image_sha256':m['sha256'],'original_rgb_sha256':digest,'original_member':m['member'],**g})
                    count+=1
            image_audit.append({**m,'official_split':split,'patch_count':count,'raw_label_sha256':STATE['files'][label_member]['sha256'],'rgb_sha256':digest})
        output[split]=records
    return output,image_audit

def disjoint_region_envelopes(regions):
    boxes={}
    for name,r in regions.items():
        transformed=[Transformer.from_crs(m['epsg'],4326,always_xy=True).transform_bounds(*m['bbox'],densify_pts=21) for s in ('train','val') for m in r[s]]
        boxes[name]=[min(b[0] for b in transformed),min(b[1] for b in transformed),max(b[2] for b in transformed),max(b[3] for b in transformed)]
    overlaps=[]
    for i,(a,x) in enumerate(boxes.items()):
        for b,y in list(boxes.items())[i+1:]:
            if max(x[0],y[0])<=min(x[2],y[2]) and max(x[1],y[1])<=min(x[3],y[3]):overlaps.append([a,b])
    if overlaps:raise RuntimeError('geographic region envelopes overlap '+str(overlaps))
    return {'wgs84_region_envelopes':boxes,'overlapping_region_pairs':overlaps}

def adjacency(records):
    centers=np.array([p['center_xy'] for p in records]);widths=np.array([max(p['edge_lengths_m']) for p in records]);groups=np.array([p['group'] for p in records])
    dist=np.max(np.abs(centers[:,None]-centers[None,:]),axis=2)
    threshold=np.maximum(widths[:,None],widths[None,:])*1.00001
    edges=np.triu((dist<=threshold)&(dist>0),1);pairs=np.argwhere(edges);cross=sum(groups[a]!=groups[b] for a,b in pairs)
    near=dist.copy();np.fill_diagonal(near,np.inf)
    return {'metric':'L-infinity projected center distance','edge_rule':'distance <= max(two measured patch side lengths) * 1.00001','epsg':sorted({p['epsg'] for p in records}),'patch_count':len(records),'undirected_edges':int(len(pairs)),'cross_original_edges':int(cross),'within_original_edges':int(len(pairs)-cross),'nodes_with_neighbor':int(np.count_nonzero(np.any(edges|edges.T,axis=1))),'nearest_center_distance_m_min':float(near.min()),'edge_index_pairs':pairs.tolist(),'patch_ids_in_index_order':[p['id'] for p in records],'target_label_statistics_computed':False}

OFFICIAL={}
def main():
    global DEADLINE,STATE,OFFICIAL
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=('source','all','probe'),default='source');p.add_argument('--seconds',type=int,default=14400);a=p.parse_args();DEADLINE=time.monotonic()+a.seconds
    report=ROOT/'reports/data_download.json';old=json.loads(report.read_text()) if report.exists() else {}
    if time.time()<old.get('resume_not_before_unix',0):raise RuntimeError('server-requested cooldown still active; no network started')
    STATE={'archive_url':URL,'archive_size_bytes':SIZE,'full_archive_digest_verified':False,'network_bytes_this_run':0,'previous_network_bytes':old.get('previous_network_bytes',0)+old.get('network_bytes_this_run',0),'files':{},'rate_limit_events':old.get('rate_limit_events',[]),'started_at_utc':utc()}
    plan=freeze(ROOT/'reports/data_selection_plan.json',{'salt':SALT,'source_primary':PRIMARY,'source_backup_order':BACKUP,'targets':{k:list(v) for k,v in TARGETS.items()},'source_minimum_originals':[300,40],'source_minimum_retained_crop_class_image_support':[20,5],'source_support_pixel_threshold':1024,'source_minimum_class_total_pixels':[65536,16384],'target_budget_fractions':[.01,.03,.05],'target_selection_label_blind':True,'source_expansion_may_use_source_labels':True,'source_validation_population':'class-support enriched selected source regions; not natural population estimate'})
    archive=Archive()
    try:
        for split in ('train','val'):
            path=archive.fetch(PREFIX+split+'.txt');OFFICIAL[split]=set(path.read_text().split())
        if OFFICIAL['train']&OFFICIAL['val']:raise RuntimeError('official split overlap')
        inventory={r:{s:len(ranked(archive,s,r)) for s in ('train','val')} for r in PRIMARY+BACKUP+list(TARGETS)}
        save(ROOT/'reports/data_available_counts.json',inventory)
        if a.scope=='probe':
            probes={};errors={}
            for r in PRIMARY+list(TARGETS):
                try:
                    choices=ranked(archive,'train',r)
                    if not choices:raise RuntimeError('no paired image/label archive members in official train split')
                    n=choices[0];archive.fetch(n);probes[r]=metadata(n)
                except Exception as exc:errors[r]=str(exc)
            save(ROOT/'reports/data_metadata_probe.json',{'probes':probes,'errors':errors,'targets_unchanged':True})
            if errors:raise RuntimeError('metadata probe failures: '+str(errors))
            return
        regions={};diagnostics={};events=[];unavailable={}
        for region in PRIMARY+BACKUP:
            if region in BACKUP and aggregate(diagnostics)['meets_gate']:break
            if not inventory[region]['train'] or not inventory[region]['val']:
                unavailable[region]={'official_archive_paired_members':inventory[region],'action':'unavailable source candidate excluded; count/support gate remains mandatory and fixed backup order retained'}
                save(ROOT/'reports/data_unavailable_source_candidates.json',unavailable)
                continue
            if region in BACKUP:events.append({'added_source_region':region,'reason':'previous source-only count/support gate failed','preceding_gate':aggregate(diagnostics)})
            progress('SOURCE_REGION_START '+region)
            r=spatial_region(archive,region);regions[region]=r
            diagnostic=source_diagnostic(archive,r);diagnostics[region]=diagnostic
            save(ROOT/f'reports/data_source_support_{region}.json',diagnostic)
            support=aggregate(diagnostics)
            save(ROOT/'reports/data_source_support_progress.json',{'classes':CLASSES,'regions_completed':list(regions),'aggregate':support,'data_dependent_expansion_events':events})
            progress('SOURCE_REGION_DONE '+region+' '+json.dumps(support))
        support=aggregate(diagnostics)
        if not support['meets_gate']:raise RuntimeError('source gate not met after fixed backups: '+json.dumps(support))
        source_freeze=freeze(ROOT/'reports/data_source_frozen.json',{'regions':regions,'source_support':support,'source_support_classes':CLASSES,'data_dependent_expansion_events':events,'validation_sampling':'class-support enriched; not unbiased natural population estimate','region_geographic_separation':disjoint_region_envelopes(regions)})
        source={'schema':'oem-expanded-v2','num_classes':8,'classes':CLASSES,'crop':256,'raw_label_map':'0->255(ignore);1..8->0..7','edge_policy':'all complete nonoverlapping 256 crops; incomplete borders dropped; no resizing','source_train':[],'source_val':[],'targets':{},'selection_plan_timestamp_utc':plan['frozen_at_utc'],'source_freeze_timestamp_utc':source_freeze['frozen_at_utc'],'source_selection_sha256':sha((ROOT/'reports/data_source_frozen.json').read_bytes()),'source_validation_sampling':'class-support enriched selected regions; source-only model selection; not natural-population estimate','official_test_used':False}
        source_audit=[];seen={}
        for r in regions.values():
            rows,images=prepare_region(archive,r,seen);source['source_train'].extend(rows['train']);source['source_val'].extend(rows['val']);source_audit.extend(images)
        source_path=ROOT/'prepared/source_manifest.json'
        if source_path.exists() and json.loads(source_path.read_text())!=source:raise RuntimeError('source manifest receipt changed; refusing overwrite')
        if not source_path.exists():save(source_path,source)
        save(ROOT/'reports/data_source_audit.json',{'manifest_sha256':sha(source_path.read_bytes()),'source_support':support,'originals':source_audit,'source_regions':list(regions),'source_original_counts':{s:sum(len(r[s]) for r in regions.values()) for s in ('train','val')},'source_patch_counts':{s:len(source['source_'+s]) for s in ('train','val')},'data_dependent_expansion_events':events,'source_label_statistics_only':True})
        progress('SOURCE_MANIFEST_READY '+str(source_path))
        if a.scope=='source':return
        target_regions={}
        for name,(role,ntrain,nval) in TARGETS.items():
            r=spatial_region(archive,name,ntrain,nval)
            if len(r['train'])<20:raise RuntimeError(name+' cannot supply >=20 buffered pool originals; fixed target not replaced')
            r['role']=role;target_regions[name]=r
        # Source labels were permitted above. No target label is decoded before this freeze.
        targetfreeze=freeze(ROOT/'reports/data_target_frozen.json',{'regions':target_regions,'target_label_based_selection':False,'budget_fractions':[.01,.03,.05],'all_region_separation':disjoint_region_envelopes({**regions,**target_regions})})
        full={**source,'targets':{},'target_freeze_timestamp_utc':targetfreeze['frozen_at_utc'],'target_selection_sha256':sha((ROOT/'reports/data_target_frozen.json').read_bytes())};target_audit={}
        for name,r in target_regions.items():
            rows,images=prepare_region(archive,r,seen)
            # `test` is only the engine field name; these are official-val holdouts.
            full['targets'][name]={'pool':rows['train'],'test':rows['val'],'role':r['role']}
            if r['role']=='development':
                full['targets'][name].update(reserved=True,note='reserved; no tuning/adaptation in this batch; final source-only evaluation')
            N=len(rows['train']);budgets=[math.ceil(N*f) for f in [.01,.03,.05]]
            target_audit[name]={'role':r['role'],'pool_originals':len(r['train']),'holdout_originals':len(r['val']),'pool_patches':N,'holdout_patches':len(rows['val']),'budgets_cumulative_patches':budgets,'exact_budget_fractions':[b/N for b in budgets],'requested_holdout_originals':r['requested_val'],'holdout_shortfall':max(0,r['requested_val']-len(r['val'])),'buffer_exclusions':r['excluded_buffer'],'minimum_bbox_gap_m':r['minimum_bbox_gap_m'],'originals':images,'target_label_statistics_computed':False}
            save(ROOT/f'reports/data_pool_adjacency_{name}.json',adjacency(rows['train']))
        assert source['source_train']==full['source_train'] and source['source_val']==full['source_val']
        save(ROOT/'prepared/manifest.json',full)
        save(ROOT/'reports/data_target_audit.json',{'manifest_sha256':sha((ROOT/'prepared/manifest.json').read_bytes()),'source_lists_unchanged':True,'target_regions':target_audit,'target_class_statistics_computed':False})
        progress('FULL_MANIFEST_READY '+str(ROOT/'prepared/manifest.json'))
    finally:archive.close()

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if STATE:
            STATE['fatal_error']=str(exc);progress('FAILED_CLOSED '+str(exc))
        raise
