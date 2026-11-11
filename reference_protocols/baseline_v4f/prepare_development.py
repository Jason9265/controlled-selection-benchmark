"""Freeze v4f OEM regions and materialize only the development target labels."""
import hashlib, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / 'baseline_v2'
sys.path.insert(0, str(V2))
import prepare_data as prep

SALT = 'oem-v4f-fresh-development-confirmation-20260930:'
REGIONS = {'mahe': ('development',20,7), 'svaneti': ('confirmation',20,7),
           'tonga': ('confirmation',18,7)}
MATERIALIZE = ('mahe',)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def put(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError('refusing overwrite '+str(path))
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)

def main():
    ROOT.mkdir(exist_ok=True); (ROOT/'reports').mkdir(exist_ok=True)
    # Reuse only the v2 archive transport and geometry readers.  Its destination
    # is redirected before any fetch, while v2 remains the read-only cache.
    # Reuse only CRC-verified raw image bytes from the label-free candidate probe.
    # Selection remains defined solely by this version's salt and official splits.
    prep.ROOT=ROOT; prep.BASE=V2; prep.PILOT=ROOT.parent/'baseline_v4c'; prep.SALT=SALT
    prep.DEADLINE=time.monotonic()+14400
    prep.STATE={'archive_url':prep.URL,'archive_size_bytes':prep.SIZE,'network_bytes_this_run':0,
                'previous_network_bytes':0,'files':{},'rate_limit_events':[],'started_at_utc':prep.utc()}
    source=read(V2/'prepared/source_manifest.json')
    prior=[V2/'prepared/manifest.json', ROOT.parent/'baseline/prepared/manifest.json', ROOT.parent/'pilot/prepared/manifest.json',
           ROOT.parent/'baseline_v4c/prepared/manifest.json']
    old_originals=set(); seen={}
    for path in prior:
        if not path.exists(): continue
        m=read(path); sets=[m.get('source_train',[]),m.get('source_val',[])]
        sets += [r for t in m.get('targets',{}).values() for r in (t.get('pool',[]),t.get('test',[]))]
        for rows in sets:
            for row in rows:
                old_originals.add(row.get('original_member'))
                if row.get('original_rgb_sha256'): seen[row['original_rgb_sha256']]=row['original_member']
    plan={'protocol':'v4f OEM fresh development and confirmation', 'salt':SALT, 'regions':REGIONS,
          'selection':'ascending sha256(salt|official member filename), then official-val spatial buffer; labels excluded',
          'source_manifest_sha256':sha(V2/'prepared/source_manifest.json'),
          'prior_manifests':{str(p):sha(p) for p in prior if p.exists()},
          'excluded_prior_original_members':sorted(x for x in old_originals if x),
          'budget_fractions':[.01,.03,.05], 'source_models':'v2 frozen source best checkpoints',
          'development_rule':'Mahé only; no configuration selection after confirmation access',
          'confirmation_label_barrier':'Svaneti and Tonga labels remain unopened until the development training protocol is selected and recorded',
          'supersedes':'v4c was retained but not run because Rio budget counts collapsed to [1,1,1] patches; v4d stopped before target-label access because Chisinau overlapped v4c; v4e stopped after historical target exclusion was discovered'}
    put(ROOT/'prepared/selection_preregistration.json',plan)
    archive=prep.Archive()
    try:
        for split in ('train','val'):
            path=archive.fetch(prep.PREFIX+split+'.txt'); prep.OFFICIAL[split]=set(path.read_text().split())
        chosen={}
        for name,(role,ntrain,nval) in REGIONS.items():
            if name in {'chicago','tokyo','kyoto','paris','tyrolw','duesseldorf'}:
                raise RuntimeError('historically excluded target region '+name)
            region=prep.spatial_region(archive,name,ntrain,nval)
            if len(region['train'])<ntrain or len(region['val'])<nval: raise RuntimeError('insufficient buffered originals '+name)
            members={m['member'] for s in ('train','val') for m in region[s]}
            if members & old_originals: raise RuntimeError('prior original reuse '+name)
            region['role']=role;chosen[name]=region
        put(ROOT/'reports/target_frozen.json',{'regions':chosen,'label_dependent_selection':False,
            'selection_preregistration_sha256':sha(ROOT/'prepared/selection_preregistration.json')})
        manifest={'schema':'oem-v4f-development-only','num_classes':8,'classes':prep.CLASSES,'crop':256,
                  'raw_label_map':'0->255(ignore);1..8->0..7','source_train':[],'source_val':[], 'targets':{},
                  'source_manifest_sha256':sha(V2/'prepared/source_manifest.json'),
                  'target_freeze_sha256':sha(ROOT/'reports/target_frozen.json')}
        for key in ('source_train','source_val'):
            for row in source[key]:
                x=dict(row); x['image']=str((V2/row['image']).resolve());x['mask']=str((V2/row['mask']).resolve());manifest[key].append(x)
        audit={}
        for name in MATERIALIZE:
            region=chosen[name]
            rows,images=prep.prepare_region(archive,region,seen)
            for role in ('train','val'):
                for row in rows[role]:
                    row['image']=str((ROOT/row['image']).resolve());row['mask']=str((ROOT/row['mask']).resolve())
            manifest['targets'][name]={'pool':rows['train'],'test':rows['val'],'role':region['role']}
            n=len(rows['train']);budget=[int(__import__('math').ceil(n*x)) for x in (.01,.03,.05)]
            audit[name]={'role':region['role'],'pool_originals':len(region['train']),'test_originals':len(region['val']),
                         'pool_patches':n,'test_patches':len(rows['val']),'budgets':budget,
                         'actual_fractions':[x/n for x in budget],'minimum_bbox_gap_m':region['minimum_bbox_gap_m'],
                         'target_label_statistics_computed':False,'images':images}
        put(ROOT/'prepared/manifest.json',manifest)
        put(ROOT/'reports/data_guard.json',{'status':'PASS','manifest_sha256':sha(ROOT/'prepared/manifest.json'),
            'source_manifest_sha256':sha(V2/'prepared/source_manifest.json'),'targets':audit,
            'prior_original_overlap':0,'exact_rgb_duplicate_overlap':0,'target_label_class_statistics_computed':False,
            'target_masks_decoded_only_for_mapping_and_training_materialization':True,
            'materialized_regions':list(MATERIALIZE),'confirmation_access_began':False,
            'confirmation_regions_frozen_but_not_materialized':['svaneti','tonga']})
    finally: archive.close()

if __name__=='__main__': main()
