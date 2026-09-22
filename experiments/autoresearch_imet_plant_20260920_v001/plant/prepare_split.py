import csv,hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(); raw=root/'raw'
train=raw/'train'; test=raw/'test'
classes=sorted([p.name for p in train.iterdir() if p.is_dir()])
rows=[]
for c in classes:
    for p in sorted((train/c).glob('*')):
        if p.is_file(): rows.append({'path':f'train/{c}/{p.name}','label':classes.index(c),'class':c,'id':p.stem})
# deterministic within-class 80/10/10 by sha
parts={'fit':[],'tune':[],'search':[]}
for r in rows:
    h=int(hashlib.sha256(r['path'].encode()).hexdigest()[:8],16)%100
    parts['fit' if h<80 else 'tune' if h<90 else 'search'].append(r)
for v in parts.values(): v.sort(key=lambda r:r['path'])
testrows=[{'path':f'test/{p.name}','id':p.stem} for p in sorted(test.glob('*')) if p.is_file()]
out=root/'data'; host=root/'host-labels'; out.mkdir(exist_ok=True); host.mkdir(exist_ok=True)
for split,rs in parts.items():
    with (out/f'{split}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','path']+(['label'] if split in ['fit','tune'] else []));w.writeheader()
        for r in rs:w.writerow({k:r[k] for k in w.fieldnames})
with (out/'test.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['id','path']);w.writeheader();w.writerows(testrows)
with (host/'search.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['id','label']);w.writeheader();w.writerows({'id':r['id'],'label':r['label']} for r in parts['search'])
manifest={'dataset':'plant-seedlings-classification','classes':classes,'counts':{k:len(v) for k,v in parts.items()},'test_count':len(testrows),'split_paths':{k:[r['path'] for r in v] for k,v in parts.items()},'search_labels_sha256':hashlib.sha256((host/'search.csv').read_bytes()).hexdigest()}
(root/'SPLIT.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:len(v) for k,v in parts.items()},indent=2),len(testrows),classes)
