import os, csv, json, time, random
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

N_LABELS = 3474
IMAGE_SIZE = 160


def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def _path(root, p):
    p = str(p)
    return p if os.path.isabs(p) else os.path.join(root, p)


def _read_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    names = list(rows[0].keys())
    preferred = ['image_path','image','filename','file','path','id']
    col = next((x for x in preferred if x in names), None)
    if col is None:
        col = next((x for x in names if any(str(r[x]).lower().endswith(e) for e in ['.jpg','.jpeg','.png','.webp'])), names[0])
    return [r[col] for r in rows]


def _labels(path):
    try:
        import scipy.sparse as sp
        return sp.load_npz(path).toarray().astype(np.float32)
    except Exception:
        z = np.load(path, allow_pickle=False)
        if 'arr_0' in z: a = z['arr_0']
        elif 'labels' in z: a = z['labels']
        else: a = z[z.files[0]]
        return np.asarray(a, dtype=np.float32)


def _img(path, train=False):
    im = Image.open(path).convert('RGB')
    if train and random.random() < 0.5: im = ImageOps.mirror(im)
    im = ImageOps.fit(im, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.BILINEAR, centering=(0.5,0.5))
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - np.array([0.485,0.456,0.406], np.float32)) / np.array([0.229,0.224,0.225], np.float32)
    return torch.from_numpy(a.transpose(2,0,1)).float()


class ImgSet(Dataset):
    def __init__(self, root, paths, y=None, train=False):
        self.root, self.paths, self.y, self.train = root, paths, y, train
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        x = _img(_path(self.root, self.paths[i]), self.train)
        return (x, torch.from_numpy(self.y[i])) if self.y is not None else x


class Block(nn.Module):
    def __init__(self, a, b, stride):
        super().__init__()
        self.c1=nn.Conv2d(a,b,3,stride,1,bias=False); self.n1=nn.BatchNorm2d(b)
        self.c2=nn.Conv2d(b,b,3,1,1,bias=False); self.n2=nn.BatchNorm2d(b)
        self.skip=nn.Sequential(nn.Conv2d(a,b,1,stride,bias=False),nn.BatchNorm2d(b)) if (a!=b or stride!=1) else nn.Identity()
    def forward(self,x):
        y=torch.relu(self.n1(self.c1(x))); y=self.n2(self.c2(y)); return torch.relu(y+self.skip(x))


class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(3,32,3,2,1,bias=False),nn.BatchNorm2d(32),nn.ReLU())
        self.body=nn.Sequential(Block(32,32,1),Block(32,64,2),Block(64,64,1),Block(64,128,2),Block(128,128,1),Block(128,192,2),Block(192,192,1))
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.25),nn.Linear(192,N_LABELS))
    def forward(self,x): return self.head(self.body(self.stem(x)))


def _predict_model(model, loader, device):
    model.eval(); out=[]
    with torch.no_grad():
        for x in loader: out.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    return np.concatenate(out,0)


def _f1(p, y):
    tp=np.logical_and(p,y>0.5).sum(); fp=np.logical_and(p,y<=0.5).sum(); fn=np.logical_and(~p,y>0.5).sum()
    return float(2*tp/max(1,2*tp+fp+fn))


def train(data_dir, artifact_dir, budget_seconds, seed):
    _seed(seed); os.makedirs(artifact_dir, exist_ok=True); start=time.time()
    fitp=_read_csv(os.path.join(data_dir,'fit.csv')); tunep=_read_csv(os.path.join(data_dir,'tune.csv'))
    yf=_labels(os.path.join(data_dir,'fit_labels.npz')); yt=_labels(os.path.join(data_dir,'tune_labels.npz'))
    yf=yf.reshape(len(fitp),-1); yt=yt.reshape(len(tunep),-1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SmallResNet().to(device)
    pos=yf.sum(0); neg=len(yf)-pos
    pw=np.clip(np.sqrt((neg+1)/(pos+1)),1.0,5.0).astype(np.float32)
    lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pw).to(device))
    bs=64 if device.type=='cuda' else 24
    tr=DataLoader(ImgSet(data_dir,fitp,yf,True),batch_size=bs,shuffle=True,num_workers=2,pin_memory=device.type=='cuda')
    va=DataLoader(ImgSet(data_dir,tunep,None,False),batch_size=bs,shuffle=False,num_workers=2,pin_memory=device.type=='cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12, eta_min=2e-6)
    best=-1.0; best_state=None; best_t=0.2; trajectory=[]
    epochs=max(3,min(12,int((budget_seconds-20)//max(20, len(fitp)/max(1,bs)*0.18))))
    epochs=max(3,epochs)
    for ep in range(epochs):
        model.train(); total=0.0; nb=0
        for x,y in tr:
            if time.time()-start > budget_seconds*0.88: break
            opt.zero_grad(set_to_none=True); z=model(x.to(device)); loss=lossfn(z,y.to(device)); loss.backward(); opt.step(); total+=loss.item(); nb+=1
        sched.step()
        pr=_predict_model(model,va,device); local_best=-1; local_t=0.2
        for t in np.arange(0.02,0.501,0.01):
            q=pr>=float(t); empty=~q.any(1); q[empty]=False; q[empty,np.argmax(pr[empty],1)]=True
            score=_f1(q,yt)
            if score>local_best: local_best,local_t=score,float(t)
        trajectory.append({'epoch':ep+1,'loss':total/max(1,nb),'tune_micro_f1':local_best,'threshold':local_t})
        if local_best>best:
            best=local_best; best_t=local_t; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); torch.save(model.state_dict(),os.path.join(artifact_dir,'model.pt'))
    meta={'architecture':'SmallResNet-192','image_size':IMAGE_SIZE,'labels':N_LABELS,'seed':seed,'threshold':best_t,'selected_epoch':int(np.argmax([x['tune_micro_f1'] for x in trajectory])+1),'tune_micro_f1':best,'trajectory':trajectory,'optimizer':'AdamW(lr=3e-4,weight_decay=1e-4)','public_models':[],'preprocessing':'RGB, ImageOps.fit, ImageNet normalization, train horizontal flip'}
    with open(os.path.join(artifact_dir,'meta.json'),'w') as f: json.dump(meta,f)


def predict(artifact_dir, paths):
    with open(os.path.join(artifact_dir,'meta.json')) as f: meta=json.load(f)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=SmallResNet().to(device)
    model.load_state_dict(torch.load(os.path.join(artifact_dir,'model.pt'),map_location=device)); model.eval()
    root='/data'; ds=ImgSet(root,paths,None,False); dl=DataLoader(ds,batch_size=64,shuffle=False,num_workers=2)
    pr=_predict_model(model,dl,device); q=pr>=float(meta['threshold']); empty=~q.any(1)
    if empty.any(): q[empty]=False; q[empty,np.argmax(pr[empty],1)]=True
    return q.astype(np.int8)
