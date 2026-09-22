import os,json,socket
from pathlib import Path
import torch,numpy as np,pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

def vector(texts):
 return torch.tensor(HashingVectorizer(n_features=1024,alternate_sign=False).transform(texts).toarray(),dtype=torch.float32)

def train(data_dir,artifact_dir,budget_seconds,seed):
 assert torch.cuda.is_available()
 assert not any(k in os.environ for k in ['OPENAI_API_KEY','AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY'])
 assert not Path('/var/run/docker.sock').exists()
 assert not Path('/workspace').exists()
 assert not Path('/data/search.csv').read_text().splitlines()[0].endswith('author')
 try: socket.create_connection(('1.1.1.1',443),timeout=1)
 except OSError: pass
 else: raise RuntimeError('candidate unexpectedly has network egress')
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 d=pd.read_csv(Path(data_dir)/'fit.csv').iloc[:1024]
 x=vector(d.text.fillna('').tolist()).cuda();y=torch.tensor([['EAP','HPL','MWS'].index(v) for v in d.author],device='cuda')
 model=torch.nn.Linear(1024,3).cuda();opt=torch.optim.Adam(model.parameters(),lr=.01)
 for _ in range(5):
  opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward();opt.step()
 torch.save({k:v.cpu() for k,v in model.state_dict().items()},Path(artifact_dir)/'gpu.pt')
 return {'architecture':'GPU hashing linear smoke only','steps':5,'seed':seed,'device':torch.cuda.get_device_name(),'fit_loss':float(loss.item()),'isolation_assertions_passed':True}

def predict(artifact_dir,texts):
 model=torch.nn.Linear(1024,3);model.load_state_dict(torch.load(Path(artifact_dir)/'gpu.pt',map_location='cpu',weights_only=True));model.eval()
 with torch.no_grad():return model(vector(texts)).softmax(1).numpy()
