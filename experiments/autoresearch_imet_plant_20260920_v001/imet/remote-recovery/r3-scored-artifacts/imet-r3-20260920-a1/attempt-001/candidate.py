import os, csv, json, time, random
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

N_LABELS = 3474
IMAGE_SIZE = 160
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _path(root, value):
    value = str(value)
    return value if os.path.isabs(value) else os.path.join(root, value)


def _read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    names = list(rows[0].keys())
    preferred = ['image_path', 'image', 'filename', 'file', 'path', 'id']
    col = next((x for x in preferred if x in names), None)
    if col is None:
        col = next((x for x in names if any(str(r[x]).lower().endswith(e) for e in ('.jpg', '.jpeg', '.png', '.webp'))), names[0])
    return [r[col] for r in rows]


def _labels(path):
    try:
        import scipy.sparse as sp
        a = sp.load_npz(path).toarray()
    except Exception:
        z = np.load(path, allow_pickle=False)
        key = 'arr_0' if 'arr_0' in z else ('labels' if 'labels' in z else z.files[0])
        a = z[key]
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != N_LABELS:
        raise ValueError('labels must be a two-dimensional N x 3474 indicator matrix')
    return a


def _image(path, train=False):
    im = Image.open(path).convert('RGB')
    if train and random.random() < 0.5:
        im = ImageOps.mirror(im)
    im = ImageOps.fit(im, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1)).float()


class ImageSet(Dataset):
    def __init__(self, root, paths, labels=None, train=False):
        self.root = root
        self.paths = paths
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        x = _image(_path(self.root, self.paths[index]), self.train)
        if self.labels is None:
            return x
        return x, torch.from_numpy(self.labels[index])


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.BatchNorm2d(cout)
        self.skip = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                     if cin != cout or stride != 1 else nn.Identity())

    def forward(self, x):
        y = torch.relu(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return torch.relu(y + self.skip(x))


class SmallResNet(nn.Module):
    def __init__(self, output_bias=None):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.body = nn.Sequential(
            Block(32, 32, 1), Block(32, 64, 2), Block(64, 64, 1),
            Block(64, 128, 2), Block(128, 128, 1), Block(128, 192, 2), Block(192, 192, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.25)
        self.fc = nn.Linear(192, N_LABELS)
        if output_bias is not None:
            with torch.no_grad():
                self.fc.bias.copy_(torch.as_tensor(output_bias, dtype=torch.float32))

    def forward(self, x):
        x = self.body(self.stem(x))
        return self.fc(self.drop(self.pool(x).flatten(1)))


def _probs(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for x in loader:
            out.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    return np.concatenate(out, axis=0)


def _micro_f1(pred, truth):
    tp = np.logical_and(pred, truth > 0.5).sum()
    fp = np.logical_and(pred, truth <= 0.5).sum()
    fn = np.logical_and(~pred, truth > 0.5).sum()
    return float(2.0 * tp / max(1.0, 2.0 * tp + fp + fn))


def _threshold_score(prob, truth, threshold):
    pred = prob >= threshold
    empty = ~pred.any(axis=1)
    if empty.any():
        pred[empty] = False
        pred[empty, np.argmax(prob[empty], axis=1)] = True
    return _micro_f1(pred, truth)


def _predict_labels(prob, threshold):
    pred = prob >= float(threshold)
    empty = ~pred.any(axis=1)
    if empty.any():
        pred[empty] = False
        pred[empty, np.argmax(prob[empty], axis=1)] = True
    return pred.astype(np.int8)


def train(data_dir, artifact_dir, budget_seconds, seed):
    _seed(seed)
    os.makedirs(artifact_dir, exist_ok=True)
    started = time.time()
    fit_paths = _read_csv(os.path.join(data_dir, 'fit.csv'))
    tune_paths = _read_csv(os.path.join(data_dir, 'tune.csv'))
    y_fit = _labels(os.path.join(data_dir, 'fit_labels.npz'))
    y_tune = _labels(os.path.join(data_dir, 'tune_labels.npz'))
    if len(fit_paths) != len(y_fit) or len(tune_paths) != len(y_tune):
        raise ValueError('image and label row counts differ')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prevalence = np.clip(y_fit.mean(axis=0), 1e-4, 1.0 - 1e-4)
    bias = np.log(prevalence / (1.0 - prevalence)).astype(np.float32)
    model = SmallResNet(bias).to(device)
    positives = y_fit.sum(axis=0)
    negatives = len(y_fit) - positives
    pos_weight = np.clip(np.sqrt((negatives + 1.0) / (positives + 1.0)), 1.0, 5.0).astype(np.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight).to(device))
    batch_size = 64 if device.type == 'cuda' else 24
    workers = 2 if os.name != 'nt' else 0
    train_loader = DataLoader(ImageSet(data_dir, fit_paths, y_fit, True), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=device.type == 'cuda')
    tune_loader = DataLoader(ImageSet(data_dir, tune_paths, None, False), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == 'cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    max_epochs = max(3, min(12, int((budget_seconds - 20) / max(20.0, len(fit_paths) / max(1, batch_size) * 0.18))))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=2e-6)
    best_score = -1.0
    best_threshold = 0.2
    best_state = None
    trajectory = []
    for epoch in range(max_epochs):
        model.train()
        total = 0.0
        batches = 0
        for x, y in train_loader:
            if time.time() - started > budget_seconds * 0.88:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
        scheduler.step()
        prob = _probs(model, tune_loader, device)
        epoch_score = -1.0
        epoch_threshold = 0.2
        for threshold in np.arange(0.02, 0.501, 0.01):
            score = _threshold_score(prob, y_tune, float(threshold))
            if score > epoch_score:
                epoch_score, epoch_threshold = score, float(threshold)
        trajectory.append({'epoch': epoch + 1, 'loss': total / max(1, batches), 'tune_micro_f1': epoch_score, 'threshold': epoch_threshold})
        if epoch_score > best_score:
            best_score = epoch_score
            best_threshold = epoch_threshold
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if time.time() - started > budget_seconds * 0.90:
            break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(best_state, os.path.join(artifact_dir, 'model.pt'))
    selected_epoch = int(np.argmax([x['tune_micro_f1'] for x in trajectory]) + 1) if trajectory else 0
    metadata = {
        'architecture': 'SmallResNet-192 with prevalence-aware output bias initialization',
        'image_size': IMAGE_SIZE,
        'labels': N_LABELS,
        'seed': int(seed),
        'threshold': float(best_threshold),
        'selected_epoch': selected_epoch,
        'tune_micro_f1': float(best_score),
        'trajectory': trajectory,
        'optimizer': 'AdamW(lr=3e-4,weight_decay=1e-4)',
        'loss': 'BCEWithLogitsLoss with clipped square-root positive weights',
        'public_models': [],
        'preprocessing': 'RGB, ImageOps.fit, train horizontal flip, ImageNet normalization'
    }
    with open(os.path.join(artifact_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f)


def predict(artifact_dir, paths):
    with open(os.path.join(artifact_dir, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallResNet().to(device)
    model.load_state_dict(torch.load(os.path.join(artifact_dir, 'model.pt'), map_location=device))
    loader = DataLoader(ImageSet('/data', list(paths), None, False), batch_size=64, shuffle=False, num_workers=2 if os.name != 'nt' else 0)
    return _predict_labels(_probs(model, loader, device), meta['threshold'])
