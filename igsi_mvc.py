'IGSI-MVC: multi-view clustering with input correction and subspace intervention.\n\nRun a bundled dataset:\n    python igsi_mvc.py run --dataset Dermatology --condition shared\n\nOther commands:\n    python igsi_mvc.py train --help\n    python igsi_mvc.py evaluate --help\n'
import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits


def mlp(widths):
    layers = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(widths) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class Autoencoders(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.enc = nn.ModuleList([mlp([d, 512, 512, 1024, 256]) for d in dims])
        self.dec = nn.ModuleList([mlp([256, 1024, 512, 512, d]) for d in dims])

    def forward(self, xs):
        low = [enc(x) for enc, x in zip(self.enc, xs)]
        return low, [dec(h) for dec, h in zip(self.dec, low)]


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(128), nn.LayerNorm(128)
        self.attn = nn.MultiheadAttention(128, 4, dropout=0, batch_first=True)
        self.ff = mlp([128, 512, 128])

    def forward(self, tokens):
        h = self.n1(tokens)
        tokens = tokens + self.attn(h, h, h, need_weights=False)[0]
        return tokens + self.ff(self.n2(tokens))


class IGSI(nn.Module):
    def __init__(self, dims, clusters):
        super().__init__()
        self.v, self.k = len(dims), clusters
        self.backbone = Autoencoders(dims)
        self.high = nn.ModuleList([mlp([256, 256, 128]) for _ in dims])
        self.centers = nn.Parameter(torch.randn(clusters, 128) * .03)
        self.fuse = mlp([128, 64, 1])
        self.view_embedding = nn.Parameter(torch.randn(1, self.v, 128) * .02)


        # Keep the initialization stream consistent across configurations.
        torch.randn(1, self.v, 128)
        self.blocks = nn.ModuleList([AttentionBlock(), AttentionBlock()])
        for _ in dims:
            mlp([128, 256, 128])
        self.repair = nn.ModuleList([mlp([128, 256, 128]) for _ in dims])

    def probabilities(self, h):
        return (F.normalize(h, dim=-1) @ F.normalize(self.centers, dim=1).T / .2).softmax(-1)

    def forward(self, xs):
        low, recs = self.backbone(xs)
        h = torch.stack([F.normalize(head(x), dim=1)
                         for head, x in zip(self.high, low)], 1)
        mean = h.mean(1)
        centered = mean - mean.mean(0, keepdim=True)
        with torch.no_grad():
            basis = torch.linalg.svd(centered.detach(), full_matrices=False).Vh[:2].T
        common = (centered @ basis) @ basis.T
        residual = h - .5 * common[:, None, :]
        residual = residual + .1 * torch.stack([r(h[:, v])
                                                for v, r in enumerate(self.repair)], 1)
        intervened = F.normalize(residual, dim=2)
        tokens = intervened + self.view_embedding
        for block in self.blocks:
            tokens = block(tokens)
        hv = F.normalize(intervened + .1 * tokens, dim=2)
        weights = self.fuse(hv).squeeze(2).softmax(1)
        z = F.normalize((hv * weights[:, :, None]).sum(1), dim=1)
        return dict(low=low, recs=recs, h=h, hv=hv, z=z, weights=weights,
                    removed_common=common, q=self.probabilities(z),
                    qviews=self.probabilities(hv))


def reconstruction(xs, recs):
    'Normalized multi-view reconstruction error.'
    return sum((x - r).square().mean() / x.square().mean().clamp_min(1e-8)
               for x, r in zip(xs, recs)) / len(xs)


def column_contrast(a, b):
    'Contrast corresponding assignment columns across two views.'
    k = len(a)
    h = F.normalize(torch.cat([a, b]), dim=1)
    logits = h @ h.T
    logits = logits.masked_fill(torch.eye(2 * k, device=a.device, dtype=torch.bool), -1e4)
    targets = (torch.arange(2 * k, device=a.device) + k) % (2 * k)
    return F.cross_entropy(logits, targets)


def neighbor_geometry(h, z, positive, negative):
    'Weighted neighborhood contrastive objective.'
    h, z = F.normalize(h, dim=1, eps=1e-8), F.normalize(z, dim=1, eps=1e-8)
    cross, within = h @ z.T / .5, h @ h.T / .5
    floor = torch.log(z.new_tensor(1e-45)) * 2
    a = torch.where((negative > 0) & (within != 0),
                    2 * within + negative.clamp_min(1e-45).log(), floor)
    b = torch.where((negative > 0) & (cross != 0),
                    2 * cross + negative.clamp_min(1e-45).log(), floor)
    return (-(positive * cross).sum()
            + torch.logsumexp(torch.cat([a, b], 1), 1).sum()) / len(z)


def objective(xs, out, relations, prior, shared, teacher=None):
    'Warm-up or refinement objective selected by teacher availability.'
    rec = reconstruction(xs, out['recs'])
    pairs = list(itertools.combinations(range(len(xs)), 2))
    assign = sum(column_contrast(out['qviews'][:, v].T, out['qviews'][:, w].T)
                 for v, w in pairs) / len(pairs)
    zero = out['z'].sum() * 0
    prior_loss, geometry = zero, zero
    if teacher is None:
        mass = out['qviews'].mean(0)
        if shared:
            mass = mass.sort(1).values
            pi = prior.sort().values[None, :].clamp_min(1e-8)
            prior_loss = (mass * (mass.clamp_min(1e-8) / pi).log()).sum(1).mean()
        else:
            prior_loss = (mass * (mass.clamp_min(1e-8) * mass.shape[1]).log()).sum(1).mean()


        # Advance Adam state for the fusion scorer during warm-up.
        loss = rec + .1 * assign + .1 * prior_loss + zero
    else:
        same = teacher @ teacher.T
        values = []
        for v, relation in enumerate(relations):
            mask = (relation + torch.eye(len(same), dtype=torch.int32,
                                         device=same.device)).bool()
            values.append(neighbor_geometry(out['hv'][:, v], out['z'],
                                            mask * same, (~mask) * (1 - same).clamp_min(0)))
        geometry = torch.stack(values).mean()
        loss = rec + .1 * assign + .2 * geometry
    return loss, dict(reconstruction=rec, assignment=assign,
                      prior=prior_loss, geometry=geometry)


@torch.no_grad()
def global_factor(xs):
    'Rank-two factor initialization.'
    xs = [x.detach().cpu() for x in xs]
    v = len(xs)
    scales = [x.double().square().mean().sqrt().clamp_min(1e-8)
              * math.sqrt(x.shape[1] * v) for x in xs]
    y = torch.cat([(x.double() - x.double().median(0).values) / s
                   for x, s in zip(xs, scales)], 1)
    generator = torch.Generator().manual_seed(391009)
    omega = torch.randn(y.shape[1], min(10, *y.shape), generator=generator,
                        dtype=torch.float64)
    q = torch.linalg.qr(y @ omega, mode='reduced').Q
    for _ in range(2):
        q = torch.linalg.qr(y @ (y.T @ q), mode='reduced').Q
    u, singular, vt = torch.linalg.svd(q.T @ y, full_matrices=False)
    scores, loadings = q @ u[:, :2], vt[:2].T * singular[:2]

    def solve(a, c, d, u, v):
        determinant = (a * d - c * c).clamp_min(1e-18)
        return torch.stack([(d * u - c * v) / determinant,
                            (a * v - c * u) / determinant], 1)

    for _ in range(32):
        f0, f1 = scores[:, 0, None], scores[:, 1, None]
        loadings = solve(f0.square().sum(0).expand(y.shape[1]) + 1e-6,
                         (f0 * f1).sum(0).expand(y.shape[1]),
                         f1.square().sum(0).expand(y.shape[1]) + 1e-6,
                         (y * f0).sum(0), (y * f1).sum(0))
        b0, b1 = loadings[None, :, 0], loadings[None, :, 1]
        scores = solve(b0.square().sum(1).expand(len(y)) + 1e-6,
                       (b0 * b1).sum(1).expand(len(y)),
                       b1.square().sum(1).expand(len(y)) + 1e-6,
                       (y * b0).sum(1), (y * b1).sum(1))
        scores, rotation = torch.linalg.qr(scores, mode='reduced')
        loadings = loadings @ rotation.T
    blocks, offset = [], 0
    for x, scale in zip(xs, scales):
        blocks.append((loadings[offset:offset + x.shape[1]] * scale).float())
        offset += x.shape[1]
    return dict(scores=scores.float(), loadings=blocks)


def block_variance(value, dims):
    return torch.cat([part.mean().expand(d) for part, d in zip(value.split(dims), dims)])


def log_density(y, mu, b, psi, pi):
    inv = psi.reciprocal()
    m = torch.eye(2, device=y.device, dtype=y.dtype) + b.T @ (inv[:, None] * b)
    cov = torch.linalg.inv(m)
    beta = (inv[:, None] * b) @ cov
    yp, mp = y * inv, mu * inv
    distance = yp.mul(y).sum(1)[:, None] + mp.mul(mu).sum(1)[None, :] - 2 * yp @ mu.T
    delta = (y @ beta)[:, None, :] - (mu @ beta)[None, :, :]
    reduction = (((y @ (inv[:, None] * b))[:, None, :]
                  - (mu @ (inv[:, None] * b))[None, :, :]) * delta).sum(2)
    logdet = psi.log().sum() + torch.linalg.slogdet(m).logabsdet
    logits = -.5 * (distance - reduction + logdet + y.shape[1] * math.log(2 * math.pi)) + pi.log()[None, :]
    return logits, delta, cov


def conditional_em(y, k, labels, b0, dims, iterations=60):
    n = len(y)
    q = F.one_hot(torch.as_tensor(labels, device=y.device).long(), k).to(y.dtype)
    count = q.sum(0).clamp_min(1e-8)
    mu, b = (q.T @ y) / count[:, None], b0.clone()
    variance = block_variance(y.var(0, unbiased=False), dims).clamp_min(1e-8)
    floor = (variance * 1e-5).clamp_min(1e-8)
    psi = block_variance(variance - b.square().sum(1), dims).clamp_min(floor)
    pi = count / count.sum()
    for _ in range(iterations):
        logits, latent, cov = log_density(y, mu, b, psi, pi)
        q = logits.softmax(1)
        count = q.sum(0)
        qt = (q[:, :, None] * latent).sum(0)
        expectation = (q[:, :, None] * latent).sum(1)
        second = n * cov + torch.einsum('nk,nka,nkb->ab', q, latent, latent)
        suu = torch.cat([torch.cat([torch.diag(count), qt], 1),
                         torch.cat([qt.T, second], 1)], 0)
        syu = torch.cat([y.T @ q, y.T @ expectation], 1)
        w = torch.linalg.solve(suu + 1e-8 * torch.eye(k + 2, device=y.device,
                                                    dtype=y.dtype), syu.T).T
        mu, b = w[:, :k].T, w[:, k:]
        residual = (y.square().sum(0) - 2 * (w * syu).sum(1) + ((w @ suu) * w).sum(1)) / n
        psi = block_variance(residual, dims).clamp_min(floor)
        pi = (count + 1e-8) / (n + k * 1e-8)
    logits, latent, _ = log_density(y, mu, b, psi, pi)
    return dict(b=b, pi=pi, scores=(logits.softmax(1)[:, :, None] * latent).sum(1),
                likelihood=float(logits.logsumexp(1).mean()))


@torch.no_grad()
def correct_inputs(xs, clusters, seed):
    'Estimate shared factor contributions with two EM initializations.'
    pack = global_factor(xs)
    n, v = len(xs[0]), len(xs)
    scales = [x.double().square().mean().sqrt().clamp_min(1e-8)
              * math.sqrt(x.shape[1] * v) for x in xs]
    y = torch.cat([(x.double() - x.double().median(0).values) / s
                   for x, s in zip(xs, scales)], 1)
    b0 = torch.cat([b.to(y.device).double() / s for b, s in zip(pack['loadings'], scales)], 0) / math.sqrt(n)
    f0 = pack['scores'].to(y.device).double() * math.sqrt(n)
    candidates = []
    for start in (y, y - f0 @ b0.T):
        labels = KMeans(clusters, n_init=10, max_iter=100,
                        random_state=831000 + seed).fit_predict(start.cpu().numpy())
        candidates.append(conditional_em(y, clusters, labels, b0, [x.shape[1] for x in xs]))
    index = max(range(2), key=lambda i: candidates[i]['likelihood'])
    best = candidates[index]
    scores = best['scores'].float()
    loadings = [(part * scale).float() for part, scale in zip(best['b'].split([x.shape[1] for x in xs]), scales)]
    corrected = [x - scores @ loading.T for x, loading in zip(xs, loadings)]
    prior = best['pi'].float()
    prior = prior / prior.sum()
    return corrected, prior, dict(scores=scores.cpu(), loadings=[x.cpu() for x in loadings],
                                  prior=prior.cpu(), selected_start=index,
                                  likelihoods=[x['likelihood'] for x in candidates])


def neighbors(xs):
    'Construct symmetric nearest-neighbor relations from received features.'
    relations = []
    for x in xs:
        graph = torch.zeros(len(x), len(x), dtype=torch.int32, device=x.device)
        for i in range(len(x)):
            ids = (x - x[i]).square().sum(1).topk(10, largest=False).indices[1:]
            graph[i, ids] = 1
        relations.append(torch.maximum(graph, graph.T))
    return relations


def rownorm(x):
    return x / x.sum(1, keepdim=True).clamp_min(1e-12)


def align_partition(labels, z, prototypes, previous):
    k = len(prototypes)
    if previous is None:
        one = np.eye(k)[labels]
        centers = one.T @ z / np.maximum(one.sum(0)[:, None], 1)
        cost = ((centers[:, None] - prototypes[None]) ** 2).sum(2)
    else:
        cost = -np.bincount(labels * k + previous, minlength=k * k).reshape(k, k).astype(float)
    a, b = linear_sum_assignment(cost)
    mapping = np.empty(k, dtype=np.int64)
    mapping[a] = b
    return mapping[labels]


@torch.no_grad()
def refresh(net, out, graph, seed, step, previous):
    z = out['z'].cpu().numpy()
    labels = KMeans(net.k, n_init=5, random_state=96000 + seed + step).fit_predict(z)
    aligned = align_partition(labels, z, net.centers.detach().cpu().numpy(), previous)
    one = F.one_hot(torch.as_tensor(aligned, device=out['z'].device), net.k).float()
    teacher = rownorm(.5 * one + .5 * (graph @ one)).detach()
    mass = one.sum(0)
    centers = torch.einsum('nk,nd->kd', one, out['z']) / mass.clamp_min(1)[:, None]
    net.centers[mass > 0] = centers[mass > 0]
    net.centers.requires_grad_(False)
    return teacher, aligned


def take_step(net, optimizer, loss):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite objective')
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
    if not torch.isfinite(grad):
        raise FloatingPointError('Nonfinite gradient')
    optimizer.step()


def initialize(xs, k, seed):
    'Pretrain the autoencoders and initialize the shared prototypes.'
    device, dims = xs[0].device, [x.shape[1] for x in xs]
    torch.manual_seed(seed)
    ae = Autoencoders(dims).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=.0003)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        _, recs = ae(xs)
        take_step(ae, opt, reconstruction(xs, recs))
    torch.manual_seed(100 + seed)
    net = IGSI(dims, k).to(device)
    net.backbone.load_state_dict(ae.state_dict())
    with torch.no_grad():
        low, _ = net.backbone(xs)
        h = torch.stack([F.normalize(head(x), dim=1) for head, x in zip(net.high, low)], 1)
        z = F.normalize(h.mean(1), dim=1)
        centers = KMeans(k, n_init=20, random_state=4100 + seed).fit(z.cpu().numpy()).cluster_centers_
        net.centers.copy_(torch.as_tensor(centers, device=device))
    return net


def fit(xs, k, shared, seed, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    relations = neighbors(xs)
    graph = rownorm(torch.stack([r.bool().float() for r in relations]).mean(0)
                    + torch.eye(len(xs[0]), device=xs[0].device))
    pack = None
    if shared:
        xs, prior, pack = correct_inputs(xs, k, seed)
    else:
        prior = xs[0].new_full((k,), 1 / k)
    torch.save(pack, output / 'frontend.pt')
    net = initialize(xs, k, seed)
    torch.save(net.state_dict(), output / 'initial.pt')
    optimizer = torch.optim.Adam(net.parameters(), lr=.0003)
    teacher, previous, curve = None, None, []
    torch.manual_seed(100 + seed)
    for step in range(400):
        if step == 200:
            with torch.no_grad():
                z = net(xs)['z'].cpu().numpy()
                centers = KMeans(k, n_init=20, random_state=4500 + seed).fit(z).cluster_centers_
                net.centers.copy_(torch.as_tensor(centers, device=xs[0].device))
            optimizer.state.pop(net.centers, None)
        if step >= 200 and step % 20 == 0:
            with torch.no_grad():
                teacher, previous = refresh(net, net(xs), graph, seed, step, previous)
            optimizer.state.pop(net.centers, None)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = objective(xs, net(xs), relations, prior, shared, teacher)
        take_step(net, optimizer, loss)
        curve.append(dict(update=step + 1, loss=float(loss.detach()),
                          **{k: float(v.detach()) for k, v in parts.items()}))
        if (step + 1) % 50 == 0:
            print(json.dumps(curve[-1]), flush=True)
    with torch.no_grad():
        out = net(xs)
    torch.save(dict(model=net.state_dict(), dims=[x.shape[1] for x in xs],
                    clusters=k, condition='shared' if shared else 'clean', seed=seed), output / 'model.pt')
    for key in ['z', 'q', 'weights']:
        np.save(output / (key + '.npy'), out[key].cpu().numpy())
    (output / 'curve.json').write_text(json.dumps(curve, indent=2) + '\n', encoding='utf8')
    return net


def train_main(argv=None):
    parser = argparse.ArgumentParser(prog="igsi_mvc.py train", description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='NPZ with x0,x1,...; float32 received features')
    parser.add_argument('--clusters', type=int, help='Cluster count; defaults to NPZ metadata')
    parser.add_argument('--condition', choices=['clean', 'shared'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('--out must not exist; previous results are never overwritten')
    with np.load(args.data, allow_pickle=False) as data:
        prefix = 'shared_x' if args.condition == 'shared' and 'shared_x0' in data else 'x'
        keys = sorted((s for s in data.files if s.startswith(prefix) and s[len(prefix):].isdigit()),
                      key=lambda s: int(s[len(prefix):]))
        if keys != [prefix + str(i) for i in range(len(keys))] or len(keys) < 2:
            parser.error('Expected at least two consecutively numbered view matrices')
        arrays = [np.asarray(data[key], dtype=np.float32) for key in keys]
        if args.clusters is None:
            if 'n_clusters' not in data:
                parser.error('Provide --clusters or store n_clusters in the NPZ file')
            args.clusters = int(data['n_clusters'].item())
    if any(x.ndim != 2 for x in arrays):
        parser.error('Each view must be a 2D matrix')
    n = len(arrays[0])
    if not 2 <= args.clusters <= n or n < 10:
        parser.error('Require 2 <= clusters <= samples and samples >= 10')
    if any(x.ndim != 2 or len(x) != n or x.shape[1] == 0 or not np.isfinite(x).all() for x in arrays):
        parser.error('Views must be finite aligned 2D arrays with nonempty columns')
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    with threadpool_limits(args.threads):
        fit([torch.from_numpy(x).to(args.device) for x in arrays], args.clusters,
            args.condition == 'shared', args.seed, args.out)
    meta = dict(condition=args.condition, seed=args.seed, clusters=args.clusters,
                input_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                wall_seconds=time.perf_counter() - started, device=args.device,
                torch=torch.__version__, views=[list(x.shape) for x in arrays],
                ae_updates=200, main_updates=400, labels_used=False)
    (args.out / 'run.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf8')


def metrics(labels, prediction):
    labels = np.unique(labels, return_inverse=True)[1]
    prediction = np.unique(prediction, return_inverse=True)[1]
    size = max(labels.max(), prediction.max()) + 1
    counts = np.bincount(labels * size + prediction, minlength=size * size).reshape(size, size)
    a, b = linear_sum_assignment(-counts)
    return dict(acc=float(counts[a, b].sum() / len(labels)),
                nmi=float(normalized_mutual_info_score(labels, prediction, average_method='arithmetic')),
                ari=float(adjusted_rand_score(labels, prediction)))


def evaluate_main(argv=None):
    p = argparse.ArgumentParser(prog="igsi_mvc.py evaluate", description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True, help='NPY labels or a bundled dataset NPZ')
    args = p.parse_args(argv)
    if args.labels.suffix.lower() == '.npz':
        with np.load(args.labels, allow_pickle=False) as data:
            y = np.asarray(data['y'])
    else:
        y = np.load(args.labels, allow_pickle=False)
    z = np.load(args.run / 'z.npy', allow_pickle=False)
    q = np.load(args.run / 'q.npy', allow_pickle=False)
    k = json.loads((args.run / 'run.json').read_text())['clusters']
    if y.ndim != 1 or len(y) != len(z):
        p.error('Labels must be 1D and aligned with the learned representation')
    dest = args.run / 'evaluation'
    dest.mkdir(exist_ok=False)
    rows, predictions = [], []
    with threadpool_limits(2):
        for seed in range(100):
            pred = KMeans(k, n_init=1, max_iter=100, random_state=seed).fit_predict(z)
            rows.append(dict(seed=seed, **metrics(y, pred)))
            predictions.append(pred)
    report = dict(primary_mean={key: float(np.mean([r[key] for r in rows])) for key in ['acc', 'nmi', 'ari']},
                  native=metrics(y, q.argmax(1)), evaluation_repeats=rows,
                  note='Metrics are reported on a 0-1 scale.')
    np.save(dest / 'predictions.npy', np.stack(predictions))
    (dest / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(report['primary_mean']))


def run_main(argv=None):
    parser = argparse.ArgumentParser(prog="igsi_mvc.py run",
                                     description="Train and evaluate a bundled dataset.")
    parser.add_argument('--dataset', choices=['Dermatology', 'MSRC', '100Leaves', 'HW', 'WebKB'],
                        default='Dermatology')
    parser.add_argument('--condition', choices=['clean', 'shared'], default='shared')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    base = Path(__file__).resolve().parent
    data = base / 'data' / (args.dataset + '.npz')
    if not data.is_file():
        parser.error('Dataset file not found: ' + str(data))
    output = args.out or base / 'runs' / (args.dataset + '-' + args.condition + '-s' + str(args.seed))
    train_main(['--data', str(data), '--condition', args.condition, '--seed', str(args.seed),
                '--device', args.device, '--threads', str(args.threads), '--out', str(output)])
    evaluate_main(['--run', str(output), '--labels', str(data)])


def main():
    commands = {'run': run_main, 'train': train_main, 'evaluate': evaluate_main}
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print(__doc__)
        return
    if sys.argv[1] not in commands:
        raise SystemExit("Expected command: run, train or evaluate; use --help for examples.")
    commands[sys.argv[1]](sys.argv[2:])


if __name__ == '__main__':
    main()
