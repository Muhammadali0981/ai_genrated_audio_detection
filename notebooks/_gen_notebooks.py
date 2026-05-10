"""
Generates notebooks 03, 05, and 06.
Run: python notebooks/_gen_notebooks.py
"""
import json, pathlib

NB_DIR = pathlib.Path(__file__).parent
ROOT_STR = r"C:\Ali\Programming\ai_genrated_audio_detection"

METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"},
}


def nb(cells):
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": METADATA, "cells": cells}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def save(name, notebook):
    p = NB_DIR / name
    p.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Written: {p}")


# ---------------------------------------------------------------------------
# Shared code blocks reused across eval notebooks
# ---------------------------------------------------------------------------

IMPORTS_DATASET = (
    "from pathlib import Path\n"
    "from collections import Counter\n"
    "\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import torch\n"
    "from torch.utils.data import Dataset, DataLoader\n"
    "import torchaudio\n"
    "\n"
    "torchaudio.set_audio_backend(\"soundfile\")\n"
    "\n"
    'ROOT           = Path(r"' + ROOT_STR + '")\n'
    "TARGET_SAMPLES = 64_600\n"
    'DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
    "\n"
    'assert ROOT.exists(), f"Project root not found: {ROOT}"\n'
    "\n"
    "AUDIO_DIRS = {\n"
    '    "train"  : ROOT / "data/LA/ASVspoof2019_LA_train/flac",\n'
    '    "dev"    : ROOT / "data/LA/ASVspoof2019_LA_dev/flac",\n'
    '    "la_eval": ROOT / "data/LA/ASVspoof2019_LA_eval/flac",\n'
    '    "df_eval": ROOT / "data/DF/flac",\n'
    "}\n"
    "\n"
    "PROTOCOL_FILES = {\n"
    '    "train"  : ROOT / "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",\n'
    '    "dev"    : ROOT / "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",\n'
    '    "la_eval": ROOT / "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",\n'
    '    "df_eval": ROOT / "data/DF/keys/DF/CM/trial_metadata.txt",\n'
    "}\n"
    "\n"
    'CHECKPOINT = ROOT / "checkpoints/best_model.pth"\n'
    "\n"
    'print(f"ROOT   = {ROOT}")\n'
    'print(f"DEVICE = {DEVICE}")\n'
)

PARSERS_DATASET = """\
def parse_la_protocol(path):
    \"\"\"
    LA format (space-sep): SPEAKER  FILE_NAME  -  SYSTEM_ID  KEY
    Returns {utt_id: (label:int, attack_id:str)}
    \"\"\"
    label_map = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            uid, atk, key = parts[1], parts[3], parts[4]
            label_map[uid] = (0 if key == "bonafide" else 1, atk)
    return label_map


def parse_df_protocol(path):
    \"\"\"
    DF format (space-sep, NO header):
      col 0 speaker | col 1 file_name | col 2 codec | col 3 source | col 4 system_id | col 5 key
    Returns {utt_id: (label:int, attack_id:str)}
    \"\"\"
    label_map = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            uid = parts[1]
            atk = parts[4]
            key = parts[5]
            label_map[uid] = (0 if key == "bonafide" else 1, atk)
    return label_map


class ASVspoofDataset(Dataset):
    \"\"\"
    Unified dataset for ASVspoof 2019 LA and 2021 DF.

    subset: 'train' | 'dev' | 'la_eval' | 'df_eval'

    __getitem__ returns:
        waveform  : Tensor [1, 64600]
        label     : int  (0=bonafide, 1=spoof)
        utt_id    : str
        attack_id : str
    \"\"\"

    def __init__(self, subset: str):
        assert subset in AUDIO_DIRS, f"subset must be one of {list(AUDIO_DIRS)}"
        self.audio_dir = AUDIO_DIRS[subset]
        protocol_path  = PROTOCOL_FILES[subset]

        if subset == "df_eval":
            if not self.audio_dir.exists() or not protocol_path.exists():
                print(f"DF data not found — skipping.\\n"
                      f"  audio    : {self.audio_dir}\\n"
                      f"  protocol : {protocol_path}")
                self.items = []
                return
            label_map = parse_df_protocol(protocol_path)
        else:
            label_map = parse_la_protocol(protocol_path)

        self.items = [
            (str(self.audio_dir / f"{uid}.flac"), lbl, uid, atk)
            for uid, (lbl, atk) in label_map.items()
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, uid, atk = self.items[idx]
        try:
            wav, _ = torchaudio.load(path)
        except Exception:
            try:
                import av
                container = av.open(path)
                stream = container.streams.audio[0]
                chunks = [frame.to_ndarray() for frame in container.decode(stream)]
                container.close()
                wav = torch.from_numpy(np.concatenate(chunks, axis=1).astype(np.float32))
            except Exception:
                wav = torch.zeros(1, TARGET_SAMPLES)
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        return self._fix_length(wav), label, uid, atk

    @staticmethod
    def _fix_length(wav: torch.Tensor) -> torch.Tensor:
        n = wav.size(1)
        if n < TARGET_SAMPLES:
            wav = wav.repeat(1, TARGET_SAMPLES // n + 1)
        return wav[:, :TARGET_SAMPLES]


print("Dataset classes defined.")
"""

MODEL_CODE = """\
import math
import torch.nn as nn
import torch.nn.functional as F


# ── SincConv1d ──────────────────────────────────────────────────────────────

class SincConv1d(nn.Module):
    MIN_LOW_HZ = 50.0;  MIN_BAND_HZ = 50.0
    @staticmethod
    def _to_mel(hz):  return 2595.0 * math.log10(1.0 + hz / 700.0)
    @staticmethod
    def _to_hz(mel):  return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def __init__(self, n_filters=70, kernel_size=1024, stride=16, padding=512, sample_rate=16000):
        super().__init__()
        if kernel_size % 2 == 0: kernel_size += 1
        self.stride = stride; self.padding = padding; self.sample_rate = sample_rate
        mel_min = self._to_mel(self.MIN_LOW_HZ)
        mel_max = self._to_mel(sample_rate / 2.0 - self.MIN_BAND_HZ)
        mel_pts = np.linspace(mel_min, mel_max, n_filters + 1)
        hz_pts  = np.array([self._to_hz(m) for m in mel_pts])
        self.low_freq  = nn.Parameter(torch.from_numpy(hz_pts[:-1]).float().unsqueeze(1))
        self.band_freq = nn.Parameter(torch.from_numpy(hz_pts[1:] - hz_pts[:-1]).float().unsqueeze(1))
        half = (kernel_size - 1) // 2
        n = torch.arange(-half, half + 1).float()
        self.register_buffer("window", 0.54 - 0.46 * torch.cos(2.0 * math.pi * (n + half) / (kernel_size - 1)))
        self.register_buffer("n", n)

    def _build_filters(self):
        sr   = float(self.sample_rate)
        low  = self.MIN_LOW_HZ  / sr + torch.abs(self.low_freq)  / sr
        high = low + self.MIN_BAND_HZ / sr + torch.abs(self.band_freq) / sr
        high = torch.clamp(high, max=0.5)
        n    = self.n.unsqueeze(0)
        return ((2.0 * high * torch.sinc(2.0 * high * n) -
                 2.0 * low  * torch.sinc(2.0 * low  * n)) * self.window).unsqueeze(1)

    def forward(self, x):
        return F.conv1d(x, self._build_filters(), stride=self.stride, padding=self.padding)


# ── ResBlock ─────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, first=False):
        super().__init__()
        self.first = first
        if not first: self.bn1 = nn.BatchNorm1d(in_ch)
        self.lrelu = nn.LeakyReLU(0.3, inplace=True)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.fms_fc = nn.Linear(out_ch, out_ch)
        self.pool   = nn.MaxPool1d(3)

    def forward(self, x):
        out = x if self.first else self.lrelu(self.bn1(x))
        out = self.lrelu(self.bn2(self.conv1(out)))
        out = self.conv2(out) + self.skip(x)
        scale = torch.sigmoid(self.fms_fc(out.mean(2))).unsqueeze(2)
        return self.pool(out * scale)


# ── RawNet2Encoder ────────────────────────────────────────────────────────────

class RawNet2Encoder(nn.Module):
    GRU_HIDDEN = 1024
    def __init__(self):
        super().__init__()
        self.sinc = SincConv1d()
        self.bn0  = nn.BatchNorm1d(70)
        self.res1 = ResBlock(70,  20, first=True)
        self.res2 = ResBlock(20,  20)
        self.res3 = ResBlock(20,  128)
        self.res4 = ResBlock(128, 128)
        self.res5 = ResBlock(128, 128)
        self.res6 = ResBlock(128, 128)
        self.bn_pre_gru = nn.BatchNorm1d(128)
        self.gru = nn.GRU(128, self.GRU_HIDDEN, batch_first=True)

    def forward(self, x):
        x = self.bn0(torch.abs(self.sinc(x)))
        x = self.res6(self.res5(self.res4(self.res3(self.res2(self.res1(x))))))
        x = self.bn_pre_gru(x).permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)                            # [B, T', 1024]
        B, T, H = x.shape
        return x.view(B, T, 128, H // 128).permute(0, 2, 3, 1)  # [B, 128, 8, T']


# ── Graph Attention Layer ─────────────────────────────────────────────────────

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, temperature=1.0, dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.att_proj       = nn.Linear(in_dim, out_dim)
        self.att_weight     = nn.Parameter(torch.empty(out_dim, 1))
        nn.init.xavier_normal_(self.att_weight)
        self.proj_with_att    = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn  = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.SELU(inplace=True)

    def _att_map(self, x):
        N = x.size(1)
        xi = x.unsqueeze(2).expand(-1,-1,N,-1)
        e  = torch.tanh(self.att_proj(xi * xi.transpose(1,2)))
        return F.softmax(torch.matmul(e, self.att_weight) / self.temperature, dim=2)

    def _bn(self, x):
        s = x.shape; return self.bn(x.view(-1,s[-1])).view(s)

    def forward(self, x):
        x   = self.drop(x)
        att = self._att_map(x)
        out = self.proj_with_att(torch.matmul(att.squeeze(-1), x)) + self.proj_without_att(x)
        return self.act(self._bn(out))


# ── Graph Pooling ─────────────────────────────────────────────────────────────

class GraphPool(nn.Module):
    def __init__(self, ratio, in_dim, dropout=0.3):
        super().__init__()
        self.ratio = ratio
        self.proj  = nn.Linear(in_dim, 1)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.sig   = nn.Sigmoid()

    def forward(self, h):
        scores = self.sig(self.proj(self.drop(h)))
        n_keep = max(int(h.size(1) * self.ratio), 1)
        _, idx = torch.topk(scores, n_keep, dim=1)
        return torch.gather(h * scores, 1, idx.expand(-1,-1,h.size(2)))


# ── Heterogeneous Graph Layer ─────────────────────────────────────────────────

class HeterogeneousGraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim, temperature=2.0, dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.proj_s = nn.Linear(in_dim, in_dim)
        self.proj_t = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_w_ss = nn.Parameter(torch.empty(out_dim, 1)); nn.init.xavier_normal_(self.att_w_ss)
        self.att_w_tt = nn.Parameter(torch.empty(out_dim, 1)); nn.init.xavier_normal_(self.att_w_tt)
        self.att_w_st = nn.Parameter(torch.empty(out_dim, 1)); nn.init.xavier_normal_(self.att_w_st)
        self.att_proj_m = nn.Linear(in_dim, out_dim)
        self.att_w_m    = nn.Parameter(torch.empty(out_dim, 1)); nn.init.xavier_normal_(self.att_w_m)
        self.proj_with_att      = nn.Linear(in_dim, out_dim)
        self.proj_without_att   = nn.Linear(in_dim, out_dim)
        self.proj_m_with_att    = nn.Linear(in_dim, out_dim)
        self.proj_m_without_att = nn.Linear(in_dim, out_dim)
        self.bn   = nn.BatchNorm1d(out_dim)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.SELU(inplace=True)

    def _pairwise(self, x):
        N = x.size(1)
        xi = x.unsqueeze(2).expand(-1,-1,N,-1)
        return xi * xi.transpose(1,2)

    def _build_att(self, x, Ns, Nt):
        e = torch.tanh(self.att_proj(self._pairwise(x)))
        board = torch.zeros(*e.shape[:3], 1, device=x.device)
        board[:,:Ns,:Ns,:] = torch.matmul(e[:,:Ns,:Ns,:], self.att_w_ss)
        board[:,Ns:,Ns:,:] = torch.matmul(e[:,Ns:,Ns:,:], self.att_w_tt)
        board[:,:Ns,Ns:,:] = torch.matmul(e[:,:Ns,Ns:,:], self.att_w_st)
        board[:,Ns:,:Ns,:] = torch.matmul(e[:,Ns:,:Ns,:], self.att_w_st)
        return F.softmax(board / self.temperature, dim=2)

    def _update_master(self, x, master):
        att = F.softmax(torch.matmul(torch.tanh(self.att_proj_m(x * master)), self.att_w_m) / self.temperature, dim=1)
        return self.proj_m_with_att(torch.matmul(att.transpose(1,2), x)) + self.proj_m_without_att(master)

    def _bn(self, x):
        s = x.shape; return self.bn(x.view(-1,s[-1])).view(s)

    def forward(self, x_s, x_t, master=None):
        Ns, Nt = x_s.size(1), x_t.size(1)
        x = torch.cat([self.proj_s(x_s), self.proj_t(x_t)], dim=1)
        if master is None: master = x.mean(1, keepdim=True)
        x   = self.drop(x)
        att = self._build_att(x, Ns, Nt)
        m   = self._update_master(x, master)
        out = self.proj_with_att(torch.matmul(att.squeeze(-1), x)) + self.proj_without_att(x)
        out = self.act(self._bn(out))
        return out[:,:Ns,:], out[:,Ns:,:], m


# ── Graph Module ──────────────────────────────────────────────────────────────

class GraphModule(nn.Module):
    def __init__(self, in_dim=128, gat_dim0=24, gat_dim1=32,
                 pool_ratio_s=0.4, pool_ratio_t=0.5, pool_ratio_hs=0.7, pool_ratio_ht=0.5,
                 temp0=2.0, temp1=100.0):
        super().__init__()
        self.gat_s  = GraphAttentionLayer(in_dim, gat_dim0, temperature=temp0)
        self.gat_t  = GraphAttentionLayer(in_dim, gat_dim0, temperature=temp0)
        self.pool_s = GraphPool(pool_ratio_s, gat_dim0)
        self.pool_t = GraphPool(pool_ratio_t, gat_dim0)
        self.hg1a   = HeterogeneousGraphLayer(gat_dim0, gat_dim1, temperature=temp1)
        self.hg1b   = HeterogeneousGraphLayer(gat_dim1, gat_dim1, temperature=temp1)
        self.pool_hs1 = GraphPool(pool_ratio_hs, gat_dim1)
        self.pool_ht1 = GraphPool(pool_ratio_hs, gat_dim1)
        self.hg2a   = HeterogeneousGraphLayer(gat_dim0, gat_dim1, temperature=temp1)
        self.hg2b   = HeterogeneousGraphLayer(gat_dim1, gat_dim1, temperature=temp1)
        self.pool_hs2 = GraphPool(pool_ratio_hs, gat_dim1)
        self.pool_ht2 = GraphPool(pool_ratio_hs, gat_dim1)
        self.drop_way = nn.Dropout(0.2)
        self.readout_dim = 5 * gat_dim1
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dim0))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dim0))

    def forward(self, feat):
        e_s, _ = feat.abs().max(3);  e_s = e_s.permute(0,2,1)
        e_t, _ = feat.abs().max(2);  e_t = e_t.permute(0,2,1)
        s  = self.pool_s(self.gat_s(e_s))
        t  = self.pool_t(self.gat_t(e_t))
        B  = feat.size(0)
        m1 = self.master1.expand(B,-1,-1);  m2 = self.master2.expand(B,-1,-1)
        s1, t1, m1 = self.hg1a(s, t, master=m1)
        s1 = self.pool_hs1(s1);  t1 = self.pool_ht1(t1)
        s1a, t1a, m1a = self.hg1b(s1, t1, master=m1)
        s1, t1, m1 = s1+s1a, t1+t1a, m1+m1a
        s2, t2, m2 = self.hg2a(s, t, master=m2)
        s2 = self.pool_hs2(s2);  t2 = self.pool_ht2(t2)
        s2a, t2a, m2a = self.hg2b(s2, t2, master=m2)
        s2, t2, m2 = s2+s2a, t2+t2a, m2+m2a
        s_o = self.drop_way(torch.max(s1, s2))
        t_o = self.drop_way(torch.max(t1, t2))
        m_o = self.drop_way(torch.max(m1, m2))
        return torch.cat([t_o.abs().max(1)[0], t_o.mean(1),
                          s_o.abs().max(1)[0], s_o.mean(1),
                          m_o.squeeze(1)], dim=1)


# ── AASIST ────────────────────────────────────────────────────────────────────

class AASIST(nn.Module):
    GAT_DIMS=[24,32]; POOL_RATIOS=[0.4,0.5,0.7,0.5]; TEMPS=[2.0,2.0,100.0,100.0]
    def __init__(self):
        super().__init__()
        gd,pr,tp = self.GAT_DIMS, self.POOL_RATIOS, self.TEMPS
        self.encoder = RawNet2Encoder()
        self.graph   = GraphModule(in_dim=128, gat_dim0=gd[0], gat_dim1=gd[1],
                                   pool_ratio_s=pr[0], pool_ratio_t=pr[1],
                                   pool_ratio_hs=pr[2], pool_ratio_ht=pr[3],
                                   temp0=tp[0], temp1=tp[2])
        self.drop = nn.Dropout(0.5)
        self.out  = nn.Linear(self.graph.readout_dim, 2)

    def forward(self, x):
        return self.out(self.drop(self.graph(self.encoder(x))))


print("AASIST model defined.")
"""

METRICS_CODE = """\
from scipy.stats import norm
from sklearn.metrics import roc_curve


def compute_eer(labels, scores):
    \"\"\"
    labels : array-like of int  (0=bonafide, 1=spoof)
    scores : array-like of float (higher => more likely spoof)
    Returns (eer: float, threshold: float)
    \"\"\"
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def plot_det(labels, scores, title="DET Curve", ax=None, color="steelblue", label=None):
    \"\"\"Plots a Detection Error Tradeoff curve on normal-deviate axes.\"\"\"
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr  = 1.0 - tpr
    eps  = 1e-6
    x_nd = norm.ppf(np.clip(fpr, eps, 1 - eps))
    y_nd = norm.ppf(np.clip(fnr, eps, 1 - eps))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ticks_pct = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]
    tv = norm.ppf([t / 100 for t in ticks_pct])

    # EER diagonal (where FPR == FNR)
    ax.plot([tv[0], tv[-1]], [tv[0], tv[-1]],
            color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.plot(x_nd, y_nd, color=color, linewidth=1.5, label=label)

    ax.set_xlim(tv[0], tv[-1])
    ax.set_ylim(tv[0], tv[-1])
    ax.set_xticks(tv); ax.set_xticklabels([str(t) for t in ticks_pct], rotation=45, ha="right")
    ax.set_yticks(tv); ax.set_yticklabels([str(t) for t in ticks_pct])
    ax.set_xlabel("False Alarm Rate (%)")
    ax.set_ylabel("Miss Rate (%)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    if label:
        ax.legend()
    return ax


def per_attack_eer(labels, scores, attack_ids):
    \"\"\"
    For each spoof attack, compute EER against ALL bonafide samples.
    Bonafide samples are identified by label == 0.
    Returns {attack_id: eer_float} sorted ascending by EER.
    \"\"\"
    labels     = np.asarray(labels)
    scores     = np.asarray(scores)
    attack_ids = np.asarray(attack_ids)

    bona_mask = labels == 0          # all bonafide samples

    results = {}
    for atk in np.unique(attack_ids):
        spoof_mask = (attack_ids == atk) & (labels == 1)
        if spoof_mask.sum() == 0:    # skip groups with no spoof samples
            continue
        mask = bona_mask | spoof_mask
        eer, _ = compute_eer(labels[mask], scores[mask])
        results[atk] = eer

    return dict(sorted(results.items(), key=lambda x: x[1]))


print("Metrics functions defined.")
"""

# ---------------------------------------------------------------------------
# 03_metrics.ipynb
# ---------------------------------------------------------------------------

nb03 = nb([
    md("# 03 — Metrics\n\n"
       "EER (Equal Error Rate), DET curves, and per-attack breakdown utilities.\n"
       "These functions are imported verbatim into notebooks 05 and 06."),

    md("## Imports"),
    code("from pathlib import Path\n\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom scipy.stats import norm\nfrom sklearn.metrics import roc_curve\n\nprint(\"Imports OK\")"),

    md("## EER"),
    code("""\
def compute_eer(labels, scores):
    \"\"\"
    labels : array-like of int  (0=bonafide, 1=spoof)
    scores : array-like of float (higher => more likely spoof)
    Returns (eer: float, threshold: float)
    \"\"\"
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])

print("compute_eer defined.")"""),

    md("## DET Curve"),
    code("""\
def plot_det(labels, scores, title="DET Curve", ax=None, color="steelblue", label=None):
    \"\"\"Plots a Detection Error Tradeoff curve on normal-deviate axes.\"\"\"
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr  = 1.0 - tpr
    eps  = 1e-6
    x_nd = norm.ppf(np.clip(fpr, eps, 1 - eps))
    y_nd = norm.ppf(np.clip(fnr, eps, 1 - eps))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ticks_pct = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]
    tv = norm.ppf([t / 100 for t in ticks_pct])

    # EER diagonal (where FPR == FNR)
    ax.plot([tv[0], tv[-1]], [tv[0], tv[-1]],
            color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.plot(x_nd, y_nd, color=color, linewidth=1.5, label=label)

    ax.set_xlim(tv[0], tv[-1])
    ax.set_ylim(tv[0], tv[-1])
    ax.set_xticks(tv); ax.set_xticklabels([str(t) for t in ticks_pct], rotation=45, ha="right")
    ax.set_yticks(tv); ax.set_yticklabels([str(t) for t in ticks_pct])
    ax.set_xlabel("False Alarm Rate (%)")
    ax.set_ylabel("Miss Rate (%)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    if label:
        ax.legend()
    return ax

print("plot_det defined.")"""),

    md("## Per-Attack EER"),
    code("""\
def per_attack_eer(labels, scores, attack_ids):
    \"\"\"
    For each spoof attack, compute EER against ALL bonafide samples.
    Bonafide samples are identified by label == 0.
    Returns {attack_id: eer_float} sorted ascending by EER.
    \"\"\"
    labels     = np.asarray(labels)
    scores     = np.asarray(scores)
    attack_ids = np.asarray(attack_ids)

    bona_mask = labels == 0          # all bonafide samples

    results = {}
    for atk in np.unique(attack_ids):
        spoof_mask = (attack_ids == atk) & (labels == 1)
        if spoof_mask.sum() == 0:    # skip groups with no spoof samples
            continue
        mask = bona_mask | spoof_mask
        eer, _ = compute_eer(labels[mask], scores[mask])
        results[atk] = eer

    return dict(sorted(results.items(), key=lambda x: x[1]))

print("per_attack_eer defined.")"""),

    md("## Sanity Check"),
    code("""\
# Synthetic data: two Gaussian blobs
np.random.seed(42)
n = 2000
syn_labels = np.array([0] * n + [1] * n)
syn_scores = np.concatenate([
    np.random.normal(0.3, 0.15, n),
    np.random.normal(0.7, 0.15, n),
])
syn_scores = np.clip(syn_scores, 0, 1)

eer, thr = compute_eer(syn_labels, syn_scores)
print(f"Synthetic EER : {eer * 100:.2f}%   threshold = {thr:.4f}")

fig, ax = plt.subplots(figsize=(5, 5))
plot_det(syn_labels, syn_scores, title="Synthetic DET", ax=ax, label=f"EER = {eer*100:.2f}%")
plt.tight_layout()
plt.show()"""),
])

save("03_metrics.ipynb", nb03)

# ---------------------------------------------------------------------------
# 05_evaluate_la.ipynb
# ---------------------------------------------------------------------------

nb05 = nb([
    md("# 05 — Evaluate LA\n\n"
       "Loads `checkpoints/best_model.pth`, runs inference over the LA eval split,\n"
       "and reports EER + DET curve + per-attack breakdown."),

    md("## Imports & Paths"),
    code(IMPORTS_DATASET),

    md("## Dataset"),
    code(PARSERS_DATASET),

    md("## Model"),
    code(MODEL_CODE),

    md("## Metrics"),
    code(METRICS_CODE),

    md("## Load Checkpoint"),
    code("""\
model = AASIST().to(DEVICE)

assert CHECKPOINT.exists(), f"Checkpoint not found: {CHECKPOINT}"
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

print(f"Loaded checkpoint: {CHECKPOINT}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")"""),

    md("## Inference — LA Eval"),
    code("""\
la_eval_ds = ASVspoofDataset("la_eval")
print(f"LA eval samples: {len(la_eval_ds):,}")

loader = DataLoader(la_eval_ds, batch_size=32, shuffle=False, num_workers=0)

all_labels  = []
all_scores  = []
all_attacks = []

with torch.no_grad():
    for wavs, labels, utt_ids, attack_ids in loader:
        wavs   = wavs.to(DEVICE)
        logits = model(wavs)                      # [B, 2]
        scores = torch.sigmoid(logits[:, 1])      # spoof score
        all_labels.extend(labels.numpy().tolist())
        all_scores.extend(scores.cpu().numpy().tolist())
        all_attacks.extend(list(attack_ids))

all_labels  = np.array(all_labels)
all_scores  = np.array(all_scores)
all_attacks = np.array(all_attacks)

print(f"Inference done. Unique attacks: {np.unique(all_attacks).tolist()}")"""),

    md("## EER"),
    code("""\
eer, thr = compute_eer(all_labels, all_scores)
print(f"LA Eval EER : {eer * 100:.2f}%")
print(f"Threshold   : {thr:.4f}")

out_dir = ROOT / "results/la"
out_dir.mkdir(parents=True, exist_ok=True)

np.savez(
    out_dir / "la_eval_scores.npz",
    labels=all_labels, scores=all_scores, attacks=all_attacks,
)
print(f"Scores saved to {out_dir / 'la_eval_scores.npz'}")"""),

    md("## DET Curve"),
    code("""\
fig, ax = plt.subplots(figsize=(6, 6))
plot_det(all_labels, all_scores, title="LA Eval DET", ax=ax,
         color="steelblue", label=f"EER = {eer*100:.2f}%")
plt.tight_layout()
det_path = out_dir / "la_eval_det.png"
plt.savefig(det_path, dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved: {det_path}")"""),

    md("## Per-Attack EER"),
    code("""\
atk_eers = per_attack_eer(all_labels, all_scores, all_attacks)

print(f"\\n{'Attack':>12s}   {'EER (%)':>8s}")
print("-" * 25)
for atk, e in atk_eers.items():
    print(f"{atk:>12s}   {e * 100:>7.2f}%")

# Bar chart
n = len(atk_eers)
fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 5))
bars = ax.bar(range(n), [v * 100 for v in atk_eers.values()],
              color="steelblue", edgecolor="white", linewidth=0.5)
ax.set_xticks(range(n))
ax.set_xticklabels(list(atk_eers.keys()), rotation=45, ha="right")
for bar, val in zip(bars, atk_eers.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val * 100:.1f}%", ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Attack system")
ax.set_ylabel("EER (%)")
ax.set_title("LA Eval — Per-Attack EER")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
bar_path = out_dir / "la_eval_per_attack_eer.png"
plt.savefig(bar_path, dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved: {bar_path}")"""),
])

save("05_evaluate_la.ipynb", nb05)

# ---------------------------------------------------------------------------
# 06_evaluate_df.ipynb
# ---------------------------------------------------------------------------

nb06 = nb([
    md("# 06 — Evaluate DF\n\n"
       "Loads `checkpoints/best_model.pth`, runs inference over the 2021 DF eval split,\n"
       "and reports EER + DET curve + per-attack breakdown.\n\n"
       "Raises `SystemExit(0)` gracefully when DF data are not present."),

    md("## Imports & Paths"),
    code(IMPORTS_DATASET),

    md("## DF Gate"),
    code("""\
DF_AUDIO    = AUDIO_DIRS["df_eval"]
DF_PROTOCOL = PROTOCOL_FILES["df_eval"]

if not DF_AUDIO.exists() or not DF_PROTOCOL.exists():
    print("DF data not found — nothing to evaluate.")
    print(f"  audio    : {DF_AUDIO}")
    print(f"  protocol : {DF_PROTOCOL}")
    raise SystemExit(0)

print("DF data found. Continuing...")"""),

    md("## Dataset"),
    code(PARSERS_DATASET),

    md("## Model"),
    code(MODEL_CODE),

    md("## Metrics"),
    code(METRICS_CODE),

    md("## Load Checkpoint"),
    code("""\
model = AASIST().to(DEVICE)

assert CHECKPOINT.exists(), f"Checkpoint not found: {CHECKPOINT}"
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

print(f"Loaded checkpoint: {CHECKPOINT}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")"""),

    md("## Inference — DF Eval"),
    code("""\
df_eval_ds = ASVspoofDataset("df_eval")
print(f"DF eval samples: {len(df_eval_ds):,}")

loader = DataLoader(df_eval_ds, batch_size=32, shuffle=False, num_workers=0)

all_labels  = []
all_scores  = []
all_attacks = []

with torch.no_grad():
    for wavs, labels, utt_ids, attack_ids in loader:
        wavs   = wavs.to(DEVICE)
        logits = model(wavs)                      # [B, 2]
        scores = torch.sigmoid(logits[:, 1])      # spoof score
        all_labels.extend(labels.numpy().tolist())
        all_scores.extend(scores.cpu().numpy().tolist())
        all_attacks.extend(list(attack_ids))

all_labels  = np.array(all_labels)
all_scores  = np.array(all_scores)
all_attacks = np.array(all_attacks)

print(f"Inference done. Unique attacks: {len(np.unique(all_attacks))}")"""),

    md("## EER"),
    code("""\
eer, thr = compute_eer(all_labels, all_scores)
print(f"DF Eval EER : {eer * 100:.2f}%")
print(f"Threshold   : {thr:.4f}")

out_dir = ROOT / "results/df"
out_dir.mkdir(parents=True, exist_ok=True)

np.savez(
    out_dir / "df_eval_scores.npz",
    labels=all_labels, scores=all_scores, attacks=all_attacks,
)
print(f"Scores saved to {out_dir / 'df_eval_scores.npz'}")"""),

    md("## DET Curve"),
    code("""\
fig, ax = plt.subplots(figsize=(6, 6))
plot_det(all_labels, all_scores, title="DF Eval DET", ax=ax,
         color="tomato", label=f"EER = {eer*100:.2f}%")
plt.tight_layout()
det_path = out_dir / "df_eval_det.png"
plt.savefig(det_path, dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved: {det_path}")"""),

    md("## Per-Attack EER"),
    code("""\
atk_eers = per_attack_eer(all_labels, all_scores, all_attacks)

print(f"\\n{'Attack':>18s}   {'EER (%)':>8s}")
print("-" * 32)
for atk, e in atk_eers.items():
    print(f"{atk:>18s}   {e * 100:>7.2f}%")

# Bar chart
n = len(atk_eers)
fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 5))
bars = ax.bar(range(n), [v * 100 for v in atk_eers.values()],
              color="tomato", edgecolor="white", linewidth=0.5)
ax.set_xticks(range(n))
ax.set_xticklabels(list(atk_eers.keys()), rotation=45, ha="right")
for bar, val in zip(bars, atk_eers.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val * 100:.1f}%", ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Attack system")
ax.set_ylabel("EER (%)")
ax.set_title("DF Eval — Per-Attack EER")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
bar_path = out_dir / "df_eval_per_attack_eer.png"
plt.savefig(bar_path, dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved: {bar_path}")"""),
])

save("06_evaluate_df.ipynb", nb06)
print("\nAll done.")
