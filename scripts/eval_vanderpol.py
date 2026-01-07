"""Van der Pol 파라미터 일반화 테스트.

사용법:
    PYTHONPATH=./src python3 scripts/eval_vanderpol.py
    PYTHONPATH=./src python3 scripts/eval_vanderpol.py --visualize
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
from pathlib import Path
from scipy.integrate import odeint

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from onn.tlm.model import TopologicalLanguageModel, TLMConfig

# matplotlib optional
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def vanderpol(state, t, mu):
    x, y = state
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]


def generate_trajectory(mu, x0, y0, seq_len, dt=0.1):
    t = np.linspace(0, seq_len * dt, seq_len)
    return odeint(vanderpol, [x0, y0], t, args=(mu,))


class Quantizer:
    def __init__(self, vocab_size=500, scale=5.0):
        self.vocab_size = vocab_size
        self.scale = scale
    
    def encode(self, x):
        normalized = (x / self.scale + 1) / 2
        return np.clip(normalized * self.vocab_size, 0, self.vocab_size - 1).astype(np.int64)
    
    def decode(self, tok):
        normalized = tok / self.vocab_size
        return (normalized * 2 - 1) * self.scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='best_vanderpol.pt')
    parser.add_argument('--n-trials', type=int, default=30)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Van der Pol 파라미터 일반화 테스트")
    print("=" * 60)
    
    # 모델 로드
    cp = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    config = cp['config']
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    vocab_size = config['vocab_size']
    seq_len = config['seq_len']
    scale = config.get('scale', 5.0)
    
    print(f"  Vocab: {vocab_size}, SeqLen: {seq_len}, Scale: {scale}")
    
    quant = Quantizer(vocab_size, scale)
    
    tlm_config = TLMConfig(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        relation_dim=config.get('relation_dim', 64),
        context_window=config['context_window'],
        num_heads=config['num_heads'],
    )
    
    model = TopologicalLanguageModel(tlm_config, num_layers=config.get('num_layers', 4)).to(device)
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    
    def test_mu(mu_list, n=30):
        maes = []
        accs = []
        
        for mu in mu_list:
            for _ in range(n):
                x0 = np.random.uniform(-3, 3)
                y0 = np.random.uniform(-3, 3)
                
                time_steps = seq_len // 2 + 1
                traj = generate_trajectory(mu, x0, y0, time_steps, 0.1)
                tokens = quant.encode(traj.flatten())
                
                inp = torch.tensor([tokens[:seq_len]], dtype=torch.long).to(device)
                tgt = tokens[seq_len] if len(tokens) > seq_len else tokens[-1]
                
                with torch.no_grad():
                    logits, _ = model(inp)
                    pred_tok = logits[0, -1, :].argmax().item()
                
                pred_val = quant.decode(pred_tok)
                true_val = quant.decode(tgt)
                
                maes.append(abs(pred_val - true_val))
                accs.append(1.0 if pred_tok == tgt else 0.0)
        
        return np.mean(maes), np.std(maes), np.mean(accs)
    
    print(f"\n{'Test Case':<35} {'MAE':>8} {'Std':>8} {'Acc':>8}")
    print("-" * 62)
    
    results = {}
    
    # In-distribution: μ = 0.5 ~ 5.0 (학습 범위)
    m, s, a = test_mu([1.0, 2.0, 4.0], args.n_trials)
    print(f"{'In-distribution (μ=1,2,4)':<35} {m:>8.4f} {s:>8.4f} {a*100:>7.1f}%")
    results['in_dist'] = (m, a)
    
    # OOD - 약한 비선형
    m, s, a = test_mu([0.1, 0.3], args.n_trials)
    print(f"{'OOD-Low (μ=0.1,0.3)':<35} {m:>8.4f} {s:>8.4f} {a*100:>7.1f}%")
    results['low'] = (m, a)
    
    # OOD - 강한 비선형
    m, s, a = test_mu([6.0, 8.0], args.n_trials)
    print(f"{'OOD-High (μ=6,8)':<35} {m:>8.4f} {s:>8.4f} {a*100:>7.1f}%")
    results['high'] = (m, a)
    
    # Extreme
    m, s, a = test_mu([0.05, 10.0], args.n_trials)
    print(f"{'Extreme (μ=0.05,10)':<35} {m:>8.4f} {s:>8.4f} {a*100:>7.1f}%")
    results['extreme'] = (m, a)
    
    # 요약
    print("\n" + "=" * 60)
    mae_in = results['in_dist'][0]
    mae_ood = (results['low'][0] + results['high'][0]) / 2
    ratio = mae_ood / mae_in if mae_in > 0 else float('inf')
    
    print(f"In-dist MAE: {mae_in:.4f}, Acc: {results['in_dist'][1]*100:.1f}%")
    print(f"OOD avg MAE: {mae_ood:.4f}")
    print(f"Ratio: {ratio:.2f}x")
    
    if ratio < 1.5:
        print("\n✅ 일반화 우수")
    elif ratio < 2.5:
        print("\n⚠️ 일반화 보통")
    else:
        print("\n❌ 일반화 부족")
    
    # 시각화
    if args.visualize and HAS_MATPLOTLIB:
        visualize(model, quant, seq_len, device)


def visualize(model, quant, seq_len, device):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    mus = [0.5, 2.0, 6.0]
    titles = ['Weak (μ=0.5)', 'Medium (μ=2)', 'Strong (μ=6)']
    
    for idx, (mu, title) in enumerate(zip(mus, titles)):
        ax_phase = axes[0, idx]
        ax_time = axes[1, idx]
        
        # 궤적
        traj = generate_trajectory(mu, 0.5, 0.5, 100, 0.1)
        tokens = quant.encode(traj[:seq_len//2+1].flatten())
        
        inp = torch.tensor([tokens[:seq_len]], dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits, _ = model(inp)
            pred_tokens = logits[0].argmax(dim=-1).cpu().numpy()
        
        pred_vals = quant.decode(pred_tokens).reshape(-1, 2)
        actual = traj[1:len(pred_vals)+1]
        
        # Phase
        ax_phase.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.3, label='Full')
        ax_phase.plot(actual[:, 0], actual[:, 1], 'g-', lw=2, label='Actual')
        ax_phase.plot(pred_vals[:, 0], pred_vals[:, 1], 'r--', lw=2, label='Pred')
        ax_phase.set_title(f'Phase: {title}')
        ax_phase.legend(fontsize=8)
        ax_phase.grid(True, alpha=0.3)
        
        # Time
        t = np.arange(len(actual)) * 0.1
        ax_time.plot(t, actual[:, 0], 'g-', lw=2, label='Actual')
        ax_time.plot(t, pred_vals[:, 0], 'r--', lw=2, label='Pred')
        ax_time.set_title(f'x(t): {title}')
        ax_time.legend(fontsize=8)
        ax_time.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vanderpol_generalization.png', dpi=150)
    print("\n  Saved: vanderpol_generalization.png")


if __name__ == "__main__":
    main()
