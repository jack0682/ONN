"""Oscillator 파라미터 일반화 테스트 (스케일 정렬 버전).

사용법:
    PYTHONPATH=./src python3 scripts/eval_oscillator.py
    PYTHONPATH=./src python3 scripts/eval_oscillator.py --ckpt best_oscillator.pt --visualize
"""

import torch
import numpy as np
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from onn.tlm.model import TopologicalLanguageModel, TLMConfig

# matplotlib 선택적 import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def oscillator(t, A, omega0, zeta, phi):
    """감쇠 진동자 해석해."""
    if zeta < 1:
        wd = omega0 * np.sqrt(1 - zeta**2)
        return A * np.exp(-zeta * omega0 * t) * np.cos(wd * t + phi)
    elif zeta == 1:
        return A * np.exp(-omega0 * t) * (1 + omega0 * t)
    else:
        g1 = omega0 * (zeta + np.sqrt(zeta**2 - 1))
        g2 = omega0 * (zeta - np.sqrt(zeta**2 - 1))
        return A * 0.5 * (np.exp(-g1 * t) + np.exp(-g2 * t))


class Quantizer:
    """학습과 동일한 양자화 스케일 사용."""
    
    def __init__(self, vocab_size=500, scale=10.0):
        self.vocab_size = vocab_size
        self.scale = scale
    
    def encode(self, x):
        """연속값 → 토큰."""
        x_clipped = np.clip(x, -self.scale, self.scale)
        return int((x_clipped + self.scale) / (2 * self.scale) * (self.vocab_size - 1))
    
    def decode(self, tok):
        """토큰 → 연속값."""
        return (tok / (self.vocab_size - 1)) * (2 * self.scale) - self.scale


def main():
    parser = argparse.ArgumentParser(description='Oscillator 일반화 테스트')
    parser.add_argument('--ckpt', type=str, default='best_oscillator.pt', help='체크포인트 경로')
    parser.add_argument('--scale', type=float, default=10.0, help='양자화 스케일')
    parser.add_argument('--seq-len', type=int, default=8, help='입력 시퀀스 길이')
    parser.add_argument('--n-trials', type=int, default=30, help='시행 횟수')
    parser.add_argument('--visualize', action='store_true', help='궤적 시각화')
    parser.add_argument('--full-seq', action='store_true', help='전체 시퀀스 예측 모드')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Oscillator 파라미터 일반화 테스트")
    print("=" * 60)
    
    # 모델 로드
    cp = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    config = cp['config']
    
    # config에서 스케일 읽기 (없으면 기본값)
    scale = config.get('scale', args.scale)
    vocab_size = config.get('vocab_size', 500)
    seq_len = config.get('seq_len', args.seq_len)
    
    print(f"  Scale: {scale}, Vocab: {vocab_size}, SeqLen: {seq_len}")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # 양자화기
    quant = Quantizer(vocab_size=vocab_size, scale=scale)
    
    # TLM 생성
    tlm_config = TLMConfig(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        relation_dim=config.get('relation_dim', 32),
        context_window=config['context_window'],
        num_heads=config['num_heads'],
    )
    
    model = TopologicalLanguageModel(tlm_config, num_layers=config.get('num_layers', 2)).to(device)
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    
    def test_zeta(zeta_list, n=30, full_seq=False):
        """특정 감쇠비에 대한 테스트."""
        maes = []
        accs = []
        
        for zeta in zeta_list:
            for _ in range(n):
                A = np.random.uniform(0.5, 1.5)
                phi = np.random.uniform(0, 2*np.pi)
                omega0 = 2 * np.pi
                
                t = np.linspace(0, 2, seq_len * 2 + 1)
                traj = oscillator(t, A, omega0, zeta, phi)
                tokens = [quant.encode(x) for x in traj]
                
                if full_seq:
                    # 전체 시퀀스 예측: 하나씩 생성
                    inp = torch.tensor([tokens[:seq_len]], dtype=torch.long).to(device)
                    preds = []
                    
                    with torch.no_grad():
                        for step in range(seq_len):
                            logits, _ = model(inp)
                            pred_tok = logits[0, -1, :].argmax().item()
                            preds.append(pred_tok)
                            
                            # 다음 입력에 추가
                            inp = torch.cat([inp, torch.tensor([[pred_tok]], device=device)], dim=1)
                            if inp.size(1) > config['context_window']:
                                inp = inp[:, -config['context_window']:]
                    
                    # MAE 계산
                    pred_vals = np.array([quant.decode(t) for t in preds])
                    true_vals = np.array([quant.decode(tokens[seq_len + i]) for i in range(seq_len)])
                    mae = np.abs(pred_vals - true_vals).mean()
                    
                    # 정확도
                    correct = sum(1 for p, t in zip(preds, tokens[seq_len:seq_len*2]) if p == t)
                    acc = correct / len(preds)
                    
                else:
                    # 단일 스텝 예측
                    inp = torch.tensor([tokens[:seq_len]], dtype=torch.long).to(device)
                    
                    with torch.no_grad():
                        logits, _ = model(inp)
                        pred_tok = logits[0, -1, :].argmax().item()
                    
                    pred_val = quant.decode(pred_tok)
                    true_val = quant.decode(tokens[seq_len])
                    mae = abs(pred_val - true_val)
                    acc = 1.0 if pred_tok == tokens[seq_len] else 0.0
                
                maes.append(mae)
                accs.append(acc)
        
        return np.mean(maes), np.std(maes), np.mean(accs)
    
    print(f"\n{'Test Case':<35} {'MAE':>8} {'Std':>8} {'Acc':>8}")
    print("-" * 62)
    
    results = {}
    
    m, s, a = test_zeta([0.3, 1.0, 1.5], args.n_trials, args.full_seq)
    print(f"{'In-distribution (ζ=0.3,1,1.5)':<35} {m:>8.4f} {s:>8.4f} {a*100:>7.1f}%")
    results['in_dist'] = (m, a)
    
    m, s, a = test_zeta([0.1, 0.15], args.n_trials, args.full_seq)
    print(f"{'OOD-Low (ζ=0.1,0.15)':<35} {m:>8.4f} {s:>8.4f} {a*100:>7.1f}%")
    results['low'] = (m, a)
    
    m, s, a = test_zeta([2.0, 3.0], args.n_trials, args.full_seq)
    print(f"{'OOD-High (ζ=2,3)':<35} {m:>8.4f} {s:>8.4f} {a*100:>7.1f}%")
    results['high'] = (m, a)
    
    m, s, a = test_zeta([0.05, 5.0], args.n_trials, args.full_seq)
    print(f"{'Extreme (ζ=0.05,5)':<35} {m:>8.4f} {s:>8.4f} {a*100:>7.1f}%")
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
        visualize(model, quant, seq_len, config, device)
    elif args.visualize:
        print("\n(matplotlib not available, skipping visualization)")


def visualize(model, quant, seq_len, config, device):
    """실제 vs 예측 궤적 시각화."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    zetas = [0.3, 1.0, 2.0, 0.1]
    titles = ['Underdamped (ζ=0.3)', 'Critical (ζ=1.0)', 'Overdamped (ζ=2.0)', 'OOD (ζ=0.1)']
    
    for ax, zeta, title in zip(axes.flat, zetas, titles):
        # 궤적 생성
        A = 1.0
        phi = 0.0
        omega0 = 2 * np.pi
        t = np.linspace(0, 2, seq_len * 2 + 1)
        traj = oscillator(t, A, omega0, zeta, phi)
        tokens = [quant.encode(x) for x in traj]
        
        # 예측
        inp = torch.tensor([tokens[:seq_len]], dtype=torch.long).to(device)
        preds = []
        
        with torch.no_grad():
            for step in range(seq_len):
                logits, _ = model(inp)
                pred_tok = logits[0, -1, :].argmax().item()
                preds.append(quant.decode(pred_tok))
                inp = torch.cat([inp, torch.tensor([[pred_tok]], device=device)], dim=1)
        
        # 플롯
        ax.plot(t[:seq_len], traj[:seq_len], 'b-', linewidth=2, label='Input')
        ax.plot(t[seq_len:seq_len*2], traj[seq_len:seq_len*2], 'g-', linewidth=2, label='True')
        ax.plot(t[seq_len:seq_len*2], preds, 'r--', linewidth=2, label='Predicted')
        ax.axvline(x=t[seq_len-1], color='gray', linestyle=':', alpha=0.5)
        ax.set_title(title)
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('oscillator_generalization.png', dpi=150)
    print("\n  Saved: oscillator_generalization.png")
    plt.show()


if __name__ == "__main__":
    main()
