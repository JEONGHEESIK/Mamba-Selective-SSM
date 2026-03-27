import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from ssm_basic import BasicSSM
from selective_ssm import SelectiveSSM
from mamba_block import MambaBlock, MambaModel

# GPU 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n사용 디바이스: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

def test_basic_ssm():
    """기본 SSM 테스트"""
    print("=" * 60)
    print("1. Basic SSM 테스트")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 50
    d_model = 64
    d_state = 16
    
    model = BasicSSM(d_model, d_state).to(device)
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    print(f"\n입력 shape: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
    
    print(f"출력 shape: {y.shape}")
    print(f"\nSSM 파라미터:")
    print(f"  A (상태 전이): {model.A.shape}")
    print(f"  B (입력): {model.B.shape}")
    print(f"  C (출력): {model.C.shape}")
    print(f"  delta (시간 스케일): {model.delta.item():.4f}")
    
    return model, x, y


def test_selective_ssm():
    """Selective SSM 테스트 - 입력에 따른 동적 파라미터 변화 확인"""
    print("\n" + "=" * 60)
    print("2. Selective SSM 테스트")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 50
    d_model = 64
    d_state = 16
    
    model = SelectiveSSM(d_model, d_state).to(device)
    
    # 두 가지 다른 입력 생성
    x1 = torch.randn(batch_size, seq_len, d_model).to(device) * 0.5  # 작은 변화
    x2 = torch.randn(batch_size, seq_len, d_model).to(device) * 2.0  # 큰 변화
    
    print(f"\n입력 1 (작은 변화) 통계:")
    print(f"  평균: {x1.mean().item():.4f}, 표준편차: {x1.std().item():.4f}")
    print(f"\n입력 2 (큰 변화) 통계:")
    print(f"  평균: {x2.mean().item():.4f}, 표준편차: {x2.std().item():.4f}")
    
    with torch.no_grad():
        # 입력 1에 대한 동적 파라미터
        delta1 = torch.nn.functional.softplus(model.dt_proj(model.x_proj(x1)))
        B1 = model.B_proj(x1)
        
        # 입력 2에 대한 동적 파라미터
        delta2 = torch.nn.functional.softplus(model.dt_proj(model.x_proj(x2)))
        B2 = model.B_proj(x2)
        
        y1 = model(x1)
        y2 = model(x2)
    
    print(f"\n동적 파라미터 비교:")
    print(f"\n  delta (시간 스케일):")
    print(f"    입력 1 - 평균: {delta1.mean().item():.4f}, 범위: [{delta1.min().item():.4f}, {delta1.max().item():.4f}]")
    print(f"    입력 2 - 평균: {delta2.mean().item():.4f}, 범위: [{delta2.min().item():.4f}, {delta2.max().item():.4f}]")
    
    print(f"\n  B (입력 가중치):")
    print(f"    입력 1 - 평균: {B1.mean().item():.4f}, 표준편차: {B1.std().item():.4f}")
    print(f"    입력 2 - 평균: {B2.mean().item():.4f}, 표준편차: {B2.std().item():.4f}")
    
    print(f"\n출력 shape: {y1.shape}, {y2.shape}")
    
    return model, (x1, x2), (y1, y2), (delta1, delta2)


def test_mamba_block():
    """Mamba Block 테스트"""
    print("\n" + "=" * 60)
    print("3. Mamba Block 테스트")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 50
    d_model = 128
    
    block = MambaBlock(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand_factor=2,
        dropout=0.1
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    print(f"\n입력 shape: {x.shape}")
    
    with torch.no_grad():
        y = block(x)
    
    print(f"출력 shape: {y.shape}")
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in block.parameters())
    trainable_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    
    print(f"\n파라미터 정보:")
    print(f"  전체 파라미터: {total_params:,}")
    print(f"  학습 가능 파라미터: {trainable_params:,}")
    print(f"  내부 차원 (d_inner): {block.d_inner}")
    print(f"  확장 비율: {block.d_inner / d_model:.1f}x")
    
    return block, x, y


def test_mamba_model():
    """완전한 Mamba 모델 테스트"""
    print("\n" + "=" * 60)
    print("4. 완전한 Mamba 모델 테스트 (언어 모델링)")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 100
    vocab_size = 5000
    d_model = 256
    n_layers = 6
    
    model = MambaModel(
        d_model=d_model,
        n_layers=n_layers,
        d_state=16,
        d_conv=4,
        expand_factor=2,
        dropout=0.1,
        vocab_size=vocab_size
    ).to(device)
    
    # 랜덤 토큰 시퀀스
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    print(f"\n입력 토큰 shape: {tokens.shape}")
    print(f"어휘 크기: {vocab_size:,}")
    print(f"레이어 수: {n_layers}")
    print(f"모델 차원: {d_model}")
    
    with torch.no_grad():
        logits = model(tokens)
    
    print(f"\n출력 logits shape: {logits.shape}")
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n전체 파라미터 수: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    # 레이어별 파라미터
    print(f"\n레이어별 파라미터:")
    if model.embedding:
        emb_params = sum(p.numel() for p in model.embedding.parameters())
        print(f"  Embedding: {emb_params:,} ({emb_params / total_params * 100:.1f}%)")
    
    mamba_params = sum(p.numel() for layer in model.layers for p in layer.parameters())
    print(f"  Mamba Blocks: {mamba_params:,} ({mamba_params / total_params * 100:.1f}%)")
    
    # 간단한 다음 토큰 예측
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    next_token = torch.argmax(probs, dim=-1)
    
    print(f"\n다음 토큰 예측 예시 (첫 번째 시퀀스):")
    print(f"  마지막 토큰: {tokens[0, -1].item()}")
    print(f"  예측된 다음 토큰: {next_token[0].item()}")
    print(f"  예측 확률: {probs[0, next_token[0]].item():.4f}")
    
    return model, tokens, logits


def visualize_selective_behavior():
    """Selective SSM의 선택적 동작 시각화"""
    print("\n" + "=" * 60)
    print("5. Selective SSM 동작 시각화")
    print("=" * 60)
    
    batch_size = 1
    seq_len = 100
    d_model = 64
    d_state = 16
    
    model = SelectiveSSM(d_model, d_state).to(device)
    
    # 특별한 패턴을 가진 입력 생성
    # 앞부분: 작은 값, 중간: 큰 값 (중요한 정보), 뒷부분: 작은 값
    x = torch.randn(batch_size, seq_len, d_model).to(device) * 0.5
    x[:, 40:60, :] *= 4.0  # 중간 부분을 강조
    
    with torch.no_grad():
        delta = torch.nn.functional.softplus(model.dt_proj(model.x_proj(x)))
        y = model(x)
    
    # 시각화
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. 입력 신호 (첫 번째 특징)
    axes[0].plot(x[0, :, 0].cpu().numpy(), label='Input Signal', color='blue', alpha=0.7)
    axes[0].axvspan(40, 60, alpha=0.2, color='red', label='Important Region')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Input Signal (First Feature)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Delta (시간 스케일) - 입력에 따라 동적으로 변화
    delta_mean = delta[0].mean(dim=-1).cpu().numpy()
    axes[1].plot(delta_mean, label='Delta (Time Scale)', color='green', linewidth=2)
    axes[1].axvspan(40, 60, alpha=0.2, color='red')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Delta Value')
    axes[1].set_title('Delta (Time Scale) - Adapts to Important Information')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. 출력 신호 (첫 번째 특징)
    axes[2].plot(y[0, :, 0].cpu().numpy(), label='Output Signal', color='orange', linewidth=2)
    axes[2].axvspan(40, 60, alpha=0.2, color='red')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Value')
    axes[2].set_title('Output Signal (First Feature)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/jeonghs/workspace/mamba/selective_ssm_visualization.png', dpi=150, bbox_inches='tight')
    print("\n시각화 저장: selective_ssm_visualization.png")
    
    # 통계 출력
    print(f"\nDelta 통계:")
    print(f"  전체 평균: {delta_mean.mean():.4f}")
    print(f"  중요 구간 (40-60) 평균: {delta_mean[40:60].mean():.4f}")
    print(f"  일반 구간 평균: {np.concatenate([delta_mean[:40], delta_mean[60:]]).mean():.4f}")
    
    return fig


def compare_with_transformer():
    """Transformer와 복잡도 비교"""
    print("\n" + "=" * 60)
    print("6. Transformer vs Mamba 복잡도 비교")
    print("=" * 60)
    
    d_model = 512
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    
    print(f"\n모델 차원: {d_model}")
    print(f"\n{'Seq Length':<12} {'Transformer':<20} {'Mamba':<20} {'비율':<10}")
    print("-" * 65)
    
    for seq_len in seq_lengths:
        # Transformer: O(N^2 * D)
        transformer_ops = seq_len ** 2 * d_model
        
        # Mamba: O(N * D^2) (대략적)
        mamba_ops = seq_len * d_model ** 2
        
        ratio = transformer_ops / mamba_ops
        
        print(f"{seq_len:<12} {transformer_ops:>18,}  {mamba_ops:>18,}  {ratio:>8.2f}x")
    
    print("\n* Transformer가 Mamba보다 느린 배수")
    print("* 시퀀스가 길어질수록 Mamba의 효율성이 증가")


def main():
    """전체 테스트 실행"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "Mamba 구현 테스트" + " " * 25 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # 1. Basic SSM
    test_basic_ssm()
    
    # 2. Selective SSM
    test_selective_ssm()
    
    # 3. Mamba Block
    test_mamba_block()
    
    # 4. 완전한 Mamba 모델
    test_mamba_model()
    
    # 5. 시각화
    visualize_selective_behavior()
    
    # 6. 복잡도 비교
    compare_with_transformer()
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
