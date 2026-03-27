import torch
from mamba_block import MambaModel

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n사용 디바이스: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

def language_modeling():
    """
    언어 모델링
    시퀀스 생성 데모
    """
    print("=" * 60)
    print("Mamba를 사용한 언어 모델링")
    print("=" * 60)
    
    vocab_size = 1000
    d_model = 256
    n_layers = 4
    
    model = MambaModel(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab_size,
        d_state=16,
        dropout=0.0
    ).to(device)
    
    model.eval()
    
    print(f"\n모델 정보:")
    print(f"  어휘 크기: {vocab_size:,}")
    print(f"  모델 차원: {d_model}")
    print(f"  레이어 수: {n_layers}")
    print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    prompt = torch.randint(0, vocab_size, (1, 10)).to(device)
    print(f"\n프롬프트 길이: {prompt.shape[1]}")
    print(f"프롬프트 토큰: {prompt[0].tolist()}")
    
    max_new_tokens = 20
    generated = prompt.clone()
    
    print(f"\n생성 중... (최대 {max_new_tokens} 토큰)")
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            logits = model(generated)
            
            next_token_logits = logits[:, -1, :]
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if (i + 1) % 5 == 0:
                print(f"  {i + 1} 토큰 생성됨...")
    
    print(f"\n생성된 시퀀스 길이: {generated.shape[1]}")
    print(f"생성된 토큰: {generated[0].tolist()}")
    print(f"\n✓ 완료")


def sequence_classification():
    """
    시퀀스 분류
    """
    print("\n" + "=" * 60)
    print("Mamba를 사용한 시퀀스 분류")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 50
    d_model = 128
    n_classes = 5
    
    backbone = MambaModel(
        d_model=d_model,
        n_layers=3,
        d_state=16,
        dropout=0.1,
        vocab_size=None
    ).to(device)
    
    classifier = torch.nn.Linear(d_model, n_classes).to(device)
    
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    print(f"\n입력 shape: {x.shape}")
    print(f"클래스 수: {n_classes}")
    
    with torch.no_grad():
        features = backbone(x)
        
        pooled = features.mean(dim=1)
        
        logits = classifier(pooled)
        
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)
    
    print(f"\n특징 shape: {features.shape}")
    print(f"풀링된 특징 shape: {pooled.shape}")
    print(f"로짓 shape: {logits.shape}")
    
    print(f"\n예측 결과:")
    for i in range(batch_size):
        pred_class = predictions[i].item()
        confidence = probs[i, pred_class].item()
        print(f"  샘플 {i+1}: 클래스 {pred_class} (신뢰도: {confidence:.2%})")
    
    print(f"\n✓ 분류 완료")


def compare_inference_speed():
    """
    추론 속도 비교 (시퀀스 길이에 따른)
    """
    print("\n" + "=" * 60)
    print("시퀀스 길이에 따른 추론 속도")
    print("=" * 60)
    
    import time
    
    d_model = 256
    n_layers = 4
    
    model = MambaModel(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=None,
        dropout=0.0
    ).to(device)
    
    model.eval()
    
    seq_lengths = [64, 128, 256, 512, 1024]
    batch_size = 1
    
    print(f"\n배치 크기: {batch_size}")
    print(f"모델 차원: {d_model}")
    print(f"레이어 수: {n_layers}")
    
    print(f"\n{'시퀀스 길이':<12} {'추론 시간 (ms)':<18} {'토큰/초':<15}")
    print("-" * 50)
    
    for seq_len in seq_lengths:
        x = torch.randn(batch_size, seq_len, d_model).to(device)
        
        with torch.no_grad():
            _ = model(x)
            
            start_time = time.time()
            for _ in range(10):
                _ = model(x)
            end_time = time.time()
        
        avg_time_ms = (end_time - start_time) / 10 * 1000
        tokens_per_sec = seq_len / (avg_time_ms / 1000)
        
        print(f"{seq_len:<12} {avg_time_ms:>15.2f}  {tokens_per_sec:>12.0f}")
    
    if device.type == 'cuda':
        print(f"\n* GPU에서 측정된 값")
    else:
        print(f"\n* CPU에서 측정된 값")
        print(f"* GPU에서는 훨씬 빠른 속도 기대")


def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 20 + "Mamba 사용법" + " " * 27 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    language_modeling()
    
    sequence_classification()
    
    compare_inference_speed()
    
    print("\n" + "=" * 60)
    print("완료")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
