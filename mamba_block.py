import torch
import torch.nn as nn
import torch.nn.functional as F
from selective_ssm import SelectiveSSM

class MambaBlock(nn.Module):
    """
    Mamba Block - Replaces Transformer's Attention + FFN with unified O(N) block
    
    Key differences from Transformer:
    - Selective SSM replaces Multi-Head Attention (O(N²) → O(N))
    - Single unified block instead of separate Attention + FFN sub-blocks
    - Added 1D convolution for local context
    
    See README.md for detailed comparison with Transformer architecture.
    """
    
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand_factor=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dropout=0.0
    ):
        """
        Args:
            d_model: 모델 차원
            d_state: SSM 상태 차원
            d_conv: 1D convolution 커널 크기
            expand_factor: 내부 차원 확장 비율
            dt_rank: delta projection rank
            dt_min, dt_max: delta 범위
            dropout: dropout 비율
        """
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand_factor * d_model)
        self.d_conv = d_conv
        
        # ============================================================
        # 1. Input projection (확장)
        # [Transformer FFN의 첫 번째 Linear 역할]
        # Transformer FFN: d_model -> d_ff (보통 4x)
        # Mamba: d_model -> d_inner * 2 (보통 2x, gating용으로 2배)
        # ============================================================
        # x -> [z, x_proj]
        # z: gating용, x_proj: SSM 입력용
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # ============================================================
        # 2. 1D Convolution (causal, 과거 정보만 사용)
        # [Mamba의 새로운 요소 - Transformer에는 없음]
        # 지역적 문맥 정보를 효율적으로 캡처
        # ============================================================
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # causal padding
            groups=self.d_inner,  # depthwise convolution
            bias=True
        )
        
        # ============================================================
        # 3. Activation
        # [Transformer FFN의 중간 활성화 함수와 유사]
        # Transformer: 보통 ReLU or GELU
        # Mamba: SiLU (Swish)
        # ============================================================
        self.activation = nn.SiLU()
        
        # ============================================================
        # 4. Selective SSM - 핵심
        # [Transformer의 Multi-Head Self-Attention을 대체]
        # Attention: Q, K, V로 모든 토큰 간 상호작용 (O(N²))
        # Selective SSM: 입력 의존적 상태 공간으로 시퀀스 처리 (O(N))
        # ============================================================
        self.ssm = SelectiveSSM(
            d_model=self.d_inner,
            d_state=d_state,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max
        )
        
        # ============================================================
        # 5. Output projection (축소)
        # [Transformer FFN의 두 번째 Linear 역할]
        # d_inner -> d_model로 원래 차원으로 복원
        # ============================================================
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # 6. Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # 7. Layer Norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Forward pass - Transformer 블록과 비교
        
        Transformer Block Forward:
          x -> LayerNorm -> Attention -> Add -> LayerNorm -> FFN -> Add
          
        Mamba Block Forward:
          x -> LayerNorm -> [Projection -> Conv -> SSM -> Gate -> Projection] -> Add
                            └─────────── 단일 통합 블록 ──────────────┘
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Residual connection을 위해 저장
        residual = x
        
        # Pre-normalization (Transformer와 동일)
        x = self.norm(x)
        
        # ============================================================
        # 1. Input projection: x -> [x_proj, z]
        # [Transformer FFN: x -> intermediate_dim]
        # ============================================================
        x_and_z = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_proj, z = x_and_z.chunk(2, dim=-1)  # 각각 (batch, seq_len, d_inner)
        
        # ============================================================
        # 2. 1D Convolution (causal) - 지역 정보 캡처
        # [Transformer에는 없는 새로운 요소]
        # ============================================================
        x_conv = x_proj.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv[:, :, :seq_len]  # causal padding 제거
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        # 3. Activation
        x_conv = self.activation(x_conv)
        
        # ============================================================
        # 4. Selective SSM - 핵심 연산
        # [Transformer Attention: Q@K^T@V 대체]
        # Attention처럼 시퀀스 전체를 고려하지만 O(N) 복잡도
        # ============================================================
        y = self.ssm(x_conv)  # (batch, seq_len, d_inner)
        
        # ============================================================
        # 5. Gating (z로 정보 흐름 제어)
        # [GLU (Gated Linear Unit) 스타일, FFN의 비선형성 강화]
        # ============================================================
        y = y * self.activation(z)
        
        # ============================================================
        # 6. Output projection
        # [Transformer FFN: intermediate_dim -> d_model]
        # ============================================================
        output = self.out_proj(y)  # (batch, seq_len, d_model)
        
        # 7. Dropout & Residual (Transformer와 동일)
        output = self.dropout(output)
        output = output + residual
        
        return output


class MambaModel(nn.Module):
    """
    여러 Mamba 블록을 쌓은 완전한 모델
    Transformer와 유사한 구조
    """
    
    def __init__(
        self,
        d_model=256,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand_factor=2,
        dropout=0.1,
        vocab_size=None
    ):
        """
        Args:
            d_model: 모델 차원
            n_layers: Mamba 블록 개수
            d_state: SSM 상태 차원
            d_conv: convolution 커널 크기
            expand_factor: 확장 비율
            dropout: dropout 비율
            vocab_size: 어휘 크기 (언어 모델링용, None이면 임베딩 레이어 없음)
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Token embedding (언어 모델링용)
        self.embedding = nn.Embedding(vocab_size, d_model) if vocab_size else None
        
        # Mamba 블록들
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Final norm
        self.norm_f = nn.LayerNorm(d_model)
        
        # Output head (언어 모델링용)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False) if vocab_size else None
        
        # Tie weights (임베딩과 출력 레이어 가중치 공유)
        if self.embedding and self.lm_head:
            self.lm_head.weight = self.embedding.weight
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) if vocab_size else (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, vocab_size) if vocab_size else (batch, seq_len, d_model)
        """
        # Token embedding
        if self.embedding is not None:
            x = self.embedding(x)  # (batch, seq_len, d_model)
        
        # Mamba 블록들 통과
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.norm_f(x)
        
        # Output projection (언어 모델링)
        if self.lm_head is not None:
            x = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        return x


if __name__ == "__main__":
    print("=== Mamba Block 테스트 ===\n")
    
    # 1. 단일 Mamba 블록 테스트
    batch_size = 2
    seq_len = 20
    d_model = 128
    
    block = MambaBlock(d_model=d_model, d_state=16, expand_factor=2)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"입력 shape: {x.shape}")
    y = block(x)
    print(f"출력 shape: {y.shape}")
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in block.parameters())
    print(f"Mamba Block 파라미터 수: {total_params:,}")
    
    print("\n=== 완전한 Mamba 모델 테스트 ===\n")
    
    # 2. 완전한 Mamba 모델 (언어 모델링)
    vocab_size = 1000
    n_layers = 4
    
    model = MambaModel(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab_size,
        dropout=0.1
    )
    
    # 토큰 시퀀스 입력
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"입력 토큰 shape: {tokens.shape}")
    
    logits = model(tokens)
    print(f"출력 logits shape: {logits.shape}")
    
    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n전체 모델 파라미터 수: {total_params:,}")
    
    # 레이어별 파라미터 수
    print(f"\n레이어별 파라미터:")
    if model.embedding:
        print(f"  Embedding: {sum(p.numel() for p in model.embedding.parameters()):,}")
    print(f"  Mamba Blocks: {sum(p.numel() for layer in model.layers for p in layer.parameters()):,}")
    if model.lm_head:
        print(f"  LM Head: {sum(p.numel() for p in model.lm_head.parameters()):,}")
