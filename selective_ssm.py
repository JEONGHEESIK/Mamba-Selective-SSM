import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (Mamba의 핵심)
    
    기본 SSM과의 차이점:
    1. B, C, Δ(delta)가 고정된 파라미터가 아니라 입력에 따라 동적으로 생성됨
    2. 이를 통해 중요한 정보는 기억하고 불필요한 정보는 필터링
    3. Attention처럼 context-aware하지만 선형 복잡도 유지
    
    핵심 아이디어:
    - Δ가 크면: 현재 입력을 많이 반영 (빠른 변화)
    - Δ가 작으면: 이전 상태를 많이 유지 (느린 변화)
    - B, C를 조정하여 어떤 정보를 얼마나 기억할지 선택
    """
    
    def __init__(self, d_model, d_state=16, dt_rank="auto", dt_min=0.001, dt_max=0.1):
        """
        Args:
            d_model: 입력/출력 차원
            d_state: 숨겨진 상태 차원 (N)
            dt_rank: delta 생성을 위한 rank (보통 d_model // 16)
            dt_min, dt_max: delta의 범위
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank
        
        # A: 고정된 대각 행렬 (학습되지 않음, 초기화만)
        # 일반적으로 음수 값으로 초기화 (안정성)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))  # log space에서 학습
        
        # D: skip connection 파라미터
        self.D = nn.Parameter(torch.ones(d_model))
        
        # 입력 의존적 파라미터를 생성하는 projection 레이어들
        # x -> [B, C, delta]
        
        # delta (Δ) 생성: x -> delta
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        
        # delta를 위한 입력 투영
        self.x_proj = nn.Linear(d_model, self.dt_rank, bias=False)
        
        # B 생성: x -> B
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        
        # C 생성: x -> C  
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        
        # delta 범위 제한을 위한 파라미터
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        # 초기화
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # delta bias를 log space에서 초기화
        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
    
    def forward(self, x):
        """
        Selective SSM forward pass
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # A 파라미터 (고정, 음수로 유지)
        A = -torch.exp(self.A_log)  # (d_model, d_state)
        
        # 입력 의존적 파라미터 생성
        # delta: (batch, seq_len, d_model)
        delta = F.softplus(self.dt_proj(self.x_proj(x)))
        delta = torch.clamp(delta, self.dt_min, self.dt_max)
        
        # B: (batch, seq_len, d_state)
        B = self.B_proj(x)
        
        # C: (batch, seq_len, d_state)
        C = self.C_proj(x)
        
        # SSM 계산 (순차 처리)
        # 초기 상태: (batch, d_model, d_state)
        h = torch.zeros(batch_size, d_model, self.d_state, device=x.device)
        
        outputs = []
        
        for t in range(seq_len):
            # 현재 시간 스텝의 파라미터들
            delta_t = delta[:, t, :].unsqueeze(-1)  # (batch, d_model, 1)
            B_t = B[:, t, :].unsqueeze(1)  # (batch, 1, d_state)
            C_t = C[:, t, :].unsqueeze(1)  # (batch, 1, d_state)
            x_t = x[:, t, :].unsqueeze(-1)  # (batch, d_model, 1)
            
            # 이산화
            # A_bar = exp(delta * A)
            A_bar = torch.exp(delta_t * A.unsqueeze(0))  # (batch, d_model, d_state)
            
            # B_bar = delta * B
            B_bar = delta_t * B_t  # (batch, d_model, d_state)
            
            # 상태 업데이트: h_t = A_bar * h_{t-1} + B_bar * x_t
            h = A_bar * h + B_bar * x_t  # (batch, d_model, d_state)
            
            # 출력: y_t = C * h_t + D * x_t
            y_t = (C_t * h).sum(dim=-1)  # (batch, d_model)
            y_t = y_t + self.D * x[:, t, :]  # skip connection
            
            outputs.append(y_t)
        
        # (batch, seq_len, d_model)
        y = torch.stack(outputs, dim=1)
        
        return y


if __name__ == "__main__":
    # 테스트
    batch_size = 2
    seq_len = 10
    d_model = 64
    d_state = 16
    
    print("=== Selective SSM 테스트 ===\n")
    
    model = SelectiveSSM(d_model, d_state)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"입력 shape: {x.shape}")
    
    y = model(x)
    print(f"출력 shape: {y.shape}")
    
    print(f"\n파라미터 정보:")
    print(f"  A_log: {model.A_log.shape}")
    print(f"  D: {model.D.shape}")
    print(f"  dt_rank: {model.dt_rank}")
    
    # 입력에 따라 파라미터가 어떻게 변하는지 확인
    with torch.no_grad():
        delta = F.softplus(model.dt_proj(model.x_proj(x)))
        B = model.B_proj(x)
        C = model.C_proj(x)
        
        print(f"\n동적 파라미터 shape:")
        print(f"  delta: {delta.shape} - 시간 스케일 (입력마다 다름)")
        print(f"  B: {B.shape} - 입력 가중치 (입력마다 다름)")
        print(f"  C: {C.shape} - 출력 가중치 (입력마다 다름)")
        
        print(f"\ndelta 통계 (첫 번째 배치):")
        print(f"  평균: {delta[0].mean().item():.4f}")
        print(f"  최소: {delta[0].min().item():.4f}")
        print(f"  최대: {delta[0].max().item():.4f}")
