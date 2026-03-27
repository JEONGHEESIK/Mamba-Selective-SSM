import torch
import torch.nn as nn
import math

class BasicSSM(nn.Module):
    """
    기본 State Space Model (SSM) 구현
    
    연속 시간 시스템:
        h'(t) = A*h(t) + B*x(t)
        y(t) = C*h(t)
    
    이산화 (Zero-Order Hold):
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C * h_t
    
    여기서:
        A_bar = exp(Δ * A)
        B_bar = (A_bar - I) * A^{-1} * B
    """
    
    def __init__(self, d_model, d_state=16):
        """
        Args:
            d_model: 입력/출력 차원
            d_state: 숨겨진 상태 차원 (N)
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # SSM 파라미터
        # A: (d_state, d_state) - 상태 전이 행렬
        # 일반적으로 대각 행렬로 초기화 (효율성)
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        
        # B: (d_state, d_model) - 입력 행렬
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        
        # C: (d_model, d_state) - 출력 행렬
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        
        # Δ (delta): 시간 스케일 파라미터
        self.delta = nn.Parameter(torch.ones(1))
        
    def discretize(self):
        """
        연속 시간 파라미터를 이산 시간으로 변환
        
        Returns:
            A_bar: 이산화된 상태 전이 행렬
            B_bar: 이산화된 입력 행렬
        """
        # A_bar = exp(Δ * A)
        A_bar = torch.matrix_exp(self.delta * self.A)
        
        # B_bar = (A_bar - I) * A^{-1} * B
        # 간단히: B_bar ≈ Δ * B (작은 Δ에 대한 근사)
        B_bar = self.delta * self.B
        
        return A_bar, B_bar
    
    def forward(self, x):
        """
        순차적 처리 (RNN 스타일)
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 이산화
        A_bar, B_bar = self.discretize()
        
        # 초기 상태
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        
        outputs = []
        
        for t in range(seq_len):
            # x_t: (batch, d_model)
            x_t = x[:, t, :]
            
            # 상태 업데이트: h_t = A_bar * h_{t-1} + B_bar * x_t
            # h: (batch, d_state), B_bar: (d_state, d_model), x_t: (batch, d_model)
            h = torch.matmul(h, A_bar.T) + torch.matmul(x_t, B_bar.T)
            
            # 출력: y_t = C * h_t
            # C: (d_model, d_state), h: (batch, d_state)
            y_t = torch.matmul(h, self.C.T)
            
            outputs.append(y_t)
        
        # (batch, seq_len, d_model)
        y = torch.stack(outputs, dim=1)
        
        return y


if __name__ == "__main__":
    # 간단한 테스트
    batch_size = 2
    seq_len = 10
    d_model = 64
    d_state = 16
    
    model = BasicSSM(d_model, d_state)
    x = torch.randn(batch_size, seq_len, d_model)
    
    y = model(x)
    
    print(f"입력 shape: {x.shape}")
    print(f"출력 shape: {y.shape}")
    print(f"\nSSM 파라미터:")
    print(f"  A: {model.A.shape}")
    print(f"  B: {model.B.shape}")
    print(f"  C: {model.C.shape}")
    print(f"  delta: {model.delta.item():.4f}")
