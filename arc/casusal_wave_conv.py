import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalWaveConv(nn.Module):
    def __init__(self, dim, n_wave_heads=4, n_scales=4,
                 sigma_scales=None, max_seq=512, dropout=0.0):
        super().__init__()
        self.dim      = dim
        self.H        = n_wave_heads
        self.K        = n_scales
        self.head_dim = dim // n_wave_heads
        assert dim % n_wave_heads == 0

        if sigma_scales is None:
            sigma_scales = [1.0, 4.0, 16.0, 64.0]
        assert len(sigma_scales) == n_scales

        self.register_buffer(
            'sigma_scales',
            torch.tensor(sigma_scales, dtype=torch.float)
        )

        self.log_amp   = nn.Parameter(torch.zeros(n_wave_heads, n_scales))
        self.mu_shift  = nn.Parameter(torch.zeros(n_wave_heads, n_scales))
        self.head_freq_bias = nn.Parameter(torch.randn(n_wave_heads, n_scales) * 0.01)
        with torch.no_grad():
            for k in range(n_scales):
                self.mu_shift[:, k] = float(k) * 0.5

        self.crystal_temp = nn.Parameter(torch.ones(n_wave_heads) * 2.0)
        self.W_v      = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.drop     = nn.Dropout(dropout)
        self._max_seq = max_seq
        self._static_basis_cache = {}

    def _get_static_basis(self, L_padded: int, device):
        device_key = (device.type, device.index if device.index is not None else -1)
        cache_key = (L_padded, device_key)
        cached = self._static_basis_cache.get(cache_key)
        if cached is not None:
            return cached

        n_freq = L_padded // 2 + 1
        omega = (
            torch.arange(n_freq, device=device, dtype=torch.float)
            * (2 * math.pi / L_padded)
        )
        omega3 = omega.unsqueeze(0).unsqueeze(0)
        sigma = self.sigma_scales.unsqueeze(0).unsqueeze(-1)
        gauss = torch.exp(-0.5 * sigma**2 * omega3**2)

        causal_mask = torch.zeros(L_padded, device=device)
        causal_mask[:L_padded // 2 + 1] = 1.0

        cached = (omega3, gauss, causal_mask)
        self._static_basis_cache[cache_key] = cached
        return cached

    def _build_freq_kernel_causal(self, L_padded: int, device):
        omega3, gauss, causal_mask = self._get_static_basis(L_padded, device)
        n_freq = omega3.size(-1)

        amp    = torch.exp(self.log_amp).unsqueeze(-1)
        mu     = (self.mu_shift + self.head_freq_bias).unsqueeze(-1)

        # H(ω) = amp · Gauss · e^{-jμω}
        H_freq = amp * gauss * torch.polar(
            torch.ones(self.H, self.K, n_freq, device=device),
            -mu.expand(self.H, self.K, n_freq)
                * omega3.expand(self.H, self.K, n_freq)
        )                                                       

        h_time = torch.fft.irfft(H_freq, n=L_padded, dim=-1)  
        h_causal = h_time * causal_mask 
        H_causal = torch.fft.rfft(h_causal, n=L_padded, dim=-1)
        return H_causal

    def _spectral_crystallize(self, x_freq, head_idx):
        return x_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        Dh = self.head_dim

        v = self.W_v(x).reshape(B, L, self.H, Dh).permute(0, 2, 1, 3)

        L_pad   = L
        L_total = L + L_pad
        kernel = self._build_freq_kernel_causal(L_total, x.device)

        v_padded = F.pad(v.float(), (0, 0, L_pad, 0))
        v_freq = torch.fft.rfft(v_padded, n=L_total, dim=2)
        v_exp = v_freq.unsqueeze(2).contiguous()
        k_exp = kernel.unsqueeze(0).unsqueeze(-1).contiguous()
        super_freq = (v_exp * k_exp).sum(dim=2)
        out_full = torch.fft.irfft(super_freq, n=L_total, dim=2)
        out = out_full[:, :, L_pad:, :].to(x.dtype)
        out = out.permute(0, 2, 1, 3).reshape(B, L, D)
        out = self.drop(out)
        return self.out_proj(out)

    def get_superposition_coherence(self, x: torch.Tensor) -> float:
        B, L, D = x.shape
        L_total = L * 2
        kernel  = self._build_freq_kernel_causal(L_total, x.device)  

        sum_k   = kernel.sum(dim=1).abs() ** 2       
        sum_sq  = (kernel.abs() ** 2).sum(dim=1)           
        coherence = sum_k / (self.K * sum_sq + 1e-8)
        return coherence.mean().item()
