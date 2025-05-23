import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel
from typing import List, Tuple, Optional, Union


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initialize spectral convergence loss module."""
        super().__init__()

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.
        
        Args:
            x_mag: Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag: Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
            
        Returns:
            Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p=1) / (torch.norm(y_mag, p=1) + 1e-8)


class STFTLoss(nn.Module):
    """STFT loss module."""

    def __init__(
        self, 
        fft_size: int = 1024, 
        shift_size: int = 120, 
        win_length: int = 600, 
        window: str = "hann",
        sample_rate: int = 24000
    ):
        """Initialize STFT loss module.
        
        Args:
            fft_size: FFT size for STFT computation.
            shift_size: Hop length for STFT computation.
            win_length: Window length for STFT computation.
            window: Window function type.
            sample_rate: Sample rate of audio.
        """
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.sample_rate = sample_rate
        
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=fft_size,
            win_length=win_length,
            hop_length=shift_size,
            window_fn=getattr(torch, f"{window}_window")
        )
        
        # Register normalization parameters as buffers
        self.register_buffer('mean', torch.tensor(-4.0))
        self.register_buffer('std', torch.tensor(4.0))
        
        self.spectral_convergence_loss = SpectralConvergenceLoss()

    def _normalize_mel(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Normalize mel spectrogram."""
        log_mel = torch.log(mel_spec + 1e-5)
        return (log_mel - self.mean) / self.std

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.
        
        Args:
            x: Predicted signal (B, T).
            y: Groundtruth signal (B, T).
            
        Returns:
            Spectral convergence loss value.
        """
        x_mag = self.to_mel(x)
        x_mag = self._normalize_mel(x_mag)
        
        y_mag = self.to_mel(y)
        y_mag = self._normalize_mel(y_mag)
        
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        return sc_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes: List[int] = [1024, 2048, 512],
        hop_sizes: List[int] = [120, 240, 50],
        win_lengths: List[int] = [600, 1200, 240],
        window: str = "hann",
        sample_rate: int = 24000
    ):
        """Initialize Multi resolution STFT loss module.
        
        Args:
            fft_sizes: List of FFT sizes.
            hop_sizes: List of hop sizes.
            win_lengths: List of window lengths.
            window: Window function type.
            sample_rate: Sample rate of audio.
        """
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths), \
            "All parameter lists must have the same length"
            
        self.stft_losses = nn.ModuleList([
            STFTLoss(fs, hs, wl, window, sample_rate)
            for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.
        
        Args:
            x: Predicted signal (B, T).
            y: Groundtruth signal (B, T).
            
        Returns:
            Multi resolution spectral convergence loss value.
        """
        sc_loss = torch.stack([
            stft_loss(x, y) for stft_loss in self.stft_losses
        ]).mean()
        
        return sc_loss


def feature_loss(fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
    """Compute feature matching loss between real and generated feature maps.
    
    Args:
        fmap_r: Real feature maps from discriminator.
        fmap_g: Generated feature maps from discriminator.
        
    Returns:
        Feature matching loss.
    """
    loss = torch.tensor(0.0, device=fmap_r[0][0].device)
    
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += F.l1_loss(rl, gl)
    
    return loss * 2


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor], 
    disc_generated_outputs: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[float], List[float]]:
    """Compute discriminator loss.
    
    Args:
        disc_real_outputs: Discriminator outputs for real samples.
        disc_generated_outputs: Discriminator outputs for generated samples.
        
    Returns:
        Tuple of (total_loss, real_losses, generated_losses).
    """
    loss = torch.tensor(0.0, device=disc_real_outputs[0].device)
    r_losses = []
    g_losses = []
    
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = F.mse_loss(dr, torch.ones_like(dr))
        g_loss = F.mse_loss(dg, torch.zeros_like(dg))
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Compute generator loss.
    
    Args:
        disc_outputs: Discriminator outputs for generated samples.
        
    Returns:
        Tuple of (total_loss, individual_losses).
    """
    gen_losses = [F.mse_loss(dg, torch.ones_like(dg)) for dg in disc_outputs]
    loss = torch.stack(gen_losses).sum()
    
    return loss, gen_losses


def discriminator_tprls_loss(
    disc_real_outputs: List[torch.Tensor], 
    disc_generated_outputs: List[torch.Tensor],
    tau: float = 0.04
) -> torch.Tensor:
    """Two-sided Penalty Regularized Least Squares discriminator loss.
    
    Reference: https://dl.acm.org/doi/abs/10.1145/3573834.3574506
    
    Args:
        disc_real_outputs: Discriminator outputs for real samples.
        disc_generated_outputs: Discriminator outputs for generated samples.
        tau: Regularization parameter.
        
    Returns:
        TPRLS discriminator loss.
    """
    loss = torch.tensor(0.0, device=disc_real_outputs[0].device)
    
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        diff = dr - dg
        m_dg = torch.median(diff)
        mask = dr < dg + m_dg
        
        if mask.any():
            l_rel = ((diff - m_dg) ** 2)[mask].mean()
            loss += tau - F.relu(tau - l_rel)
    
    return loss


def generator_tprls_loss(
    disc_real_outputs: List[torch.Tensor], 
    disc_generated_outputs: List[torch.Tensor],
    tau: float = 0.04
) -> torch.Tensor:
    """Two-sided Penalty Regularized Least Squares generator loss.
    
    Args:
        disc_real_outputs: Discriminator outputs for real samples.
        disc_generated_outputs: Discriminator outputs for generated samples.
        tau: Regularization parameter.
        
    Returns:
        TPRLS generator loss.
    """
    loss = torch.tensor(0.0, device=disc_real_outputs[0].device)
    
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        diff = dr - dg
        m_dg = torch.median(diff)
        mask = dr < dg + m_dg
        
        if mask.any():
            l_rel = ((diff - m_dg) ** 2)[mask].mean()
            loss += tau - F.relu(tau - l_rel)
    
    return loss


class GeneratorLoss(nn.Module):
    """Combined generator loss module."""

    def __init__(self, mpd: nn.Module, msd: nn.Module):
        """Initialize generator loss.
        
        Args:
            mpd: Multi-period discriminator.
            msd: Multi-scale discriminator.
        """
        super().__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """Calculate generator loss.
        
        Args:
            y: Ground truth audio.
            y_hat: Generated audio.
            
        Returns:
            Combined generator loss.
        """
        # Multi-period discriminator
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_hat)
        # Multi-scale discriminator
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_hat)
        
        # Feature matching losses
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        
        # Adversarial losses
        loss_gen_f, _ = generator_loss(y_df_hat_g)
        loss_gen_s, _ = generator_loss(y_ds_hat_g)
        
        # TPRLS losses
        loss_rel = (
            generator_tprls_loss(y_df_hat_r, y_df_hat_g) + 
            generator_tprls_loss(y_ds_hat_r, y_ds_hat_g)
        )
        
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_rel
        
        return loss_gen_all


class DiscriminatorLoss(nn.Module):
    """Combined discriminator loss module."""

    def __init__(self, mpd: nn.Module, msd: nn.Module):
        """Initialize discriminator loss.
        
        Args:
            mpd: Multi-period discriminator.
            msd: Multi-scale discriminator.
        """
        super().__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """Calculate discriminator loss.
        
        Args:
            y: Ground truth audio.
            y_hat: Generated audio.
            
        Returns:
            Combined discriminator loss.
        """
        # Multi-period discriminator
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_hat)
        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
        
        # Multi-scale discriminator
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_hat)
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        
        # TPRLS losses
        loss_rel = (
            discriminator_tprls_loss(y_df_hat_r, y_df_hat_g) + 
            discriminator_tprls_loss(y_ds_hat_r, y_ds_hat_g)
        )
        
        d_loss = loss_disc_s + loss_disc_f + loss_rel
        
        return d_loss


class WavLMLoss(nn.Module):
    """WavLM-based perceptual loss module."""

    def __init__(
        self, 
        model: str, 
        wd: nn.Module, 
        model_sr: int, 
        slm_sr: int = 16000
    ):
        """Initialize WavLM loss.
        
        Args:
            model: WavLM model name/path.
            wd: WavLM discriminator module.
            model_sr: Model sample rate.
            slm_sr: WavLM sample rate.
        """
        super().__init__()
        self.wavlm = AutoModel.from_pretrained(model)
        self.wavlm.eval()  # Set to eval mode
        
        # Freeze WavLM parameters
        for param in self.wavlm.parameters():
            param.requires_grad = False
            
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
     
    def _extract_embeddings(self, wav: torch.Tensor) -> List[torch.Tensor]:
        """Extract WavLM embeddings."""
        wav_16 = self.resample(wav)
        with torch.no_grad():
            embeddings = self.wavlm(
                input_values=wav_16, 
                output_hidden_states=True
            ).hidden_states
        return embeddings
    
    def forward(self, wav: torch.Tensor, y_rec: torch.Tensor) -> torch.Tensor:
        """Calculate WavLM reconstruction loss.
        
        Args:
            wav: Ground truth audio.
            y_rec: Reconstructed audio.
            
        Returns:
            WavLM reconstruction loss.
        """
        wav_embeddings = self._extract_embeddings(wav)
        
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16.squeeze(), 
            output_hidden_states=True
        ).hidden_states

        floss = torch.stack([
            F.l1_loss(er, eg) for er, eg in zip(wav_embeddings, y_rec_embeddings)
        ]).mean()
        
        return floss
    
    def generator(self, y_rec: torch.Tensor) -> torch.Tensor:
        """Calculate WavLM generator loss.
        
        Args:
            y_rec: Reconstructed audio.
            
        Returns:
            WavLM generator loss.
        """
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16, 
            output_hidden_states=True
        ).hidden_states
        
        # Stack and reshape embeddings
        y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1)
        y_rec_embeddings = y_rec_embeddings.transpose(-1, -2).flatten(1, 2)
        
        y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = F.mse_loss(y_df_hat_g, torch.ones_like(y_df_hat_g))
        
        return loss_gen
    
    def discriminator(self, wav: torch.Tensor, y_rec: torch.Tensor) -> torch.Tensor:
        """Calculate WavLM discriminator loss.
        
        Args:
            wav: Ground truth audio.
            y_rec: Reconstructed audio.
            
        Returns:
            WavLM discriminator loss.
        """
        with torch.no_grad():
            wav_embeddings = self._extract_embeddings(wav)
            y_rec_embeddings = self._extract_embeddings(y_rec)
            
            # Stack and reshape embeddings
            y_embeddings = torch.stack(wav_embeddings, dim=1)
            y_embeddings = y_embeddings.transpose(-1, -2).flatten(1, 2)
            
            y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1)
            y_rec_embeddings = y_rec_embeddings.transpose(-1, -2).flatten(1, 2)

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)
        
        r_loss = F.mse_loss(y_d_rs, torch.ones_like(y_d_rs))
        g_loss = F.mse_loss(y_d_gs, torch.zeros_like(y_d_gs))
        
        loss_disc_f = r_loss + g_loss
                        
        return loss_disc_f

    def discriminator_forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Forward pass through WavLM discriminator.
        
        Args:
            wav: Input audio.
            
        Returns:
            Discriminator output.
        """
        with torch.no_grad():
            wav_embeddings = self._extract_embeddings(wav)
            y_embeddings = torch.stack(wav_embeddings, dim=1)
            y_embeddings = y_embeddings.transpose(-1, -2).flatten(1, 2)

        y_d_rs = self.wd(y_embeddings)
        
        return y_d_rs
