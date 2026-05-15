from typing import List

import torch
import torchaudio
from torch import nn
import math
from decoder.modules import safe_log
from encoder.modules import SEANetEncoder, SEANetDecoder
from encoder import EncodecModel
from encoder.quantization import VectorQuantizer
from einops import rearrange

class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100, padding="center"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def forward(self, audio, **kwargs):
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
        mel = self.mel_spec(audio)
        features = safe_log(mel)
        return features


class EncodecFeatures(FeatureExtractor):
    def __init__(
        self,
        encodec_model: str = "encodec_24khz",
        bandwidths: List[float] = [1.5, 3.0, 6.0, 12.0],
        num_quantizers: int = 1,
        train_codebooks: bool = False,
        dowmsamples: List[int] = [6, 5, 5, 4],
        vq_kmeans: int = 800,
        vq_type: str = 'hexagon',
        codebook_dim: List[int] = [7, 7, 7, 7, 7, 7],
    ):
        super().__init__()

        self.frame_rate = 25  # not use
        n_q = num_quantizers   # important
        encoder = SEANetEncoder(causal=False, n_residual_layers=1, norm='weight_norm', pad_mode='reflect', lstm=2,
                                dimension=512, channels=1, n_filters=32, ratios=dowmsamples, activation='ELU',
                                kernel_size=7, residual_kernel_size=3, last_kernel_size=7, dilation_base=2,
                                true_skip=False, compress=2)
        decoder = SEANetDecoder(causal=False, n_residual_layers=1, norm='weight_norm', pad_mode='reflect', lstm=2,
                                dimension=512, channels=1, n_filters=32, ratios=[8, 5, 4, 2], activation='ELU',
                                kernel_size=7, residual_kernel_size=3, last_kernel_size=7, dilation_base=2,
                                true_skip=False, compress=2)
        quantizer = VectorQuantizer(vq_type=vq_type, dimension=512, codebook_dim = codebook_dim, n_q=n_q, kmeans_iters=vq_kmeans, decay=0.99, kmeans_init=True)

        if encodec_model == "encodec_24khz":
            self.encodec = EncodecModel(encoder=encoder, decoder=decoder, quantizer=quantizer,
                                        target_bandwidths=bandwidths, sample_rate=24000, channels=1)
        else:
            raise ValueError(
                f"Unsupported encodec_model: {encodec_model}. Supported options are 'encodec_24khz'."
            )
        for param in self.encodec.parameters():
            param.requires_grad = True

        self.bandwidths = bandwidths

    def forward(self, audio: torch.Tensor, bandwidth_id: torch.Tensor):
        if self.training:
            self.encodec.train()

        audio = audio.unsqueeze(1)                  # audio(16,24000)

        emb = self.encodec.encoder(audio)
        q_res = self.encodec.quantizer(emb, self.frame_rate, bandwidth=self.bandwidths[bandwidth_id])
        quantized = q_res.quantized
        quantized = rearrange (quantized, 'b n d -> b d n')
        codes = q_res.codes
        commit_loss = q_res.penalty
        return quantized, codes, commit_loss


    def infer(self, audio: torch.Tensor, bandwidth_id: torch.Tensor):
        if self.training:
            self.encodec.train()

        audio = audio.unsqueeze(1)                  # audio(16,24000)
        emb = self.encodec.encoder(audio)
        q_res = self.encodec.quantizer.infer(emb, self.frame_rate, bandwidth=self.bandwidths[bandwidth_id])
        quantized = q_res.quantized
        codes = q_res.codes
        commit_loss = q_res.penalty                 #

        return quantized, codes, commit_loss
