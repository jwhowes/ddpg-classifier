from typing import Optional, List, Tuple
from math import sqrt

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import crop, resize
from einops import rearrange
from torch import nn, Tensor


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super(Attention, self).__init__()
        assert d_model % n_heads == 0

        self.scale = sqrt(d_model / n_heads)
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def apply_rotary_emb(x: Tensor, freqs: Tensor) -> Tensor:
        return torch.view_as_real(
            torch.view_as_complex(x.unflatten(-1, (-1, 2))) *
            freqs
        ).flatten(-2)

    def forward(self, x: Tensor, freqs: Tensor, attention_mask: Tensor) -> Tensor:
        q = rearrange(self.W_q(x), "b l (n d) -> b n l d", n=self.n_heads)
        k = rearrange(self.W_k(x), "b l (n d) -> b n l d", n=self.n_heads)
        v = rearrange(self.W_v(x), "b l (n d) -> b n l d", n=self.n_heads)

        attn = (
            self.apply_rotary_emb(q, freqs) @
            self.apply_rotary_emb(k, freqs).transpose(-2, -1)
        ) / self.scale

        return self.W_o(
            rearrange(
                F.softmax(attn + attention_mask, dim=-1) @ v, "b n l d -> b l (n d)"
            )
        )


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: Optional[int] = None):
        super(SwiGLU, self).__init__()
        if d_hidden is None:
            d_hidden = 4 * d_model

        self.gate = nn.Linear(d_model, d_hidden, bias=False)
        self.hidden = nn.Linear(d_model, d_hidden, bias=False)
        self.out = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(
            F.silu(self.gate(x)) * self.hidden(x)
        )


class StackedSinusoidalEmbedding(nn.Module):
    def __init__(self, d_model: int, num_items: int, base: float = 1e4):
        super(StackedSinusoidalEmbedding, self).__init__()
        assert d_model % num_items == 0
        assert (d_model // num_items) % 2 == 0

        d_stack = d_model // num_items
        self.register_buffer(
            "theta",
            1.0 / (base ** (2 * torch.arange(d_stack // 2) / d_stack))
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1) * self.theta

        return torch.stack((
            x.cos(),
            x.sin()
        ), dim=-1).flatten(-3)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super(TransformerBlock, self).__init__()
        self.attn = Attention(d_model, n_heads)
        self.attn_norm = nn.RMSNorm(d_model, eps=1e-6)

        self.ffn = SwiGLU(d_model)
        self.ffn_norm = nn.RMSNorm(d_model, eps=1e-6)

    def forward(self, x: Tensor, freqs: Tensor, attention_mask: Tensor) -> Tensor:
        x = x + self.attn(
            self.attn_norm(x), freqs, attention_mask
        )

        return x + self.ffn(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, max_length: int):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        self.register_buffer(
            "attention_mask",
            torch.full((max_length, max_length), float('-inf')).triu(1),
            persistent=False
        )

        assert d_model % n_heads == 0
        assert (d_model // n_heads) % 2 == 0

        d_attn = d_model // n_heads
        theta = 1.0 / (1e4 ** (2 * torch.arange(d_attn // 2) / d_attn))

        freqs = torch.outer(torch.arange(max_length), theta)

        self.register_buffer(
            "freqs",
            torch.polar(torch.ones_like(freqs), freqs),
            persistent=False
        )

    def forward(self, x: Tensor) -> Tensor:
        L = x.shape[1]

        for layer in self.layers:
            x = layer(x, self.freqs[:L], self.attention_mask[:L, :L])

        return x


class Agent(nn.Module):
    def __init__(
            self, in_channels: int, patch_size: int, num_labels: int, d_model: int, n_layers: int, n_heads: int,
            max_length: int
    ):
        super(Agent, self).__init__()
        self.patch_size = patch_size
        self.max_length = max_length
        self.in_channels = in_channels

        self.transformer = Transformer(d_model, n_layers, n_heads, max_length)

        self.patch_emb = nn.Linear(patch_size * patch_size * in_channels, d_model)
        self.box_emb = nn.Sequential(
            StackedSinusoidalEmbedding(d_model, 4),
            nn.Linear(d_model, d_model)
        )

        self.reward_head = nn.Sequential(
            nn.RMSNorm(d_model, eps=1e-6),
            nn.Linear(d_model, 2)
        )

        self.label_head = nn.Sequential(
            nn.RMSNorm(d_model, eps=1e-6),
            nn.Linear(d_model, num_labels)
        )

        self.box_head = nn.Sequential(
            nn.RMSNorm(d_model, eps=1e-6),
            nn.Linear(d_model, 4)
        )

    @torch.inference_mode()
    def random_history(self, image: Tensor, beta: float = 0.1, sigma: float = 0.3) -> List[Tuple[Tensor, Tensor]]:
        B = image.shape[0]
        image_size = image.shape[2]

        box = torch.zeros(B, 1, 4, device=image.device)

        box[:, 0, [2, 3]] = 1.0

        patch = torch.zeros(B, 0, self.in_channels * self.patch_size * self.patch_size, device=image.device)

        t = 0
        history = []
        while B > 0 and t < self.max_length:
            t += 1

            p = []
            for img, b in zip(image, box):
                y, x, h, w = (int(round(b_.item() * image_size, 0)) for b_ in b[-1])
                w = max(w, self.patch_size)
                h = max(h, self.patch_size)

                if x + w > image_size:
                    x = image_size - w

                if y + h > image_size:
                    h -= image_size - h

                p.append(
                    resize(
                        crop(img, y, x, h, w),
                        (self.patch_size, self.patch_size)
                    ).flatten()
                )

            patch = torch.concatenate((
                patch,
                torch.stack(p).unsqueeze(1)
            ), dim=1)

            reward_pred, _, box_pred = self(patch, box)

            beta_mask = torch.rand(B, device=image.device) < beta

            terminate = (
                    (beta_mask & (torch.rand(B, device=image.device) < 0.5)) |
                    (~beta_mask & (reward_pred[:, -1].argmax(-1) == 0))
            )

            history.extend([
                (p, b) for p, b, t in zip(patch, box, terminate) if t
            ])

            image = image[~terminate]
            box = box[~terminate]

            B = image.shape[0]

            if B == 0:
                break

            box = torch.concatenate((
                box,
                (box_pred[~terminate, -1] + torch.randn(B, 4, device=image.device) * sigma).sigmoid().unsqueeze(1)
            ), dim=1)

        if patch.shape[0] > 0:
            history.extend([
                (p, b) for p, b in zip(patch, box)
            ])

        return history

    def forward(self, patch: Tensor, box: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.patch_emb(patch) + self.box_emb(box)

        x = self.transformer(x)

        return self.reward_head(x), self.label_head(x), self.box_head(x)


class Critic(nn.Module):
    def __init__(
            self, in_channels: int, patch_size: int, d_model: int, n_layers: int, n_heads: int,
            max_length: int
    ):
        super(Critic, self).__init__()

        self.transformer = Transformer(d_model, n_layers, n_heads, max_length)

        self.patch_emb = nn.Linear(patch_size * patch_size * in_channels, d_model)
        self.box_emb = nn.Sequential(
            StackedSinusoidalEmbedding(d_model, 4),
            nn.Linear(d_model, d_model)
        )
        self.next_box_emb = nn.Sequential(
            StackedSinusoidalEmbedding(d_model, 4),
            nn.Linear(d_model, d_model)
        )

        self.head = nn.Sequential(
            nn.RMSNorm(d_model, eps=1e-6),
            nn.Linear(d_model, 1)
        )

    def forward(self, patch: Tensor, box: Tensor):
        x = self.patch_emb(patch) + self.box_emb(box) + self.next_box_emb(box)

        x = self.transformer(x)

        return self.head(x)
