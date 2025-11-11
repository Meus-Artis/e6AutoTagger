import re
from collections import defaultdict
from math import sqrt
from typing import Any, Iterable, Self, cast

import torch
from torch import Tensor
from torch.nn import (
    Module, ModuleList, Parameter, Buffer,
    Linear, LayerNorm, RMSNorm, Dropout, Flatten,
    init
)
from torch.nn.functional import pad, scaled_dot_product_attention

from einops import rearrange

from glu import SwiGLU

class IndexedAdd(Module):
    def __init__(
        self,
        n_indices: int,
        dim: int,
        weight_shape: tuple[int, ...] | None = None,
        *,
        inplace: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.inplace = inplace

        self.index = Buffer(torch.empty(
            2, n_indices,
            device=device, dtype=torch.int32
        ))

        self.weight = Parameter(torch.ones(
            *(sz if sz != -1 else n_indices for sz in weight_shape),
            device=device, dtype=dtype
        )) if weight_shape is not None else None

    def _save_to_state_dict(
        self,
        destination: dict[str, Any],
        prefix: str,
        keep_vars: bool
    ) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)

        if keep_vars:
            return

        with torch.no_grad():
            index_key = f"{prefix}index"
            index = destination[index_key]

            min_index = index.amin(None).item()
            if min_index >= 0:
                max_index = index.amax(None).item()
                if max_index < (1 << 8):
                    destination[index_key] = index.to(dtype=torch.uint8)
                elif max_index < (1 << 16):
                    destination[index_key] = index.to(dtype=torch.uint16)

    @torch.no_grad()
    def load_indices(self, indices: Iterable[tuple[int, int]], *, mean: bool = False) -> None:
        if mean:
            if self.weight is None:
                raise ValueError("No weights to initialize with means.")

            groups: dict[int, list[int]] = defaultdict(list)

        idx = -1
        for idx, (src, dst) in enumerate(indices):
            self.index[0, idx] = src
            self.index[1, idx] = dst

            if mean:
                groups[dst].append(idx)

        if (idx + 1) != self.index.size(1):
            raise IndexError(f"Expected {self.index.size(1)} indices, but got {idx + 1}.")

        if not mean:
            return

        assert self.weight is not None

        for idxs in groups.values():
            if len(idxs) < 2:
                continue

            self.weight.index_fill_(
                self.dim,
                torch.tensor(idxs, device=self.weight.device, dtype=torch.int64),
                1.0 / len(idxs)
            )

    def forward(self, dst: Tensor, src: Tensor) -> Tensor:
        src = src.index_select(self.dim, self.index[0])

        if self.weight is not None:
            src.mul_(self.weight)

        return (
            dst.index_add_(self.dim, self.index[1], src)
            if self.inplace else
            dst.index_add(self.dim, self.index[1], src)
        )

class BatchLinear(Module):
    def __init__(
        self,
        batch_shape: tuple[int, ...] | int,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        flatten: bool = False,
        bias_inplace: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        elif not batch_shape:
            raise ValueError("At least one batch dimension is required.")

        self.flatten = -(len(batch_shape) + 1) if flatten else 0

        self.weight = Parameter(torch.empty(
            *batch_shape, in_features, out_features,
            device=device, dtype=dtype
        ))

        bt = self.weight.flatten(end_dim=-3).mT
        for idx in range(bt.size(0)):
            init.kaiming_uniform_(bt[idx], a=sqrt(5))

        self.bias = Parameter(torch.zeros(
            *batch_shape, out_features,
            device=device, dtype=dtype
        )) if bias else None

        self.bias_inplace = bias_inplace

    def forward(self, x: Tensor) -> Tensor:
        # ... B... 1 I @ B... I O -> ... B... O
        x = torch.matmul(x.unsqueeze(-2), self.weight).squeeze(-2)

        if self.bias is not None:
            if self.bias_inplace:
                x.add_(self.bias)
            else:
                x = x + self.bias

        if self.flatten:
            x = x.flatten(self.flatten)

        return x

class Mean(Module):
    def __init__(self, dim: tuple[int, ...] | int = -1, *, keepdim: bool = False) -> None:
        super().__init__()

        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(self.dim, self.keepdim)

class _MidBlock(Module):
    def __init__(
        self,
        attn_dim: int,
        head_dim: int,
        n_classes: int,
        *,
        ff_ratio: float,
        ff_dropout: float,
        q_cls_inplace: bool = True,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()

        self.head_dim = head_dim
        self.q_cls_inplace = q_cls_inplace

        hidden_dim = int(attn_dim * ff_ratio)

        self.q_proj = Linear(
            attn_dim, attn_dim, bias=False,
            device=device, dtype=dtype
        )

        self.q_cls = Parameter(torch.zeros(
            n_classes, attn_dim,
            device=device, dtype=dtype
        ))

        self.q_norm = RMSNorm(head_dim, eps=1e-5, elementwise_affine=False)

        self.attn_out = Linear(
            attn_dim, attn_dim, bias=False,
            device=device, dtype=dtype
        )

        self.ff_norm = LayerNorm(
            attn_dim,
            device=device, dtype=dtype
        )
        self.ff_in = Linear(
            attn_dim, hidden_dim * 2, bias=False,
            device=device, dtype=dtype
        )
        self.ff_act = SwiGLU()
        self.ff_drop = Dropout(ff_dropout)
        self.ff_out = Linear(
            hidden_dim, attn_dim, bias=False,
            device=device, dtype=dtype
        )

    def _forward_q(self, x: Tensor) -> Tensor:
        x = self.q_proj(x)

        if self.q_cls_inplace:
            x.add_(self.q_cls)
        else:
            x = x + self.q_cls

        x = self.q_norm(x)
        x = rearrange(x, "... s (h e) -> ... h s e", e=self.head_dim)
        return x

    def _forward_attn(self, x: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None) -> Tensor:
        a = scaled_dot_product_attention(
            self._forward_q(x), k, v,
            attn_mask=attn_mask
        )
        a = rearrange(a, "... h s e -> ... s (h e)")
        a = self.attn_out(a)
        return x + a

    def _forward_ff(self, x: Tensor) -> Tensor:
        f = self.ff_norm(x)
        f = self.ff_in(f)
        f = self.ff_act(f)
        f = self.ff_drop(f)
        f = self.ff_out(f)
        return x + f

    def forward(self, x: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        x = self._forward_attn(x, k, v, attn_mask)
        x = self._forward_ff(x)
        return x

class HydraPool(Module):
    def __init__(
        self,
        attn_dim: int,
        head_dim: int,
        n_classes: int,
        *,
        mid_blocks: int = 0,
        roots: tuple[int, int, int] = (0, 0, 0),
        ff_ratio: float = 3.0,
        ff_dropout: float = 0.0,
        input_dim: int = -1,
        output_dim: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if input_dim < 0:
            input_dim = attn_dim

        assert attn_dim % head_dim == 0
        n_heads = attn_dim // head_dim

        self.n_classes = n_classes
        self.head_dim = head_dim
        self.output_dim = output_dim

        self._has_roots = False
        self._has_ff = False

        self.q: Parameter | Buffer
        self._q_normed: bool | None

        if roots != (0, 0, 0):
            self._has_roots = True
            n_roots, n_classroots, n_subclasses = roots

            if n_classroots < n_roots:
                raise ValueError("Number of classroots cannot be less than the number of roots.")

            self.cls = Parameter(torch.randn(
                n_heads, n_classes, head_dim,
                device=device, dtype=dtype
            ))

            self.roots = Parameter(torch.randn(
                n_heads, n_roots, head_dim,
                device=device, dtype=dtype
            )) if n_roots > 0 else None

            self.clsroots = IndexedAdd(
                n_classroots, dim=-2, weight_shape=(n_heads, -1, 1),
                device=device, dtype=dtype
            ) if n_classroots > 0 else None

            self.clscls = IndexedAdd(
                n_subclasses, dim=-2, weight_shape=(n_heads, -1, 1),
                inplace=True, device=device, dtype=dtype
            ) if n_subclasses > 0 else None

            self.q = Buffer(torch.empty(
                n_heads, n_classes, head_dim,
                device=device, dtype=dtype
            ))
            self._q_normed = None
        else:
            self.q = Parameter(torch.randn(
                n_heads, n_classes, head_dim,
                device=device, dtype=dtype
            ))
            self._q_normed = False

        self.kv = Linear(
            input_dim, attn_dim * 2, bias=False,
            device=device, dtype=dtype
        )
        self.qk_norm = RMSNorm(
            head_dim, eps=1e-5, elementwise_affine=False
        )

        if ff_ratio > 0.0:
            self._has_ff = True
            hidden_dim = int(attn_dim * ff_ratio)

            self.ff_norm = LayerNorm(
                attn_dim,
                device=device, dtype=dtype
            )
            self.ff_in = Linear(
                attn_dim, hidden_dim * 2, bias=False,
                device=device, dtype=dtype
            )
            self.ff_act = SwiGLU()
            self.ff_drop = Dropout(ff_dropout)
            self.ff_out = Linear(
                hidden_dim, attn_dim, bias=False,
                device=device, dtype=dtype
            )
        elif mid_blocks > 0:
            raise ValueError("Feedforward required with mid blocks.")

        self.mid_blocks = ModuleList(
            _MidBlock(
                attn_dim, head_dim, n_classes,
                ff_ratio=ff_ratio, ff_dropout=ff_dropout,
                device=device, dtype=dtype
            ) for _ in range(mid_blocks)
        )

        self.out_proj = BatchLinear(
            n_classes, attn_dim, output_dim * 2,
            device=device, dtype=dtype
        )
        self.out_act = SwiGLU()

    @property
    def has_roots(self) -> bool:
        return self._has_roots

    def get_extra_state(self) -> dict[str, Any]:
        return { "q_normed": self._q_normed }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self._q_normed = state["q_normed"]

    def create_head(self) -> Module:
        if self.output_dim == 1:
            return Flatten(-2)

        return Mean(-1)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)

        if mode:
            if self._has_roots:
                self._q_normed = None
            else:
                self._q_normed = False
        else:
            if self._has_roots:
                self._cache_query()

        return self

    def inference(self) -> Self:
        super().train(False)
        self._cache_query()

        if self._has_roots:
            self._has_roots = False
            self.q = Parameter(self.q)

            del self.cls, self.roots, self.clsroots, self.clscls

        return self

    def _cache_query(self) -> None:
        assert not self.training

        if self._q_normed:
            return

        with torch.no_grad():
            self.q.to(device=self.kv.weight.device)
            self.q.copy_(self._forward_q())
            self._q_normed = True

    def _forward_q(self) -> Tensor:
        match self._q_normed:
            case None:
                assert self._has_roots

                if self.roots is not None:
                    q = self.qk_norm(self.roots)
                    q = self.clsroots(self.cls, q)
                else:
                    q = self.cls

                if self.clscls is not None:
                    q = self.clscls(q, q.detach())

                q = self.qk_norm(q)
                return q

            case False:
                assert not self._has_roots
                return self.qk_norm(self.q)

            case True:
                return self.q

    def _forward_attn(self, x: Tensor, attn_mask: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        q = self._forward_q().expand(*x.shape[:-2], -1, -1, -1)

        x = self.kv(x)
        k, v = rearrange(x, "... s (n h e) -> n ... h s e", n=2, e=self.head_dim).unbind(0)
        k = self.qk_norm(k)

        x = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return rearrange(x, "... h s e -> ... s (h e)"), k, v

    def _forward_ff(self, x: Tensor) -> Tensor:
        if not self._has_ff:
            return x

        f = self.ff_norm(x)
        f = self.ff_in(f)
        f = self.ff_act(f)
        f = self.ff_drop(f)
        f = self.ff_out(f)
        return x + f

    def _forward_out(self, x: Tensor) -> Tensor:
        x = self.out_proj(x)
        x = self.out_act(x)
        return x

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        x, k, v = self._forward_attn(x, attn_mask)
        x = self._forward_ff(x)

        for block in self.mid_blocks:
            x = block(x, k, v, attn_mask)

        x = self._forward_out(x)
        return x

    def prune_roots(self, retain_classes: set[int]) -> tuple[list[int], list[int]]:
        if not self._has_roots or self.roots is None:
            raise TypeError("No roots to prune.")

        if self.clscls is not None:
            raise TypeError("Subclass roots cannot be pruned.")

        used_roots: set[int] = set()
        used_clsroots: list[int] = []

        assert self.clsroots is not None
        clsroots = [
            cast(list[int], clsroot.tolist())
            for clsroot in self.clsroots.index.cpu().unbind(1)
        ]

        for idx, (src, dest) in enumerate(clsroots):
            if dest in retain_classes:
                used_roots.add(src)
                used_clsroots.append(idx)

        sorted_roots = sorted(used_roots)
        del used_roots

        rootmap = {
            root: idx
            for idx, root in enumerate(sorted_roots)
        }

        clsmap = {
            cls: idx
            for idx, cls in enumerate(sorted(retain_classes))
        }

        for idx in used_clsroots:
            src, dest = clsroots[idx]
            self.clsroots.index[0, idx] = rootmap[src]
            self.clsroots.index[1, idx] = clsmap[dest]

        return sorted_roots, used_clsroots

    @staticmethod
    def for_state(
        state_dict: dict[str, Any],
        prefix: str = "",
        *,
        ff_dropout: float = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "HydraPool":
        n_heads, n_classes, head_dim = state_dict[f"{prefix}q"].shape
        attn_dim = n_heads * head_dim

        roots_t = state_dict.get(f"{prefix}roots")
        clsroots_t = state_dict.get(f"{prefix}clsroots.index")
        clscls_t = state_dict.get(f"{prefix}clscls.index")
        roots = (
            roots_t.size(1) if roots_t is not None else 0,
            clsroots_t.size(1) if clsroots_t is not None else 0,
            clscls_t.size(1) if clscls_t is not None else 0
        )

        input_dim = state_dict[f"{prefix}kv.weight"].size(1)
        output_dim = state_dict[f"{prefix}out_proj.weight"].size(2) // 2

        # avoid off-by-one issue due to truncation
        ffout_t = state_dict.get(f"{prefix}ff_out.weight")
        hidden_dim = ffout_t.size(1) + 0.5 if ffout_t is not None else 0
        ff_ratio = hidden_dim / attn_dim

        pattern = re.compile(rf"^{re.escape(prefix)}mid_blocks\.([0-9]+)\.")
        mid_blocks = max([-1, *(
            int(match[1])
            for key in state_dict
            if (match := pattern.match(key)) is not None
        )]) + 1

        return HydraPool(
            attn_dim,
            head_dim,
            n_classes,
            mid_blocks=mid_blocks,
            roots=roots,
            ff_ratio=ff_ratio,
            ff_dropout=ff_dropout,
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
            dtype=dtype
        )
