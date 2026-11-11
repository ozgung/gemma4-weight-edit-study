"""PyTorch/CUDA port of the forward-pass server's model operations (for Colab GPUs).  Owner: Claude.

Mirrors wsp_server.Server: generate() (prefill once, copy the KV cache per draw; draw 0 greedy, draws 1.. sampled at
T 0.6 / top_p 0.95 / top_k 64; stop at the <|"|> token), native_logprobs() (cached forward, log-softmax at contiguous
positions), capture() (residual stream after layers and MLP activations entering down_proj, via hooks on a full
forward), apply()/restore() (W' = W + B @ A, fp32 add, cast back). Attention-probability capture is NOT ported yet.

Sampling uses torch's RNG, so sampled draws are not token-identical to the MLX server's (greedy should match up to
backend numerics). A port is a new instrument: validate it (bench_torch.py) before using it for any comparison, and
never mix its generations with MLX generations in one paired analysis.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch

Q = '<|"|>'


class TorchBackend:
    def __init__(self, model_dir: str | Path, device: str = "cuda", attn: str = "sdpa"):
        from tokenizers import Tokenizer
        from transformers import Gemma4ForConditionalGeneration

        self.device = device
        self.tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.q_id = self.tok.token_to_id(Q)
        t = time.time()
        # the audio tower has no SDPA path; only the text model's attention matters here
        impl = {"text_config": attn, "vision_config": "eager", "audio_config": "eager"}
        self.model = Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16,
                                                                    attn_implementation=impl).to(device).eval()
        self.load_s = time.time() - t
        for tower in ("audio_tower", "vision_tower", "embed_audio", "embed_vision"):  # unused; frees GPU memory
            if hasattr(self.model.model, tower):
                setattr(self.model.model, tower, None)
        torch.cuda.empty_cache() if device == "cuda" else None
        self.lm = self.model.model.language_model
        self.params = dict(self.model.named_parameters())

    # ---- edits ---------------------------------------------------------------------------------------------
    def _param(self, module_name: str) -> torch.nn.Parameter:
        # MLX/kernel names: model.language_model.layers.L.<...>.weight
        tail = module_name.removeprefix("model.language_model.").removesuffix(".weight")
        for cand in (f"model.language_model.{tail}.weight", f"language_model.model.{tail}.weight"):
            if cand in self.params:
                return self.params[cand]
        raise KeyError(module_name)

    @torch.no_grad()
    def apply(self, edit_dir: Path) -> dict:
        originals = {}
        for f in sorted(Path(edit_dir).glob("*.npz")):
            z = np.load(f, allow_pickle=False)
            p = self._param(str(z["module"]))
            D = torch.from_numpy(z["B"].astype(np.float32)).to(self.device) @ torch.from_numpy(z["A"].astype(np.float32)).to(self.device)
            assert D.shape == p.shape, (str(z["module"]), D.shape, p.shape)
            originals[str(z["module"])] = p.data.clone()
            p.data.copy_((p.data.float() + D).to(p.dtype))
        return originals

    @torch.no_grad()
    def restore(self, originals: dict) -> None:
        for name, w in originals.items():
            self._param(name).data.copy_(w)

    # ---- generation ----------------------------------------------------------------------------------------
    @torch.no_grad()
    def _prefill(self, ids: list[int]):
        x = torch.tensor([ids], device=self.device)
        out = self.model(input_ids=x, use_cache=True, logits_to_keep=1)  # do not materialise [L, 262k] logits
        return out.past_key_values

    @torch.no_grad()
    def generate(self, ids: list[int], max_new: int, draws: int, seeds: list[int]) -> list[dict]:
        t0 = time.time()
        cache = self._prefill(ids[:-1])
        torch.cuda.synchronize() if self.device == "cuda" else None
        prefill_s = time.time() - t0
        outs = []
        for d in range(draws):
            t1 = time.time()
            c = copy.deepcopy(cache)
            g = torch.Generator(device=self.device).manual_seed(int(seeds[d]))
            cur, gen, closed = ids[-1], [], 0
            pos = len(ids) - 1
            for _ in range(max_new):
                out = self.model(input_ids=torch.tensor([[cur]], device=self.device), past_key_values=c, use_cache=True,
                                 cache_position=torch.tensor([pos], device=self.device))
                c, pos = out.past_key_values, pos + 1
                logits = out.logits[0, -1].float()
                if d == 0:
                    nxt = int(logits.argmax())
                else:
                    nxt = self._sample(logits, g)
                if nxt == self.q_id:
                    closed = 1
                    break
                gen.append(nxt)
                cur = nxt
            outs.append({"draw": d, "gen": self.tok.decode(gen, skip_special_tokens=False), "closed": closed,
                         "n_gen": len(gen), "gen_s": round(time.time() - t1, 2)})
        for o in outs:
            o["prefill_s"] = round(prefill_s, 2)
        return outs

    @staticmethod
    def _sample(logits: torch.Tensor, g: torch.Generator, temp=0.6, top_k=64, top_p=0.95) -> int:
        logits = logits / temp
        v, i = torch.topk(logits, top_k)
        p = torch.softmax(v, -1)
        cp = torch.cumsum(p, -1)
        keep = cp - p < top_p
        p = torch.where(keep, p, torch.zeros_like(p))
        p = p / p.sum()
        return int(i[torch.multinomial(p, 1, generator=g)])

    # ---- teacher-forced -----------------------------------------------------------------------------------
    @torch.no_grad()
    def native_logprobs(self, ids: list[int], positions: list[int]) -> np.ndarray:
        p0, n = positions[0], len(positions)
        assert positions == list(range(p0, p0 + n))
        cache = self._prefill(ids[:p0])
        out = self.model(input_ids=torch.tensor([ids[p0:p0 + n]], device=self.device), past_key_values=cache,
                         use_cache=True, cache_position=torch.arange(p0, p0 + n, device=self.device))
        lg = out.logits[0].float()
        return (lg - torch.logsumexp(lg, -1, keepdim=True)).cpu().numpy()

    @torch.no_grad()
    def capture(self, ids: list[int], positions: list[int], layers: set[int], capture: set[str]) -> dict:
        """Residual stream after each requested decoder layer and MLP activations entering down_proj, at positions."""
        out, hooks = {"resid": {}, "mlp": {}}, []
        pos = torch.tensor(positions, device=self.device)
        for i, layer in enumerate(self.lm.layers):
            if i not in layers:
                continue
            if "resid" in capture:
                hooks.append(layer.register_forward_hook(
                    lambda m, a, o, i=i: out["resid"].__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0, pos].float().cpu().numpy())))
            if "mlp" in capture:
                hooks.append(layer.mlp.down_proj.register_forward_pre_hook(
                    lambda m, a, i=i: out["mlp"].__setitem__(i, a[0][0, pos].half().cpu().numpy())))
        try:
            self.model(input_ids=torch.tensor([ids], device=self.device), use_cache=False, logits_to_keep=1)
        finally:
            for h in hooks:
                h.remove()
        return {k: np.stack([v[i] for i in sorted(v)]) for k, v in out.items() if v}
