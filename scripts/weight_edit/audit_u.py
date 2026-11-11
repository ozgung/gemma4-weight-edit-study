"""Audit of Fable's "unused head-space direction" edit (fable_p10_1, final E6).  Owner: Claude.

    outputs/small_model/mlx_venv/bin/python -I scripts/weight_edit/audit_u.py [--n-calls 8] [--out OUT.json]

Fable's claims (final_answer.md):
 C1 u lies in head dimensions that RoPE does not rotate (global layers rotate only part of each 512-dim head).
 C2 u is the direction with the smallest second moment under natural normalised queries/keys of the layer.
 C3 the q/k norm gains are uniform per layer (so the edit's attention-logit effect is predictable).
 C4 interference with natural attention is about 0.2 nats, while (old_string query, thought-quote / new_string key)
    pairs shift by several nats.
Checks, on search calls Fable never used in a read pass (natural prompt + prefix + recorded old_string):
 C1: apply each edited layer's RoPE to basis vectors at two offsets; rotated dims are those that change; report the
     share of ||u||^2 on rotated dims, per edited module and head.
 C2: second moment of u under normalised q and k (all positions, edited heads) against the eigenvalue spectrum of
     the same covariance restricted to the unrotated dims: rank, ratio to the smallest and to the median eigenvalue.
 C3: coefficient of variation of q_norm / k_norm weights (if present).
 C4: from the base hidden state entering each edited layer, attention scores (and softmax) for base vs edited
     q_proj/k_proj on the same inputs (the layer's own effect, not the downstream cascade). |delta score| and total-
     variation change of the attention row, for query positions in the prompt (natural) and in old_string.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/weight_edit"))
sys.path.insert(0, str(ROOT / "scripts/mlx_lora"))
RUN = ROOT / "outputs/weight_edit/proposers/fable_p10_1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calls", type=int, default=8)
    ap.add_argument("--n-queries", type=int, default=192)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/weight_edit/claude/native_check/audit_u_fable.json")
    a = ap.parse_args()
    import mlx.core as mx
    import wsp_server as S
    from common import text_model

    srv = S.Server()
    tm = text_model(srv.model)
    # edits
    edits = {}
    for f in sorted((RUN / "edit").glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        name = str(z["module"])
        l = int(name.split(".layers.")[1].split(".")[0])
        kind = "q" if "q_proj" in name else "k"
        edits.setdefault(l, {})[kind] = (z["B"].astype(np.float32), z["A"].astype(np.float32))
    layers = sorted(edits)
    used = set()
    for k in range(1, 11):
        q = json.loads((RUN / "requests" / f"fwd_{k}.json.claimed").read_text())
        used |= {t["key"] for t in q.get("texts", [])} | set(q.get("calls", []))
    cands = sorted(k for k, r in srv.calls.items() if k not in used and r["old_string"].count("\n") >= 2)
    rng = np.random.default_rng(20261103)
    pick = [cands[i] for i in rng.choice(len(cands), size=min(a.n_calls, len(cands)), replace=False)]
    rep = {"layers": layers, "calls": pick, "C1": {}, "C2": {}, "C3": {}, "C4": {}}

    for l in layers:
        att = tm.layers[l].self_attn
        hd, nh, nkv = att.head_dim, att.n_heads, att.n_kv_heads
        # C1: rotated dims
        basis = mx.eye(hd).reshape(1, hd, 1, hd).transpose(0, 2, 1, 3)  # [1,1,L=hd,hd]
        r0 = np.array(att.rope(basis, offset=0)[0, 0])
        r1 = np.array(att.rope(basis, offset=100000)[0, 0])
        # position i holds basis vector e_i at offset+i; compare against the same vector rotated at a different offset
        rot = np.abs(r0 - r1).max(axis=1) > 1e-4
        Bq, Aq = edits[l]["q"]
        Bk, Ak = edits[l]["k"]
        uq = {h: Bq[h * hd:(h + 1) * hd, 0] for h in range(nh) if np.abs(Bq[h * hd:(h + 1) * hd, 0]).sum() > 0}
        uk = {h: Bk[h * hd:(h + 1) * hd, 0] for h in range(nkv) if np.abs(Bk[h * hd:(h + 1) * hd, 0]).sum() > 0}
        ref = next(iter(uq.values()))
        ref = ref / np.linalg.norm(ref)
        rep["C1"][l] = {"rotated_dims": int(rot.sum()), "head_dim": hd,
                        "q_heads": {h: float((v[rot] ** 2).sum() / (v ** 2).sum()) for h, v in uq.items()},
                        "k_heads": {h: float((v[rot] ** 2).sum() / (v ** 2).sum()) for h, v in uk.items()},
                        "cos_q_vs_k_dirs": {f"q{hq}-k{hk}": float(abs(vq @ vk) / np.linalg.norm(vq) / np.linalg.norm(vk))
                                            for hq, vq in uq.items() for hk, vk in uk.items()}}
        gq = np.array(att.q_norm.weight.astype(mx.float32)) if hasattr(att.q_norm, "weight") else None
        gk = np.array(att.k_norm.weight.astype(mx.float32)) if hasattr(att.k_norm, "weight") else None
        rep["C3"][l] = {"q_norm_cv": float(gq.std() / abs(gq.mean())) if gq is not None else None,
                        "k_norm_cv": float(gk.std() / abs(gk.mean())) if gk is not None else None,
                        "q_norm_present": gq is not None, "k_norm_present": gk is not None}
        rep["C2"][l] = {"q": [], "k": []}
        rep["C4"][l] = {"natural": [], "old_string": []}
        edits[l]["rot"] = rot
        edits[l]["u"] = ref

    acc = {l: {"Sq": 0, "Sk": 0, "nq": 0, "nk": 0} for l in layers}
    for key in pick:
        r = srv.calls[key]
        p_ids = srv.tok.encode(r["prompt"] + r["prefix"], add_special_tokens=False).ids
        c_ids = srv.tok.encode(r["old_string"], add_special_tokens=False).ids[:256]
        ids = p_ids + c_ids
        L = len(ids)
        caps, _ = srv.forward_capture(ids, list(range(L)), {l - 1 for l in layers}, {"resid"}, set())
        for li, l in enumerate(layers):
            att = tm.layers[l].self_attn
            hd, nh, nkv = att.head_dim, att.n_heads, att.n_kv_heads
            h = mx.array(caps["resid"][li])[None]  # residual after layer l-1 = input of layer l, [1, L, D]
            xa = tm.layers[l].input_layernorm(h)
            Bq, Aq = edits[l]["q"]
            Bk, Ak = edits[l]["k"]
            qb = att.q_proj(xa)
            kb = att.k_proj(xa)
            xf = xa.astype(mx.float32)
            qe = qb.astype(mx.float32) + (xf @ mx.array(Aq).T) @ mx.array(Bq).T
            ke = kb.astype(mx.float32) + (xf @ mx.array(Ak).T) @ mx.array(Bk).T

            def prep(q, k):
                q = att.q_norm(q.astype(qb.dtype).reshape(1, L, nh, hd))
                k = att.k_norm(k.astype(kb.dtype).reshape(1, L, nkv, hd))
                return q, k

            qn_b, kn_b = prep(qb, kb)
            qn_e, kn_e = prep(qe, ke)
            # C2: second moments in the unrotated subspace, edited heads only
            rot = edits[l]["rot"]
            un = np.where(~rot)[0]
            qh = [h for h in range(nh) if np.abs(Bq[h * hd:(h + 1) * hd, 0]).sum() > 0]
            kh = [h for h in range(nkv) if np.abs(Bk[h * hd:(h + 1) * hd, 0]).sum() > 0]
            Q = np.array(qn_b[0][:, qh].astype(mx.float32)).reshape(-1, hd)[:, un]
            K = np.array(kn_b[0][:, kh].astype(mx.float32)).reshape(-1, hd)[:, un]
            acc[l]["Sq"] = acc[l]["Sq"] + Q.T @ Q
            acc[l]["Sk"] = acc[l]["Sk"] + K.T @ K
            acc[l]["nq"] += len(Q)
            acc[l]["nk"] += len(K)
            # C4: attention scores base vs edit for sampled natural queries and all old_string queries
            qpos = sorted(set(np.linspace(64, len(p_ids) - 2, a.n_queries).astype(int).tolist()))
            opos = list(range(len(p_ids) - 1, L))

            def scores(qn, kn, pos):
                q = att.rope(qn.transpose(0, 2, 1, 3), offset=0)[:, :, mx.array(pos), :]
                k = att.rope(kn.transpose(0, 2, 1, 3), offset=0)
                k = mx.repeat(k, nh // nkv, axis=1)
                s = (q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * att.scale
                j = mx.arange(L)[None, :]
                mask = j <= mx.array(pos)[:, None]
                return mx.where(mask[None, None], s, -1e30)[0]  # [nh, P, L], all query heads

            other = [h for h in range(nh) if h not in qh]
            for kind, pos in (("natural", qpos), ("old_string", opos)):
                sb, se = scores(qn_b, kn_b, pos), scores(qn_e, kn_e, pos)
                pb, pe = mx.softmax(sb, axis=-1), mx.softmax(se, axis=-1)
                valid1 = np.array(mx.arange(L)[None, :] <= mx.array(pos)[:, None])
                rec = {"key": key, "n_queries": len(pos)}
                for grp, hs in (("edited_heads", qh), ("other_heads", other)):
                    if not hs:
                        continue
                    hi = mx.array(hs)
                    ds = np.abs(np.array(se[hi] - sb[hi]))[np.broadcast_to(valid1, (len(hs),) + valid1.shape)]
                    tv = np.array(0.5 * mx.abs(pe[hi] - pb[hi]).sum(-1)).ravel()
                    rec[grp] = {"heads": hs, "abs_dscore_p50": float(np.percentile(ds, 50)),
                                "abs_dscore_p99": float(np.percentile(ds, 99)), "abs_dscore_max": float(ds.max()),
                                "tv_mean": float(tv.mean()), "tv_p90": float(np.percentile(tv, 90)), "tv_max": float(tv.max())}
                rep["C4"][l][kind].append(rec)
            mx.eval(qn_e)
        print(f"audited {key} ({L} tokens)", flush=True)

    for l in layers:
        u = edits[l]["u"][~edits[l]["rot"]]
        u = u / np.linalg.norm(u)
        for kind in ("q", "k"):
            S_ = acc[l][f"S{kind}"] / acc[l][f"n{kind}"]
            ev = np.linalg.eigvalsh(S_)
            m = float(u @ S_ @ u)
            rep["C2"][l][kind] = {"u_second_moment": m, "eig_min": float(ev[0]), "eig_median": float(np.median(ev)),
                                  "eig_max": float(ev[-1]), "rank_of_u_among_eigs": int((ev < m).sum()),
                                  "n_dims": len(ev), "ratio_to_min": m / float(ev[0]) if ev[0] > 0 else None,
                                  "ratio_to_median": m / float(np.median(ev)), "n_vectors": acc[l][f"n{kind}"]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({"C1": {l: rep["C1"][l]["q_heads"] for l in layers}, "C2": rep["C2"], "C3": rep["C3"]}, default=str)[:3000])
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
