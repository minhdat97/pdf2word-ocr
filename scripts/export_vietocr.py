"""Make models/vietocr from VietOCR's vgg_transformer weights (Apache-2.0).

Not needed to run the tool: scripts/fetch_models.py downloads these files (the
release named in scripts/models.json). To use another export, copy it to the server
with scripts/fetch_models.py --from DIR after updating the manifest's SHA-256 (a
running service finds it). Needs torch, vietocr, pyyaml, onnx, onnxscript and
onnxruntime:

    python scripts/export_vietocr.py vgg_transformer.pth base.yml vgg-transformer.yml models/vietocr

Writes encoder.onnx (image [N,3,32,W] -> cross-attention keys/values of the
decoder), decoder.onnx (one greedy step with a key/value cache) and config.json.
Weights are stored as float16 and cast to float32 when loaded (half the size;
computation stays float32), stripped of the exporter's metadata (local paths),
then checked against vietocr's own decoder.
"""

import json
import math
import os
import sys

import numpy as np
import onnx
import torch
import torch.nn.functional as F
import yaml
from onnx import TensorProto, helper, numpy_helper
from torch import nn
from vietocr.tool.config import Cfg
from vietocr.tool.translate import build_model

MAX_STEPS = 128  # vietocr's translate() default


def load(weights: str, base_yml: str, model_yml: str):
    with open(base_yml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(model_yml, encoding="utf-8") as f:
        cfg.update(yaml.safe_load(f))
    cfg["device"] = "cpu"
    cfg["cnn"]["pretrained"] = False
    model, _ = build_model(Cfg(cfg))
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.eval(), cfg


def modules(model):
    tr = model.transformer
    E = tr.d_model
    layers = tr.transformer.decoder.layers
    H = layers[0].self_attn.num_heads
    HD = E // H

    def attend(q, k, v):
        return torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(HD), dim=-1) @ v

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn, self.tr = model.cnn, tr
            # key/value rows of the decoder's cross-attention projections
            self.kv_w = nn.ParameterList([nn.Parameter(ly.multihead_attn.in_proj_weight[E:].clone()) for ly in layers])
            self.kv_b = nn.ParameterList([nn.Parameter(ly.multihead_attn.in_proj_bias[E:].clone()) for ly in layers])

        def forward(self, img):
            vgg = self.cnn.model  # Vgg.forward, with permute axes ONNX accepts
            conv = vgg.last_conv_1x1(vgg.features(img))  # N, C, H, W
            src = conv.transpose(2, 3).flatten(2).permute(2, 0, 1)  # W*H, N, C
            mem = self.tr.forward_encoder(src)  # S, N, E
            s, n = mem.shape[0], mem.shape[1]
            ks, vs = [], []
            for w, b in zip(self.kv_w, self.kv_b):
                k, v = F.linear(mem, w, b).split(E, dim=-1)
                ks.append(k.reshape(s, n, H, HD).permute(1, 2, 0, 3))
                vs.append(v.reshape(s, n, H, HD).permute(1, 2, 0, 3))
            return torch.stack(ks), torch.stack(vs)

    class Decoder(nn.Module):
        """One step of nn.TransformerDecoder (post-norm) for the last position only.
        past_k/past_v: [L, N, H, P, HD], slot 0 a placeholder that attention skips."""

        def __init__(self):
            super().__init__()
            self.tr = tr
            self.pe = nn.Parameter(tr.pos_enc.pe[: MAX_STEPS + 2, 0].clone())
            self.q_w = nn.ParameterList([nn.Parameter(ly.multihead_attn.in_proj_weight[:E].clone()) for ly in layers])
            self.q_b = nn.ParameterList([nn.Parameter(ly.multihead_attn.in_proj_bias[:E].clone()) for ly in layers])

        def forward(self, token, pos, past_k, past_v, cross_k, cross_v):
            n = token.shape[0]
            x = self.tr.embed_tgt(token) * math.sqrt(E) + self.pe[pos]
            new_k, new_v = [], []
            for i, ly in enumerate(self.tr.transformer.decoder.layers):
                sa, ca = ly.self_attn, ly.multihead_attn
                q, k, v = F.linear(x, sa.in_proj_weight, sa.in_proj_bias).split(E, dim=-1)
                kk = torch.cat([past_k[i], k.reshape(n, H, 1, HD)], dim=2)
                vv = torch.cat([past_v[i], v.reshape(n, H, 1, HD)], dim=2)
                new_k.append(kk)
                new_v.append(vv)
                a = attend(q.reshape(n, H, 1, HD), kk[:, :, 1:], vv[:, :, 1:]).reshape(n, E)
                x = ly.norm1(x + sa.out_proj(a))
                q2 = F.linear(x, self.q_w[i], self.q_b[i]).reshape(n, H, 1, HD)
                x = ly.norm2(x + ca.out_proj(attend(q2, cross_k[i], cross_v[i]).reshape(n, E)))
                x = ly.norm3(x + ly.linear2(F.relu(ly.linear1(x))))
            x = self.tr.transformer.decoder.norm(x)
            return torch.softmax(self.tr.fc(x), dim=-1), torch.stack(new_k), torch.stack(new_v)

    return Encoder(), Decoder(), (len(layers), H, HD)


def finish(path: str) -> None:
    """Store float32 weights as float16 followed by a Cast (folded to float32 when loaded),
    and drop the exporter's per-node metadata (stack traces with local file paths)."""
    m = onnx.load(path)
    g = m.graph
    for node in g.node:
        del node.metadata_props[:]
        node.doc_string = ""
    inits, casts = [], []
    for t in g.initializer:
        if t.data_type == TensorProto.FLOAT and int(np.prod(t.dims)) >= 16:
            inits.append(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float16), t.name + "__fp16"))
            cast = helper.make_node("Cast", [t.name + "__fp16"], [t.name], to=TensorProto.FLOAT, name=t.name + "__cast")
            casts.append(cast)
        else:
            inits.append(t)
    del g.initializer[:]
    g.initializer.extend(inits)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.checker.check_model(m)
    onnx.save(m, path)


def main() -> None:
    weights, base_yml, model_yml, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    model, cfg = load(weights, base_yml, model_yml)
    encoder, decoder, (L, H, HD) = modules(model)
    with torch.no_grad():
        img = torch.rand(3, 3, 32, 120)
        n = torch.export.Dim("n", min=1, max=4096)
        torch.onnx.export(
            encoder,
            (img,),
            f"{out}/encoder.onnx",
            dynamo=True,
            opset_version=18,
            external_data=False,
            input_names=["image"],
            output_names=["cross_k", "cross_v"],
            dynamic_shapes=({0: n, 3: torch.export.Dim("w", min=32, max=1024)},),
        )
        ck, cv = encoder(img)
        s, p = torch.export.Dim("s", min=1, max=4096), torch.export.Dim("p", min=1, max=MAX_STEPS + 2)
        past_k, past_v = torch.rand(L, 3, H, 4, HD), torch.rand(L, 3, H, 4, HD)  # distinct tensors (no aliasing)
        torch.onnx.export(
            decoder,
            (torch.tensor([1, 1, 1]), torch.tensor(3), past_k, past_v, ck, cv),
            f"{out}/decoder.onnx",
            dynamo=True,
            opset_version=18,
            external_data=False,
            input_names=["token", "pos", "past_k", "past_v", "cross_k", "cross_v"],
            output_names=["probs", "new_k", "new_v"],
            dynamic_shapes=({0: n}, None, {1: n, 3: p}, {1: n, 3: p}, {1: n, 3: s}, {1: n, 3: s}),
        )
    for name in ("encoder.onnx", "decoder.onnx"):
        finish(f"{out}/{name}")
    ds = cfg["dataset"]
    config = {
        "vocab": cfg["vocab"],
        "image_height": ds["image_height"],
        "image_min_width": ds["image_min_width"],
        "image_max_width": ds["image_max_width"],
        "max_seq_length": MAX_STEPS,
        "layers": L,
        "heads": H,
        "head_dim": HD,
    }
    with open(f"{out}/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=1)

    # check: cached greedy steps against vietocr's full-prefix decoder
    import onnxruntime as ort

    enc = ort.InferenceSession(f"{out}/encoder.onnx", providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(f"{out}/decoder.onnx", providers=["CPUExecutionProvider"])
    tr = model.transformer
    for n_, w in ((1, 32), (4, 250)):
        x = torch.rand(n_, 3, 32, w)
        with torch.no_grad():
            mem = tr.forward_encoder(model.cnn(x))
        ck, cv = enc.run(None, {"image": x.numpy()})
        pk = np.zeros((L, n_, H, 1, HD), np.float32)
        pv = pk.copy()
        toks = torch.ones((1, n_), dtype=torch.long)
        worst = 0.0
        for t in range(5):
            with torch.no_grad():
                ref = torch.softmax(tr.forward_decoder(toks, mem)[0], -1)[:, -1].numpy()
            feed = {
                "token": toks[-1].numpy(),
                "pos": np.array(t, np.int64),
                "past_k": pk,
                "past_v": pv,
                "cross_k": ck,
                "cross_v": cv,
            }
            got, pk, pv = dec.run(None, feed)
            worst = max(worst, float(np.abs(ref - got).max()))
            toks = torch.cat([toks, torch.randint(4, 200, (1, n_))])
        print(f"batch {n_} width {w}: largest probability difference {worst:.1e} (float16 weights)")


if __name__ == "__main__":
    main()
