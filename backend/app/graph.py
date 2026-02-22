import torch
from typing import List, Dict, Any


# ── PyTorch docs URLs ───────────────────────────────────────────────────────────

_DOCS = {
    "add":                 "https://pytorch.org/docs/stable/generated/torch.add.html",
    "sub":                 "https://pytorch.org/docs/stable/generated/torch.sub.html",
    "mul":                 "https://pytorch.org/docs/stable/generated/torch.mul.html",
    "div":                 "https://pytorch.org/docs/stable/generated/torch.div.html",
    "matmul":              "https://pytorch.org/docs/stable/generated/torch.matmul.html",
    "abs":                 "https://pytorch.org/docs/stable/generated/torch.abs.html",
    "neg":                 "https://pytorch.org/docs/stable/generated/torch.neg.html",
    "clamp":               "https://pytorch.org/docs/stable/generated/torch.clamp.html",
    "sum":                 "https://pytorch.org/docs/stable/generated/torch.sum.html",
    "reshape":             "https://pytorch.org/docs/stable/generated/torch.reshape.html",
    "transpose":           "https://pytorch.org/docs/stable/generated/torch.transpose.html",
    "flatten":             "https://pytorch.org/docs/stable/generated/torch.flatten.html",
    "squeeze":             "https://pytorch.org/docs/stable/generated/torch.squeeze.html",
    "unsqueeze":           "https://pytorch.org/docs/stable/generated/torch.unsqueeze.html",
    "permute":             "https://pytorch.org/docs/stable/generated/torch.permute.html",
    "tile":                "https://pytorch.org/docs/stable/generated/torch.tile.html",
    "repeat":              "https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html",
    "narrow":              "https://pytorch.org/docs/stable/generated/torch.narrow.html",
    "chunk":               "https://pytorch.org/docs/stable/generated/torch.chunk.html",
    "cat":                 "https://pytorch.org/docs/stable/generated/torch.cat.html",
    "stack":               "https://pytorch.org/docs/stable/generated/torch.stack.html",
    # Layer ops
    "linear":              "https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html",
    "conv1d":              "https://pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html",
    "conv2d":              "https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html",
    "maxpool1d":           "https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool1d.html",
    "maxpool2d":           "https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html",
    "avgpool1d":           "https://pytorch.org/docs/stable/generated/torch.nn.functional.avg_pool1d.html",
    "avgpool2d":           "https://pytorch.org/docs/stable/generated/torch.nn.functional.avg_pool2d.html",
    "adaptive_avgpool2d":  "https://pytorch.org/docs/stable/generated/torch.nn.functional.adaptive_avg_pool2d.html",
    "embedding":           "https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html",
    "sdpa":                "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html",
}

_BINARY_OPS  = {"add", "sub", "mul", "div", "matmul"}
_MULTI_INPUT = {"cat", "stack", "sdpa"}


# ── Op label builder ────────────────────────────────────────────────────────────

def _op_label(op: str, params: dict) -> str:
    if op == "clamp":
        return f"clamp [{params.get('min_val', '-∞')}, {params.get('max_val', '+∞')}]"
    if op == "sum" and params.get("dim") is not None:
        return f"sum (dim={params['dim']})"
    if op == "reshape" and params.get("shape"):
        return f"reshape → {params['shape']}"
    if op == "transpose":
        return f"transpose ({params.get('dim0', 0)}, {params.get('dim1', 1)})"
    if op == "flatten":
        return f"flatten ({params.get('start_dim', 0)}..{params.get('end_dim', -1)})"
    if op == "unsqueeze":
        return f"unsqueeze (dim={params.get('dim')})"
    if op == "squeeze" and params.get("dim") is not None:
        return f"squeeze (dim={params.get('dim')})"
    if op == "permute" and params.get("dims"):
        return f"permute {params['dims']}"
    if op == "tile" and params.get("dims"):
        return f"tile ×{params['dims']}"
    if op == "repeat" and params.get("sizes"):
        return f"repeat ×{params['sizes']}"
    if op == "narrow":
        return f"narrow (d={params.get('dim')}, s={params.get('start')}, l={params.get('length')})"
    if op == "chunk":
        return f"chunk (n={params.get('chunks')}, d={params.get('dim', 0)})"
    if op in {"cat", "stack"}:
        return f"{op} (dim={params.get('dim', 0)})"
    if op == "linear":
        return f"linear → {params.get('out_features')}"
    if op == "conv1d":
        return f"conv1d k={params.get('kernel_size', 3)} s={params.get('stride', 1)}"
    if op == "conv2d":
        return f"conv2d k={params.get('kernel_size', 3)} s={params.get('stride', 1)}"
    if op in {"maxpool1d", "maxpool2d", "avgpool1d", "avgpool2d"}:
        return f"{op} k={params.get('kernel_size', 2)}"
    if op == "adaptive_avgpool2d":
        return f"adaptive_avg_pool2d → {params.get('output_size')}"
    if op == "embedding":
        return f"embedding dim={params.get('embed_dim')}"
    if op == "sdpa":
        return "scaled_dot_product_attention"
    return op


# ── Apply op (mirrors operations.py, returns a single output tensor) ───────────

def _apply_op(current_t: torch.Tensor, step: Dict) -> torch.Tensor:
    import torch.nn.functional as F
    op     = step["op"]
    params = step.get("params", {})
    tb     = step.get("tensor_b")

    if op == "add":       return current_t + torch.tensor(tb, dtype=torch.float32)
    if op == "sub":       return current_t - torch.tensor(tb, dtype=torch.float32)
    if op == "mul":       return current_t * torch.tensor(tb, dtype=torch.float32)
    if op == "div":       return current_t / torch.tensor(tb, dtype=torch.float32)
    if op == "matmul":    return torch.matmul(current_t, torch.tensor(tb, dtype=torch.float32))
    if op == "abs":       return torch.abs(current_t)
    if op == "neg":       return torch.neg(current_t)
    if op == "clamp":     return torch.clamp(current_t, min=params.get("min_val"), max=params.get("max_val"))
    if op == "sum":       return current_t.sum(dim=params.get("dim"), keepdim=params.get("keepdim", False))
    if op == "reshape":   return current_t.reshape(params["shape"])
    if op == "transpose": return torch.transpose(current_t, params.get("dim0", 0), params.get("dim1", 1))
    if op == "flatten":   return torch.flatten(current_t, params.get("start_dim", 0), params.get("end_dim", -1))
    if op == "squeeze":
        dim = params.get("dim")
        if dim is None: return current_t.squeeze()
        if isinstance(dim, list):
            out = current_t
            for d in sorted(dim, reverse=True): out = out.squeeze(d)
            return out
        return current_t.squeeze(dim)
    if op == "unsqueeze": return current_t.unsqueeze(params["dim"])
    if op == "permute":   return current_t.permute(*params["dims"])
    if op == "tile":      return torch.tile(current_t, tuple(params["dims"]))
    if op == "repeat":    return current_t.repeat(*params["sizes"])
    if op == "narrow":    return torch.narrow(current_t, dim=params["dim"], start=params["start"], length=params["length"])
    if op == "chunk":     return torch.chunk(current_t, chunks=params["chunks"], dim=params.get("dim", 0))[0]
    if op == "cat":       return torch.cat([torch.tensor(x, dtype=torch.float32) for x in params["tensors"]], dim=params.get("dim", 0))
    if op == "stack":     return torch.stack([torch.tensor(x, dtype=torch.float32) for x in params["tensors"]], dim=params.get("dim", 0))
    # Layer ops
    if op == "linear":
        out_f  = params["out_features"]
        in_f   = current_t.shape[-1]
        weight = torch.nn.init.kaiming_uniform_(torch.empty(out_f, in_f))
        return F.linear(current_t, weight, torch.zeros(out_f))
    if op == "conv1d":
        oc = params["out_channels"]; k = params.get("kernel_size", 3)
        weight = torch.nn.init.kaiming_uniform_(torch.empty(oc, current_t.shape[1], k))
        return F.conv1d(current_t, weight, stride=params.get("stride", 1), padding=params.get("padding", 0))
    if op == "conv2d":
        oc = params["out_channels"]; k = params.get("kernel_size", 3)
        weight = torch.nn.init.kaiming_uniform_(torch.empty(oc, current_t.shape[1], k, k))
        return F.conv2d(current_t, weight, stride=params.get("stride", 1), padding=params.get("padding", 0))
    if op == "maxpool1d":          return F.max_pool1d(current_t, kernel_size=params.get("kernel_size", 2), stride=params.get("stride"))
    if op == "maxpool2d":          return F.max_pool2d(current_t, kernel_size=params.get("kernel_size", 2), stride=params.get("stride"))
    if op == "avgpool1d":          return F.avg_pool1d(current_t, kernel_size=params.get("kernel_size", 2), stride=params.get("stride"))
    if op == "avgpool2d":          return F.avg_pool2d(current_t, kernel_size=params.get("kernel_size", 2), stride=params.get("stride"))
    if op == "adaptive_avgpool2d": return F.adaptive_avg_pool2d(current_t, output_size=tuple(params["output_size"]))
    if op == "embedding":
        indices = current_t.long()
        vs     = params.get("vocab_size") or int(indices.max().item()) + 1
        weight = torch.randn(vs, params["embed_dim"])
        return F.embedding(indices, weight)
    if op == "sdpa":
        q = torch.tensor(params["query"], dtype=torch.float32)
        k = torch.tensor(params["key"],   dtype=torch.float32)
        v = torch.tensor(params["value"], dtype=torch.float32)
        return F.scaled_dot_product_attention(q, k, v)
    raise ValueError(f"Unsupported operation: {op}")


# ── Stats helper ─────────────────────────────────────────────────────────────────

def _tensor_stats(t: torch.Tensor) -> dict:
    flat = t.float().flatten()
    return {
        "mean":  round(float(flat.mean()), 6),
        "std":   round(float(flat.std()) if flat.numel() > 1 else 0.0, 6),
        "min":   round(float(flat.min()), 6),
        "max":   round(float(flat.max()), 6),
        "sum":   round(float(flat.sum()), 6),
        "norm":  round(float(torch.linalg.norm(flat)), 6),
        "numel": int(flat.numel()),
        "rank":  int(t.dim()),
    }


# ── Main graph builder ───────────────────────────────────────────────────────────

def generate_graph_data(original_tensor: List, operations: List[Dict]) -> Dict:
    nodes: List[Dict] = []
    edges: List[Dict] = []

    t = torch.tensor(original_tensor, dtype=torch.float32)
    tensors = [t]

    nodes.append({
        "id":    "t0",
        "type":  "tensor",
        "label": "Tensor 0",
        "shape": list(t.shape),
        "stats": _tensor_stats(t),
        "docs_url": None,
    })

    for i, step in enumerate(operations):
        op      = step["op"]
        params  = step.get("params", {})
        tb_data = step.get("tensor_b")

        # Secondary tensor for binary ops
        if op in _BINARY_OPS and tb_data is not None:
            t_b  = torch.tensor(tb_data, dtype=torch.float32)
            b_id = f"b{i}"
            nodes.append({"id": b_id, "type": "tensor_b", "label": f"Tensor B{i}",
                          "shape": list(t_b.shape), "stats": _tensor_stats(t_b), "docs_url": None})

        # Extra tensors for cat / stack
        extra_ids = []
        if op in {"cat", "stack"} and params.get("tensors"):
            for j, extra in enumerate(params["tensors"][1:], start=1):
                t_extra  = torch.tensor(extra, dtype=torch.float32)
                extra_id = f"cat_b{i}_{j}"
                nodes.append({"id": extra_id, "type": "tensor_b", "label": f"Tensor B{i}_{j}",
                              "shape": list(t_extra.shape), "stats": _tensor_stats(t_extra), "docs_url": None})
                extra_ids.append(extra_id)

        # K / V tensors for SDPA
        sdpa_ids = []
        if op == "sdpa":
            for role, key in [("K", "key"), ("V", "value")]:
                if params.get(key):
                    t_in    = torch.tensor(params[key], dtype=torch.float32)
                    node_id = f"sdpa_{key}{i}"
                    nodes.append({"id": node_id, "type": "tensor_b", "label": f"{role}{i}",
                                  "shape": list(t_in.shape), "stats": _tensor_stats(t_in), "docs_url": None})
                    sdpa_ids.append(node_id)

        # Op node
        op_id = f"op{i}"
        nodes.append({"id": op_id, "type": "op", "label": _op_label(op, params),
                      "shape": None, "stats": None, "docs_url": _DOCS.get(op)})

        # Edges into op
        edges.append({"id": f"e{i}_in",  "source": f"t{i}",  "target": op_id})
        if op in _BINARY_OPS and tb_data is not None:
            edges.append({"id": f"e{i}_b", "source": f"b{i}", "target": op_id})
        for extra_id in extra_ids:
            edges.append({"id": f"e{i}_{extra_id}", "source": extra_id, "target": op_id})
        for sdpa_id in sdpa_ids:
            edges.append({"id": f"e{i}_{sdpa_id}", "source": sdpa_id, "target": op_id})

        # Compute output
        output_t = _apply_op(tensors[-1], step)
        tensors.append(output_t)

        # Output tensor node
        out_id = f"t{i + 1}"
        nodes.append({"id": out_id, "type": "tensor", "label": f"Tensor {i + 1}",
                      "shape": list(output_t.shape), "stats": _tensor_stats(output_t), "docs_url": None})
        edges.append({"id": f"e{i}_out", "source": op_id, "target": out_id})

    return {"nodes": nodes, "edges": edges}
