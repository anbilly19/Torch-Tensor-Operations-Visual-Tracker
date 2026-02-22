import torch
from typing import List, Dict, Any
from .operations import tensor_stats


# ── PyTorch docs URLs ───────────────────────────────────────────────────────────

_DOCS = {
    "add":       "https://pytorch.org/docs/stable/generated/torch.add.html",
    "sub":       "https://pytorch.org/docs/stable/generated/torch.sub.html",
    "mul":       "https://pytorch.org/docs/stable/generated/torch.mul.html",
    "div":       "https://pytorch.org/docs/stable/generated/torch.div.html",
    "matmul":    "https://pytorch.org/docs/stable/generated/torch.matmul.html",
    "abs":       "https://pytorch.org/docs/stable/generated/torch.abs.html",
    "neg":       "https://pytorch.org/docs/stable/generated/torch.neg.html",
    "clamp":     "https://pytorch.org/docs/stable/generated/torch.clamp.html",
    "sum":       "https://pytorch.org/docs/stable/generated/torch.sum.html",
    "reshape":   "https://pytorch.org/docs/stable/generated/torch.reshape.html",
    "transpose": "https://pytorch.org/docs/stable/generated/torch.transpose.html",
    "flatten":   "https://pytorch.org/docs/stable/generated/torch.flatten.html",
    "squeeze":   "https://pytorch.org/docs/stable/generated/torch.squeeze.html",
    "unsqueeze": "https://pytorch.org/docs/stable/generated/torch.unsqueeze.html",
    "permute":   "https://pytorch.org/docs/stable/generated/torch.permute.html",
    "tile":      "https://pytorch.org/docs/stable/generated/torch.tile.html",
    "repeat":    "https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html",
    "narrow":    "https://pytorch.org/docs/stable/generated/torch.narrow.html",
    "chunk":     "https://pytorch.org/docs/stable/generated/torch.chunk.html",
    "cat":       "https://pytorch.org/docs/stable/generated/torch.cat.html",
    "stack":     "https://pytorch.org/docs/stable/generated/torch.stack.html",
}

_BINARY_OPS = {"add", "sub", "mul", "div", "matmul"}


# ── Op label builder ────────────────────────────────────────────────────────────

def _op_label(op: str, params: dict) -> str:
    if op == "clamp":
        lo = params.get("min_val", "-∞")
        hi = params.get("max_val", "+∞")
        return f"clamp [{lo}, {hi}]"
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
    return op


# ── Apply op ───────────────────────────────────────────────────────────────────

def _apply_op(current_t: torch.Tensor, step: Dict) -> torch.Tensor:
    op = step["op"]
    params = step.get("params", {})
    tensor_b_data = step.get("tensor_b")

    if op == "add":       return current_t + torch.tensor(tensor_b_data, dtype=torch.float32)
    if op == "sub":       return current_t - torch.tensor(tensor_b_data, dtype=torch.float32)
    if op == "mul":       return current_t * torch.tensor(tensor_b_data, dtype=torch.float32)
    if op == "div":       return current_t / torch.tensor(tensor_b_data, dtype=torch.float32)
    if op == "matmul":    return torch.matmul(current_t, torch.tensor(tensor_b_data, dtype=torch.float32))
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
    raise ValueError(f"Unsupported operation: {op}")


# ── Stats helper ─────────────────────────────────────────────────────────────────

def _tensor_stats(t: torch.Tensor) -> dict:
    flat = t.flatten()
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
    """
    Build graph nodes + edges data for the frontend React Flow canvas.
    Returns { nodes: [...], edges: [...] }.
    """
    nodes: List[Dict] = []
    edges: List[Dict] = []

    t = torch.tensor(original_tensor, dtype=torch.float32)
    tensors = [t]

    # Root tensor node
    nodes.append({
        "id":    "t0",
        "type":  "tensor",
        "label": "Tensor 0",
        "shape": list(t.shape),
        "stats": _tensor_stats(t),
        "docs_url": None,
    })

    for i, step in enumerate(operations):
        op        = step["op"]
        params    = step.get("params", {})
        tb_data   = step.get("tensor_b")
        prev_node = f"t{i}"

        # Secondary tensor node for binary ops
        if op in _BINARY_OPS and tb_data is not None:
            t_b = torch.tensor(tb_data, dtype=torch.float32)
            b_id = f"b{i}"
            nodes.append({
                "id":    b_id,
                "type":  "tensor_b",
                "label": f"Tensor B{i}",
                "shape": list(t_b.shape),
                "stats": _tensor_stats(t_b),
                "docs_url": None,
            })

        # Extra tensor nodes for cat / stack
        extra_ids = []
        if op in {"cat", "stack"} and params.get("tensors"):
            for j, extra in enumerate(params["tensors"][1:], start=1):
                t_extra = torch.tensor(extra, dtype=torch.float32)
                extra_id = f"cat_b{i}_{j}"
                nodes.append({
                    "id":    extra_id,
                    "type":  "tensor_b",
                    "label": f"Tensor B{i}_{j}",
                    "shape": list(t_extra.shape),
                    "stats": _tensor_stats(t_extra),
                    "docs_url": None,
                })
                extra_ids.append(extra_id)

        # Op node
        op_id = f"op{i}"
        nodes.append({
            "id":      op_id,
            "type":    "op",
            "label":   _op_label(op, params),
            "shape":   None,
            "stats":   None,
            "docs_url": _DOCS.get(op),
        })

        # Edges into op
        edges.append({"id": f"e{i}_in",  "source": prev_node, "target": op_id})
        if op in _BINARY_OPS and tb_data is not None:
            edges.append({"id": f"e{i}_b",  "source": f"b{i}",   "target": op_id})
        for extra_id in extra_ids:
            edges.append({"id": f"e{i}_{extra_id}", "source": extra_id, "target": op_id})

        # Compute output tensor
        output_t = _apply_op(tensors[-1], step)
        tensors.append(output_t)

        # Output tensor node
        out_id = f"t{i + 1}"
        nodes.append({
            "id":    out_id,
            "type":  "tensor",
            "label": f"Tensor {i + 1}",
            "shape": list(output_t.shape),
            "stats": _tensor_stats(output_t),
            "docs_url": None,
        })
        edges.append({"id": f"e{i}_out", "source": op_id, "target": out_id})

    return {"nodes": nodes, "edges": edges}
