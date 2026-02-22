import torch
import graphviz
import base64
from typing import List, Dict


def _apply_op(current_t: torch.Tensor, step: Dict) -> torch.Tensor:
    """Apply a single operation step to current_t and return the result tensor."""
    op = step["op"]
    params = step.get("params", {})
    tensor_b_data = step.get("tensor_b")

    # ── Binary ops ─────────────────────────────────────────────────────────────
    if op == "add":
        return current_t + torch.tensor(tensor_b_data, dtype=torch.float32)
    elif op == "sub":
        return current_t - torch.tensor(tensor_b_data, dtype=torch.float32)
    elif op == "mul":
        return current_t * torch.tensor(tensor_b_data, dtype=torch.float32)
    elif op == "div":
        return current_t / torch.tensor(tensor_b_data, dtype=torch.float32)
    elif op == "matmul":
        return torch.matmul(current_t, torch.tensor(tensor_b_data, dtype=torch.float32))

    # ── Unary ops ──────────────────────────────────────────────────────────────
    elif op == "abs":
        return torch.abs(current_t)
    elif op == "neg":
        return torch.neg(current_t)
    elif op == "clamp":
        return torch.clamp(current_t, min=params.get("min_val"), max=params.get("max_val"))

    # ── Reduction ──────────────────────────────────────────────────────────────
    elif op == "sum":
        return current_t.sum(dim=params.get("dim"), keepdim=params.get("keepdim", False))

    # ── Shape ops ──────────────────────────────────────────────────────────────
    elif op == "reshape":
        return current_t.reshape(params["shape"])
    elif op == "transpose":
        return torch.transpose(current_t, params.get("dim0", 0), params.get("dim1", 1))
    elif op == "flatten":
        return torch.flatten(current_t, start_dim=params.get("start_dim", 0), end_dim=params.get("end_dim", -1))
    elif op == "squeeze":
        dim = params.get("dim")
        if dim is None:
            return current_t.squeeze()
        if isinstance(dim, list):
            out = current_t
            for d in sorted(dim, reverse=True):
                out = out.squeeze(d)
            return out
        return current_t.squeeze(dim)
    elif op == "unsqueeze":
        return current_t.unsqueeze(params["dim"])
    elif op == "permute":
        return current_t.permute(*params["dims"])
    elif op == "tile":
        return torch.tile(current_t, tuple(params["dims"]))
    elif op == "repeat":
        return current_t.repeat(*params["sizes"])
    elif op == "narrow":
        return torch.narrow(current_t, dim=params["dim"], start=params["start"], length=params["length"])
    elif op == "chunk":
        # Graph follows the first chunk only to maintain a single-path pipeline
        chunks = torch.chunk(current_t, chunks=params["chunks"], dim=params.get("dim", 0))
        return chunks[0]
    elif op == "cat":
        # params["tensors"] is the full list; tensors[0] is already the primary, so we use all
        ts = [torch.tensor(x, dtype=torch.float32) for x in params["tensors"]]
        return torch.cat(ts, dim=params.get("dim", 0))
    elif op == "stack":
        ts = [torch.tensor(x, dtype=torch.float32) for x in params["tensors"]]
        return torch.stack(ts, dim=params.get("dim", 0))

    raise ValueError(f"Unsupported operation: {op}")


# ── Graph metadata ─────────────────────────────────────────────────────────────

_OP_LABELS = {
    "add":       "add\n(+)",
    "sub":       "sub\n(−)",
    "mul":       "mul\n(×)",
    "div":       "div\n(÷)",
    "matmul":    "matmul\n(@)",
    "abs":       "abs\n|x|",
    "neg":       "neg\n(−x)",
    "clamp":     "clamp",
    "sum":       "sum",
    "reshape":   "reshape",
    "transpose": "transpose",
    "flatten":   "flatten",
    "squeeze":   "squeeze",
    "unsqueeze": "unsqueeze",
    "permute":   "permute",
    "tile":      "tile",
    "repeat":    "repeat",
    "narrow":    "narrow",
    "chunk":     "chunk",
    "cat":       "cat",
    "stack":     "stack",
}

_BINARY_OPS = {"add", "sub", "mul", "div", "matmul"}


def _op_label(op: str, params: dict) -> str:
    """Build an annotated label for the operation diamond node."""
    base = _OP_LABELS.get(op, op)
    if op == "clamp":
        lo = params.get("min_val", "−∞")
        hi = params.get("max_val", "+∞")
        return f"clamp\n[{lo}, {hi}]"
    elif op == "sum" and params.get("dim") is not None:
        return f"sum\ndim={params['dim']}"
    elif op == "reshape" and params.get("shape"):
        return f"reshape\n→{params['shape']}"
    elif op == "transpose":
        return f"transpose\n({params.get('dim0', 0)},{params.get('dim1', 1)})"
    elif op == "flatten":
        return f"flatten\n({params.get('start_dim', 0)}..{params.get('end_dim', -1)})"
    elif op == "unsqueeze":
        return f"unsqueeze\ndim={params.get('dim')}"
    elif op == "squeeze" and params.get("dim") is not None:
        return f"squeeze\ndim={params.get('dim')}"
    elif op == "permute" and params.get("dims"):
        return f"permute\n{params['dims']}"
    elif op == "tile" and params.get("dims"):
        return f"tile\n×{params['dims']}"
    elif op == "repeat" and params.get("sizes"):
        return f"repeat\n×{params['sizes']}"
    elif op == "narrow":
        return f"narrow\n(d={params.get('dim')}, s={params.get('start')}, l={params.get('length')})"
    elif op == "chunk":
        return f"chunk\n(n={params.get('chunks')}, d={params.get('dim', 0)})"
    elif op in {"cat", "stack"}:
        return f"{op}\ndim={params.get('dim', 0)}"
    return base


def generate_shape_graph(original_tensor: List, operations: List[Dict]) -> str:
    """
    Build a DAG showing tensor shapes after each operation.
    Returns a base64-encoded PNG data URI.
    """
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="TB", bgcolor="white")
    dot.attr("node", fontname="Helvetica", fontsize="11")
    dot.attr("edge", fontname="Helvetica", fontsize="10")

    t = torch.tensor(original_tensor, dtype=torch.float32)
    tensors = [t]
    tensor_nodes = []

    dot.node(
        "t0",
        f"Tensor 0\nshape {list(t.shape)}",
        shape="box",
        style="rounded,filled",
        fillcolor="#e0f2fe",
    )
    tensor_nodes.append("t0")

    for i, step in enumerate(operations):
        op = step["op"]
        tensor_b_data = step.get("tensor_b")
        params = step.get("params", {})

        input_node_ids = [tensor_nodes[-1]]

        # Second tensor node for binary ops
        if op in _BINARY_OPS and tensor_b_data is not None:
            t_b = torch.tensor(tensor_b_data, dtype=torch.float32)
            b_node_id = f"b{i}"
            dot.node(
                b_node_id,
                f"Tensor B{i}\nshape {list(t_b.shape)}",
                shape="box",
                style="rounded,filled",
                fillcolor="#fef9c3",
            )
            input_node_ids.append(b_node_id)

        # For cat/stack the extra tensors all feed into the op node
        if op in {"cat", "stack"} and params.get("tensors"):
            for j, extra in enumerate(params["tensors"][1:], start=1):
                t_extra = torch.tensor(extra, dtype=torch.float32)
                extra_node_id = f"cat_b{i}_{j}"
                dot.node(
                    extra_node_id,
                    f"Tensor B{i}_{j}\nshape {list(t_extra.shape)}",
                    shape="box",
                    style="rounded,filled",
                    fillcolor="#fef9c3",
                )
                input_node_ids.append(extra_node_id)

        op_node_id = f"op{i}"
        dot.node(
            op_node_id,
            _op_label(op, params),
            shape="diamond",
            style="filled",
            fillcolor="#f0fdf4",
        )
        for inp in input_node_ids:
            dot.edge(inp, op_node_id)

        output_t = _apply_op(tensors[-1], step)
        tensors.append(output_t)

        out_node_id = f"t{i + 1}"
        dot.node(
            out_node_id,
            f"Tensor {i + 1}\nshape {list(output_t.shape)}",
            shape="box",
            style="rounded,filled",
            fillcolor="#e0f2fe",
        )
        tensor_nodes.append(out_node_id)
        dot.edge(op_node_id, out_node_id)

    png_data = dot.pipe(format="png")
    return "data:image/png;base64," + base64.b64encode(png_data).decode("utf-8")
