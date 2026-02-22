import torch
import graphviz
import base64
from typing import List, Dict


def _apply_op(current_t: torch.Tensor, step: Dict) -> torch.Tensor:
    """Apply a single operation step to current_t and return the result tensor."""
    op = step["op"]
    params = step.get("params", {})
    tensor_b_data = step.get("tensor_b")

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
    elif op == "abs":
        return torch.abs(current_t)
    elif op == "neg":
        return torch.neg(current_t)
    elif op == "clamp":
        return torch.clamp(
            current_t,
            min=params.get("min_val"),
            max=params.get("max_val"),
        )
    elif op == "sum":
        dim = params.get("dim")
        keepdim = params.get("keepdim", False)
        return current_t.sum(dim=dim, keepdim=keepdim)
    elif op == "reshape":
        return current_t.reshape(params["shape"])
    elif op == "transpose":
        return torch.transpose(current_t, params.get("dim0", 0), params.get("dim1", 1))
    else:
        raise ValueError(f"Unsupported operation: {op}")


# Label map: human-readable op names for graph nodes
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
}

# Ops that require a second tensor input in the graph
_BINARY_OPS = {"add", "sub", "mul", "div", "matmul"}


def generate_shape_graph(original_tensor: List, operations: List[Dict]) -> str:
    """
    Build a DAG showing tensor shapes after each operation.
    Returns a base64-encoded PNG.
    """
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="TB", bgcolor="white")
    dot.attr("node", fontname="Helvetica", fontsize="11")
    dot.attr("edge", fontname="Helvetica", fontsize="10")

    t = torch.tensor(original_tensor, dtype=torch.float32)
    tensors = [t]
    tensor_nodes = []

    # Root tensor node
    node_id = "t0"
    dot.node(
        node_id,
        f"Tensor 0\nshape {list(t.shape)}",
        shape="box",
        style="rounded,filled",
        fillcolor="#e0f2fe",
    )
    tensor_nodes.append(node_id)

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

        # Operation node — annotate clamp with its range
        op_label = _OP_LABELS.get(op, op)
        if op == "clamp":
            lo = params.get("min_val", "−∞")
            hi = params.get("max_val", "+∞")
            op_label = f"clamp\n[{lo}, {hi}]"
        elif op == "sum" and params.get("dim") is not None:
            op_label = f"sum\ndim={params['dim']}"
        elif op == "reshape" and params.get("shape"):
            op_label = f"reshape\n→{params['shape']}"
        elif op == "transpose":
            op_label = f"transpose\n({params.get('dim0',0)},{params.get('dim1',1)})"

        op_node_id = f"op{i}"
        dot.node(
            op_node_id,
            op_label,
            shape="diamond",
            style="filled",
            fillcolor="#f0fdf4",
        )

        for inp in input_node_ids:
            dot.edge(inp, op_node_id)

        # Compute output tensor
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
