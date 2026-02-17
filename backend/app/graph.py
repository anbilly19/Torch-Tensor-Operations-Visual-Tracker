import torch
import graphviz
import base64
from typing import List, Dict

def generate_shape_graph(original_tensor: List, operations: List[Dict]) -> str:
    """
    Build a graph showing tensor shape after each operation.
    Returns a base64‑encoded PNG image.
    """
    dot = graphviz.Digraph(format='png')
    dot.attr(rankdir='TB')  # top‑to‑bottom layout

    # Start with the original tensor
    t = torch.tensor(original_tensor, dtype=torch.float32)
    tensors = [t]                     # store all intermediate tensors
    tensor_nodes = []                  # Graphviz node IDs for each tensor

    # Original tensor node
    node_id = "t0"
    dot.node(node_id, f"Tensor 0\n{list(tensors[0].shape)}", shape='box', style='rounded')
    tensor_nodes.append(node_id)

    tensor_index = 1
    op_index = 0

    for step in operations:
        op = step['op']
        params = step.get('params', {})
        tensor_b_data = step.get('tensor_b')

        # Input tensor node(s): always the last tensor from previous step
        input_node_ids = [tensor_nodes[-1]]

        # For binary operations, add a node for the second tensor
        if tensor_b_data is not None:
            t_b = torch.tensor(tensor_b_data, dtype=torch.float32)
            b_node_id = f"b{op_index}"
            dot.node(b_node_id, f"B{op_index}\n{list(t_b.shape)}", shape='box', style='rounded')
            input_node_ids.append(b_node_id)

        # Operation node
        op_node_id = f"op{op_index}"
        dot.node(op_node_id, op, shape='diamond')

        # Connect inputs to operation
        for inp in input_node_ids:
            dot.edge(inp, op_node_id)

        # Apply the operation to compute the new tensor (and thus its shape)
        current_t = tensors[-1]
        if op == 'add':
            if tensor_b_data is None:
                raise ValueError("add requires tensor_b")
            t_b = torch.tensor(tensor_b_data, dtype=torch.float32)
            output_t = current_t + t_b
        elif op == 'matmul':
            if tensor_b_data is None:
                raise ValueError("matmul requires tensor_b")
            t_b = torch.tensor(tensor_b_data, dtype=torch.float32)
            output_t = torch.matmul(current_t, t_b)
        elif op == 'sum':
            dim = params.get('dim')
            keepdim = params.get('keepdim', False)
            output_t = current_t.sum(dim=dim, keepdim=keepdim)
        elif op == 'reshape':
            shape = params.get('shape')
            if not shape:
                raise ValueError("reshape requires shape")
            output_t = current_t.reshape(shape)
        elif op == 'transpose':
            dim0 = params.get('dim0', 0)
            dim1 = params.get('dim1', 1)
            output_t = torch.transpose(current_t, dim0, dim1)
        else:
            raise ValueError(f"Unsupported operation: {op}")

        # Output tensor node
        out_node_id = f"t{tensor_index}"
        dot.node(out_node_id, f"Tensor {tensor_index}\n{list(output_t.shape)}",
                 shape='box', style='rounded')
        tensor_nodes.append(out_node_id)
        tensors.append(output_t)
        tensor_index += 1

        # Connect operation to output
        dot.edge(op_node_id, out_node_id)

        op_index += 1

    # Render to PNG and encode as base64
    png_data = dot.pipe(format='png')
    b64_data = base64.b64encode(png_data).decode('utf-8')
    return f"data:image/png;base64,{b64_data}"