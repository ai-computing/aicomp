#
# Copyright (c) 2024-present, ETRI, All rights reserved.
#

import sys
import math
import torch
import torch.nn as nn

#from transformers import GPT2Config
#from transformers import BertConfig
#from transformers import OPTConfig
#from transformers import WhisperConfig
#from transformers import LlamaConfig
#from transformers import GPT2LMHeadModel
#from transformers import GPTNeoForCausalLM
#from transformers import BertLMHeadModel
#from transformers import GPTJForCausalLM
#from transformers import BartForCausalLM
#from transformers import MBartForCausalLM
#from transformers import OPTForCausalLM
#from transformers import WhisperForCausalLM
#from transformers import GPT2ForSequenceClassification
#from transformers import LlamaForCausalLM

import transformers.utils.fx as hf_fx
import inspect

from torch import Tensor, Size
from torch import fx
from torch.fx.node import Node
from torch.fx.graph_module import GraphModule
from torch.fx.passes.split_module import split_module
import copy
import time

import torch.distributed as dist
import datetime
import logging
import os
import gc
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import operator

from transformers.utils.fx import _SUPPORTED_MODELS


# Check torch.export availability (requires PyTorch >= 2.4)
_TORCH_EXPORT_AVAILABLE = (
    hasattr(torch, 'export')
    and hasattr(torch.export, 'export')
)


def _inline_subgraph_into_graph(call_node, submod, submod_attr_node, env,
                                new_graph, grad_enabled=True):
    """Inline a synthetic subgraph (from @torch.no_grad() etc.) into the main graph.

    Args:
        call_node: The call_function node (wrap_with_set_grad_enabled) to replace
        submod: The synthetic fx.GraphModule containing the actual ATen ops
        submod_attr_node: The get_attr node that loaded the submod
        env: old_node -> new_node mapping (updated in place)
        new_graph: Target fx.Graph to add inlined nodes into
        grad_enabled: The grad-enabled flag from wrap_with_set_grad_enabled.
                      If False, detach outputs to prevent gradient flow back
                      through the inlined ops (preserving @torch.no_grad() semantics).
    """
    sub_graph = submod.graph
    sub_env = {}

    # Extract real args (skip bool/int/float constants and the GraphModule ref)
    real_args = []
    for arg in call_node.args:
        if isinstance(arg, (bool, int, float)):
            continue
        if isinstance(arg, fx.Node) and arg is submod_attr_node:
            continue
        if isinstance(arg, fx.Node):
            val = env.get(arg)
            if val is not None:
                real_args.append(val)
        else:
            real_args.append(arg)

    # Map subgraph placeholders to real args
    sub_placeholders = [n for n in sub_graph.nodes if n.op == 'placeholder']
    for i, ph in enumerate(sub_placeholders):
        if i < len(real_args):
            sub_env[ph] = real_args[i]

    def _sub_remap(x):
        if isinstance(x, fx.Node):
            return sub_env[x]
        return x

    # Process subgraph ATen ops (skip placeholder and output)
    for sub_node in sub_graph.nodes:
        if sub_node.op == 'placeholder':
            continue

        if sub_node.op == 'output':
            # Map subgraph output → env[call_node]
            out_args = sub_node.args[0]
            if isinstance(out_args, fx.Node):
                env[call_node] = sub_env[out_args]
            elif isinstance(out_args, (tuple, list)):
                mapped = []
                for a in out_args:
                    if isinstance(a, fx.Node):
                        mapped.append(sub_env[a])
                    else:
                        mapped.append(a)
                env[call_node] = tuple(mapped) if len(mapped) > 1 else mapped[0]
            else:
                env[call_node] = out_args

            # If the original region was @torch.no_grad() (grad_enabled=False),
            # insert detach() on tensor outputs to prevent gradient flow back
            # through the inlined ops during backward pass.
            if not grad_enabled:
                val = env[call_node]
                if isinstance(val, fx.Node):
                    det = new_graph.call_function(
                        torch.ops.aten.detach.default, (val,))
                    det.meta = dict(val.meta) if val.meta else {}
                    env[call_node] = det
                elif isinstance(val, tuple):
                    detached = []
                    for v in val:
                        if isinstance(v, fx.Node):
                            det = new_graph.call_function(
                                torch.ops.aten.detach.default, (v,))
                            det.meta = dict(v.meta) if v.meta else {}
                            detached.append(det)
                        else:
                            detached.append(v)
                    env[call_node] = tuple(detached)
            continue

        new_args = fx.map_arg(sub_node.args, _sub_remap)
        new_kwargs = fx.map_arg(sub_node.kwargs, _sub_remap)

        if sub_node.op == 'call_function':
            new_node = new_graph.call_function(sub_node.target, new_args, new_kwargs)
            new_node.meta = dict(sub_node.meta) if sub_node.meta else {}
            sub_env[sub_node] = new_node
        elif sub_node.op == 'call_method':
            new_node = new_graph.call_method(sub_node.target, new_args, new_kwargs)
            new_node.meta = dict(sub_node.meta) if sub_node.meta else {}
            sub_env[sub_node] = new_node
        elif sub_node.op == 'get_attr':
            # Target is relative to submod — prepend submod target for full path
            full_target = f"{submod_attr_node.target}.{sub_node.target}"
            new_node = new_graph.get_attr(full_target)
            new_node.meta = dict(sub_node.meta) if sub_node.meta else {}
            sub_env[sub_node] = new_node


def _inline_higher_order_ops(flat_gm):
    """Inline higher-order ops (from @torch.no_grad() etc.) into the main graph.

    torch.export with strict=False creates synthetic subgraph modules (submod_0,
    submod_1, ...) for @torch.no_grad() decorated functions. The main graph only
    has get_attr + call_function[wrap_with_set_grad_enabled] for these. This
    function replaces them with the actual ATen ops from the subgraph, so the
    downstream module-graph reconstruction never sees synthetic GraphModules.

    Args:
        flat_gm: The GraphModule from ExportedProgram

    Returns:
        A new GraphModule with higher-order ops inlined (or flat_gm unchanged
        if there are no higher-order ops)
    """
    old_graph = flat_gm.graph

    # Step 1: Detect synthetic GraphModule attributes (from @torch.no_grad() etc.)
    synthetic_modules = {}  # node -> fx.GraphModule
    for node in old_graph.nodes:
        if node.op == 'get_attr':
            try:
                obj = flat_gm
                for p in node.target.split('.'):
                    obj = getattr(obj, p)
                if isinstance(obj, fx.GraphModule):
                    synthetic_modules[node] = obj
            except AttributeError:
                pass

    if not synthetic_modules:
        return flat_gm  # No higher-order ops — return unchanged

    # Detect grad_enabled flags by scanning call sites that use each synthetic module
    synth_grad_flags = {}  # submod_attr_node -> grad_enabled
    for node in old_graph.nodes:
        if node.op == 'call_function':
            for arg in node.all_input_nodes:
                if arg in synthetic_modules:
                    grad_flag = True
                    for a in node.args:
                        if isinstance(a, bool):
                            grad_flag = a
                            break
                    synth_grad_flags[arg] = grad_flag

    if int(os.environ.get("RANK", "0")) == 0:
        print(f">> [IR] Inlining {len(synthetic_modules)} higher-order op(s) "
              f"(from @torch.no_grad() etc.)")
        for node, submod in synthetic_modules.items():
            sub_nodes = sum(1 for n in submod.graph.nodes
                           if n.op not in ('placeholder', 'output'))
            grad_flag = synth_grad_flags.get(node, True)
            detach_str = " → outputs detached (no-grad)" if not grad_flag else ""
            print(f">>   {node.name} ({node.target}): {sub_nodes} ATen ops, "
                  f"grad_enabled={grad_flag}{detach_str}")

    # Step 2: Build new graph with inlined subgraphs
    new_graph = fx.Graph()
    env = {}  # old_node -> new_node (or tuple of new_nodes for multi-output)

    for node in old_graph.nodes:
        if node in synthetic_modules:
            continue  # Skip get_attr for synthetic GraphModule

        if node.op == 'placeholder':
            new_node = new_graph.placeholder(node.name)
            new_node.name = node.name
            new_node.meta = dict(node.meta) if node.meta else {}
            env[node] = new_node

        elif node.op == 'output':
            out_args = fx.map_arg(node.args[0], lambda n: env[n])
            new_graph.output(out_args)

        elif node.op == 'call_function':
            # Check if this call uses a synthetic GraphModule arg
            submod_node = None
            for arg in node.all_input_nodes:
                if arg in synthetic_modules:
                    submod_node = arg
                    break

            if submod_node is not None:
                # Extract grad_enabled flag from wrap_with_set_grad_enabled.
                # The call looks like: wrap_with_set_grad_enabled(False, submod, ...)
                # where the first bool arg is the grad-enabled flag.
                grad_enabled = True
                for arg in node.args:
                    if isinstance(arg, bool):
                        grad_enabled = arg
                        break

                # Inline the subgraph
                _inline_subgraph_into_graph(
                    node, synthetic_modules[submod_node], submod_node,
                    env, new_graph, grad_enabled=grad_enabled
                )
            elif (node.target is operator.getitem
                  and isinstance(node.args[0], fx.Node)
                  and isinstance(env.get(node.args[0]), tuple)):
                # Resolve getitem on multi-output inlined op directly
                idx = node.args[1]
                env[node] = env[node.args[0]][idx]
            else:
                # Normal call_function
                new_args = fx.map_arg(node.args, lambda n: env[n])
                new_kwargs = fx.map_arg(node.kwargs, lambda n: env[n])
                new_node = new_graph.call_function(node.target, new_args, new_kwargs)
                new_node.name = node.name
                new_node.meta = dict(node.meta) if node.meta else {}
                env[node] = new_node

        elif node.op == 'get_attr':
            new_node = new_graph.get_attr(node.target)
            new_node.name = node.name
            new_node.meta = dict(node.meta) if node.meta else {}
            env[node] = new_node

        elif node.op == 'call_method':
            new_args = fx.map_arg(node.args, lambda n: env[n])
            new_kwargs = fx.map_arg(node.kwargs, lambda n: env[n])
            new_node = new_graph.call_method(node.target, new_args, new_kwargs)
            new_node.name = node.name
            new_node.meta = dict(node.meta) if node.meta else {}
            env[node] = new_node

    new_graph.lint()
    return GraphModule(flat_gm, new_graph)


def build_module_graph_from_export(exported_program, original_model, input_names=None):
    """
    Reconstruct a module-level FX graph from ExportedProgram using
    nn_module_stack metadata, completely bypassing torch.export.unflatten().

    Produces a flat graph with call_module nodes for leaf nn.Modules,
    matching the format HFTracer would generate.

    Args:
        exported_program: Result of torch.export.export()
        original_model: The original nn.Module (provides module hierarchy)
        input_names: List of user input names (e.g., ['input_ids', 'position_ids'])

    Returns:
        GraphModule compatible with split_module() and existing split logic
    """
    flat_gm = exported_program.graph_module

    # ── Pre-processing: Inline higher-order ops ──
    # torch.export (strict=False) wraps @torch.no_grad() functions in synthetic
    # subgraph modules (submod_0, submod_1, ...).  Inline them so the module-graph
    # reconstruction never sees synthetic GraphModules.
    flat_gm = _inline_higher_order_ops(flat_gm)

    flat_graph = flat_gm.graph
    sig = exported_program.graph_signature

    # ── Step 1: Identify leaf modules (no children = leaf) ──
    leaf_fqns = set()
    for fqn, mod in original_model.named_modules():
        if fqn and len(list(mod.children())) == 0:
            leaf_fqns.add(fqn)

    # ── Step 2: Classify placeholder nodes ──
    param_buffer_nodes = set()   # lifted params / buffers
    user_input_nodes = []        # actual user inputs (input_ids, ...)

    # Build name→FQN map for params, buffers, and lifted tensor constants
    param_names = set()
    buffer_names = set()
    lifted_constant_names = set()
    if hasattr(sig, 'inputs_to_parameters'):
        param_names = set(sig.inputs_to_parameters.keys())
    if hasattr(sig, 'inputs_to_buffers'):
        buffer_names = set(sig.inputs_to_buffers.keys())
    if hasattr(sig, 'inputs_to_lifted_tensor_constants') and sig.inputs_to_lifted_tensor_constants:
        lifted_constant_names = set(sig.inputs_to_lifted_tensor_constants.keys())
        if int(os.environ.get("RANK", "0")) == 0:
            print(f">> [IR] Found {len(lifted_constant_names)} lifted tensor constant(s) "
                  f"(e.g. local attention masks)")
        # Register lifted tensor constants as buffers on original_model
        # so that get_attr can find them at runtime
        for placeholder_name, fqn in sig.inputs_to_lifted_tensor_constants.items():
            tensor_val = None
            if hasattr(exported_program, 'constants') and fqn in exported_program.constants:
                tensor_val = exported_program.constants[fqn]
            elif hasattr(flat_gm, fqn.replace('.', '_')):
                tensor_val = getattr(flat_gm, fqn.replace('.', '_'))
            if tensor_val is not None:
                # Walk/create the submodule path and register as buffer
                parts = fqn.split('.')
                parent = original_model
                for part in parts[:-1]:
                    if hasattr(parent, part):
                        parent = getattr(parent, part)
                    else:
                        # Create intermediate Module if needed
                        sub = torch.nn.Module()
                        parent.add_module(part, sub)
                        parent = sub
                # Skip if attribute already exists (e.g. GPT-J embed_positions)
                if not hasattr(parent, parts[-1]):
                    parent.register_buffer(parts[-1], tensor_val, persistent=False)

    for node in flat_graph.nodes:
        if node.op == 'placeholder':
            if node.name in param_names or node.name in buffer_names or node.name in lifted_constant_names:
                param_buffer_nodes.add(node)
            else:
                user_input_nodes.append(node)

    # Map exported user‐input placeholder names → canonical names
    # When kwargs are used with export(), placeholder names already match
    # parameter names (e.g., "input_ids", "position_ids").
    # When positional args are used, names may be mangled ("arg0_1", etc.)
    # and need remapping via input_names.
    canonical_names = {}
    if input_names:
        input_name_set = set(input_names)
        # First: if placeholder name already matches an input name, use it
        matched = set()
        for node in user_input_nodes:
            if node.name in input_name_set:
                canonical_names[node.name] = node.name
                matched.add(node.name)
        # Then: for remaining unmatched, fall back to positional mapping
        remaining_names = [n for n in input_names if n not in matched]
        remaining_nodes = [n for n in user_input_nodes if n.name not in matched]
        for i, node in enumerate(remaining_nodes):
            if i < len(remaining_names):
                canonical_names[node.name] = remaining_names[i]
            else:
                canonical_names[node.name] = node.name
    else:
        for node in user_input_nodes:
            canonical_names[node.name] = node.name

    # ── Step 3: Assign each node to its owning leaf module ──
    def _get_owning_leaf(node):
        stack = node.meta.get('nn_module_stack', {})
        if not stack:
            return None
        # Walk from deepest to shallowest
        for _key, (fqn, _cls) in reversed(list(stack.items())):
            if fqn in leaf_fqns:
                return fqn
        return None

    node_owner = {}  # node → fqn or None
    for node in flat_graph.nodes:
        if node.op in ('placeholder', 'output'):
            node_owner[node] = None
        else:
            node_owner[node] = _get_owning_leaf(node)


    # ── Step 4: Build execution regions ──
    # Each region is either a "module" region (all nodes for one leaf module call)
    # or an "inline" region (ops not inside any leaf module).
    regions = []  # list of dicts: {'type': 'module'|'inline', 'fqn': str|None, 'nodes': [Node]}

    current_region = None
    for node in flat_graph.nodes:
        if node.op in ('placeholder', 'output'):
            continue
        owner = node_owner[node]
        if owner is not None:
            # Module node
            if current_region and current_region['type'] == 'module' and current_region['fqn'] == owner:
                current_region['nodes'].append(node)
            else:
                current_region = {'type': 'module', 'fqn': owner, 'nodes': [node]}
                regions.append(current_region)
        else:
            # Inline node
            if current_region and current_region['type'] == 'inline':
                current_region['nodes'].append(node)
            else:
                current_region = {'type': 'inline', 'fqn': None, 'nodes': [node]}
                regions.append(current_region)

    # ── Step 4b: Validate module regions for nn.Embedding subclasses ──
    # When a subclass of nn.Embedding overrides forward() (e.g.
    # OPTLearnedPositionalEmbedding), TorchDynamo may not attribute all
    # internal ops to the leaf module.  For OPTLearnedPositionalEmbedding:
    #   forward(attention_mask, ...):
    #       position_ids = cumsum(attention_mask) ...  # position computation
    #       return super().forward(position_ids + offset)  # aten.embedding
    #
    # TorchDynamo may classify cumsum/mul/sub/slice as belonging to the
    # parent module (OPTDecoder), leaving only add_offset + aten.embedding
    # in the embed_positions region.  The inline ops compute position_ids,
    # and the call_module receives these pre-computed position_ids.  But
    # the call_module runs the FULL overridden forward(), which expects
    # attention_mask and internally recomputes cumsum → double-computation
    # of position indices → out-of-bounds embedding access.
    #
    # Fix: for nn.Embedding SUBCLASSES that override forward(), always
    # demote to inline.  Plain nn.Embedding (no override) is safe since
    # its forward is just F.embedding → single aten.embedding op.
    for region in regions:
        if region['type'] != 'module':
            continue
        fqn = region['fqn']
        try:
            mod = original_model.get_submodule(fqn)
        except (AttributeError, ValueError):
            continue
        if not isinstance(mod, nn.Embedding):
            continue
        # Plain nn.Embedding: forward is just F.embedding, safe as call_module
        if type(mod) is nn.Embedding:
            continue
        # Subclass with overridden forward() — always demote to inline
        if int(os.environ.get("RANK", "0")) == 0:
            print(f">> [IR] DEMOTING nn.Embedding subclass region "
                  f"'{fqn}' ({type(mod).__name__}): overridden forward() "
                  f"may have ops split across inline/module boundaries")
        region['type'] = 'inline'
        region['fqn'] = None

    # ── Step 5: Build new module-level graph ──
    new_graph = fx.Graph()
    value_map = {}   # old_node → new_node
    used_names = set()
    # Track get_attr nodes we already created (for buffers referenced by inline ops)
    created_getattrs = {}  # buffer_fqn → new_node

    def _make_name(base: str) -> str:
        """Generate unique node name from a base string."""
        name = base.replace('.', '_')
        if name not in used_names:
            used_names.add(name)
            return name
        i = 1
        while f"{name}_{i}" in used_names:
            i += 1
        used_names.add(f"{name}_{i}")
        return f"{name}_{i}"

    def _remap(x):
        """Remap a single arg from old graph to new graph."""
        if isinstance(x, fx.Node):
            if x in value_map:
                return value_map[x]
            # If it's a param/buffer placeholder not yet mapped, create get_attr
            if x in param_buffer_nodes:
                return _get_or_create_getattr(x)
            raise KeyError(f"Unmapped node: {x.name} (op={x.op})")
        return x

    def _remap_all(args):
        return fx.map_arg(args, _remap)

    def _get_or_create_getattr(param_node):
        """Create a get_attr node for a parameter/buffer placeholder."""
        if param_node in value_map:
            return value_map[param_node]

        # Find the FQN for this param/buffer/lifted constant
        fqn = None
        if hasattr(sig, 'inputs_to_parameters') and param_node.name in sig.inputs_to_parameters:
            fqn = sig.inputs_to_parameters[param_node.name]
        elif hasattr(sig, 'inputs_to_buffers') and param_node.name in sig.inputs_to_buffers:
            fqn = sig.inputs_to_buffers[param_node.name]
        elif hasattr(sig, 'inputs_to_lifted_tensor_constants') and param_node.name in sig.inputs_to_lifted_tensor_constants:
            fqn = sig.inputs_to_lifted_tensor_constants[param_node.name]

        if fqn:
            if fqn not in created_getattrs:
                ga_name = _make_name(fqn)
                ga_node = new_graph.get_attr(fqn)
                ga_node.name = ga_name
                created_getattrs[fqn] = ga_node
            value_map[param_node] = created_getattrs[fqn]
            return created_getattrs[fqn]
        else:
            # Unknown param/buffer — create placeholder as fallback
            ph = new_graph.placeholder(param_node.name)
            value_map[param_node] = ph
            return ph

    def _resolve_get_attr_target(target):
        """Resolve a get_attr target to a valid FQN on original_model.

        torch.export with strict=False may create synthetic submodules
        (e.g. submod_0, submod_1) for @torch.no_grad() wrappers.
        These exist on the export graph_module but not on original_model.
        We resolve them by matching the actual tensor via data_ptr().
        """
        # Check if target is already valid on original_model
        parts = target.split('.')
        try:
            obj = original_model
            for p in parts:
                obj = getattr(obj, p)
            return target
        except AttributeError:
            pass

        # Target is synthetic — resolve via flat_gm
        try:
            obj = flat_gm
            for p in parts:
                obj = getattr(obj, p)
        except AttributeError:
            return target  # can't resolve

        if isinstance(obj, torch.Tensor):
            # Match tensor by data_ptr to find real FQN
            for name, param in original_model.named_parameters():
                if param.data_ptr() == obj.data_ptr():
                    return name
            for name, buf in original_model.named_buffers():
                if buf.data_ptr() == obj.data_ptr():
                    return name
        elif isinstance(obj, nn.Module):
            # Match module by identity
            for name, mod in original_model.named_modules():
                if mod is obj:
                    return name

        return target  # fallback: return as-is

    # 5a. Create user input placeholders with canonical names
    for node in user_input_nodes:
        cname = canonical_names.get(node.name, node.name)
        new_ph = new_graph.placeholder(cname)
        new_ph.name = _make_name(cname)
        value_map[node] = new_ph

    # ── Helper: determine if a node produces a tensor (not a scalar/int) ──
    def _is_tensor_producer(n):
        if n.op in ('call_module', 'placeholder', 'get_attr'):
            return True
        if n.op == 'call_function':
            tgt = getattr(n.target, '__name__', str(n.target))
            if 'sym_size' in tgt or 'sym_numel' in tgt:
                return False
            return True
        return False

    # Counter for tracking skipped nodes across all _emit_inline calls
    _skipped_nodes_info = []  # list of (node_name, target, error)

    # ── Helper: process a list of nodes as inline ATen ops ──
    def _emit_inline(nodes_list):
        """Copy ATen ops into new_graph, fixing device=cpu kwargs."""
        region_set = set(nodes_list)

        # Find an external TENSOR input for device inference
        _device_ref = None
        _need_dev = any(
            n.op == 'call_function'
            and isinstance(n.kwargs.get('device'), torch.device)
            and str(n.kwargs['device']) == 'cpu'
            for n in nodes_list
        )
        if _need_dev:
            # Prefer call_module / placeholder (guaranteed tensors)
            for n in nodes_list:
                for arg in n.all_input_nodes:
                    if arg not in region_set and arg not in param_buffer_nodes:
                        if arg in value_map and arg.op in ('call_module', 'placeholder'):
                            _device_ref = value_map[arg]
                            break
                if _device_ref:
                    break
            # Fallback: any tensor-producing external node
            if _device_ref is None:
                for n in nodes_list:
                    for arg in n.all_input_nodes:
                        if arg not in region_set and arg not in param_buffer_nodes:
                            if arg in value_map and _is_tensor_producer(arg):
                                _device_ref = value_map[arg]
                                break
                    if _device_ref:
                        break
            # Final fallback: when all inputs are SymInts (e.g., torch.arange
            # computing position_ids from shape metadata), find any previously
            # emitted placeholder or call_module in the graph.
            if _device_ref is None:
                for prev_node in new_graph.nodes:
                    if prev_node.op in ('placeholder', 'call_module'):
                        _device_ref = prev_node
                        break

        # Create device extraction node if needed
        _dev_node = None
        if _device_ref is not None:
            _dev_node = new_graph.call_function(
                getattr, args=(_device_ref, 'device')
            )
            _dev_node.name = _make_name('_inline_device')

        for node in nodes_list:
            try:
                new_args = _remap_all(node.args)
                new_kwargs = _remap_all(node.kwargs)
            except KeyError as e:
                tgt = getattr(node.target, '__name__', str(node.target))
                _skipped_nodes_info.append((node.name, tgt, str(e)))
                if int(os.environ.get("RANK", "0")) == 0:
                    print(f">> [IR] _emit_inline SKIPPED node '{node.name}' "
                          f"(target={tgt}): {e}")
                continue

            # Replace device=cpu with runtime device reference
            if (_dev_node is not None
                    and node.op == 'call_function'
                    and isinstance(node.kwargs.get('device'), torch.device)
                    and str(node.kwargs['device']) == 'cpu'):
                new_kwargs = dict(new_kwargs)
                new_kwargs['device'] = _dev_node

            if node.op == 'call_function':
                nn = _make_name(node.name)
                target = node.target
                # Replace aten.view / _unsafe_view with aten.reshape for
                # non-contiguous tensors (after transpose in attention)
                if (target is torch.ops.aten.view.default
                        or target is torch.ops.aten._unsafe_view.default):
                    target = torch.ops.aten.reshape.default
                new_node = new_graph.call_function(target, new_args, new_kwargs)
                new_node.name = nn
                value_map[node] = new_node
            elif node.op == 'call_method':
                nn = _make_name(node.name)
                new_node = new_graph.call_method(node.target, new_args, new_kwargs)
                new_node.name = nn
                value_map[node] = new_node
            elif node.op == 'get_attr':
                resolved = _resolve_get_attr_target(node.target)
                nn = _make_name(resolved)
                new_node = new_graph.get_attr(resolved)
                new_node.name = nn
                value_map[node] = new_node

    # 5b. Process regions
    for region in regions:
        if region['type'] == 'module':
            fqn = region['fqn']
            nodes = region['nodes']
            node_set = set(nodes)

            # Find inputs: values from outside this region, excluding
            # params/buffers AND non-tensor SymInt values (sym_size, etc.).
            # SymInts represent shape metadata that will be recomputed at
            # runtime from the actual tensors passed to the module.
            inputs = []
            seen_inputs = set()
            for n in nodes:
                for arg_node in n.all_input_nodes:
                    if arg_node not in node_set and arg_node not in param_buffer_nodes:
                        if id(arg_node) not in seen_inputs:
                            seen_inputs.add(id(arg_node))
                            if _is_tensor_producer(arg_node):
                                inputs.append(arg_node)


            # ── Signature check ──
            # If the number of external inputs doesn't match the module's
            # forward() required parameters, the call_module would fail
            # (e.g., LlamaRotaryEmbedding.forward(x, position_ids) where
            # x is only used for device/dtype — no ATen ops reference it).
            #
            # Strategy:
            # 1. If exactly 1 input is missing, try to find a suitable
            #    tensor from preceding nodes to use as a substitute.
            #    This preserves the module's internal precision handling
            #    (e.g., torch.autocast(enabled=False) in RoPE).
            # 2. Otherwise, demote the region to inline ATen ops.
            _demote = False
            _padded_inputs = None
            try:
                mod = original_model.get_submodule(fqn)
                fwd_sig = inspect.signature(mod.forward)
                required = [
                    p for p in fwd_sig.parameters.values()
                    if p.name != 'self'
                    and p.default is inspect.Parameter.empty
                    and p.kind not in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    )
                ]
                if len(inputs) < len(required):
                    n_missing = len(required) - len(inputs)
                    if n_missing == 1:
                        # Exactly 1 input is missing.  Find a suitable
                        # substitute tensor from preceding call_module outputs.
                        # The missing param is typically a "shape/device/dtype
                        # reference" (like x in LlamaRotaryEmbedding) that the
                        # module only uses for metadata, not for computation.
                        _sub_tensor = None
                        for prev_node in reversed(list(new_graph.nodes)):
                            if prev_node.op == 'call_module':
                                _sub_tensor = prev_node
                                break
                        if _sub_tensor is not None:
                            # Prepend the substitute as the first arg
                            # (the missing param is typically the first one)
                            _padded_inputs = [_sub_tensor] + inputs
                            if int(os.environ.get("RANK", "0")) == 0:
                                print(f">> [IR] Padded missing input for "
                                      f"'{fqn}' with preceding "
                                      f"call_module '{_sub_tensor.name}'")
                        else:
                            _demote = True
                    else:
                        _demote = True
            except (AttributeError, ValueError):
                pass

            if _demote:
                if int(os.environ.get("RANK", "0")) == 0:
                    print(f">> [IR] DEMOTED '{fqn}' to inline "
                          f"({len(inputs)} inputs vs {len(required)} required)")
                _emit_inline(nodes)
                continue

            # Find outputs: nodes from this region used outside
            outputs = []
            seen_outputs = set()
            for n in nodes:
                for user in n.users:
                    if user not in node_set and user.op != 'output':
                        if id(n) not in seen_outputs:
                            outputs.append(n)
                            seen_outputs.add(id(n))
                            break
                # Also check if this node is referenced by the graph's output node
                if id(n) not in seen_outputs:
                    for user in n.users:
                        if user.op == 'output':
                            outputs.append(n)
                            seen_outputs.add(id(n))
                            break

            # Create call_module node
            # _padded_inputs contains new-graph nodes for the substitute(s)
            # followed by old-graph nodes for the real inputs.
            if _padded_inputs is not None:
                # The substitute tensor is already a new-graph node;
                # the rest need remapping from old graph → new graph.
                call_args = tuple(
                    inp if inp.graph is new_graph else _remap(inp)
                    for inp in _padded_inputs
                )
            else:
                call_args = tuple(_remap(inp) for inp in inputs)
            call_name = _make_name(fqn)
            call_node = new_graph.call_module(fqn, args=call_args)
            call_node.name = call_name

            # Map outputs
            if len(outputs) == 0:
                if nodes:
                    value_map[nodes[-1]] = call_node
            elif len(outputs) == 1:
                value_map[outputs[0]] = call_node
            else:
                for i, out_node in enumerate(outputs):
                    gi_name = _make_name(f"getitem_{fqn}")
                    gi_node = new_graph.call_function(
                        operator.getitem, args=(call_node, i)
                    )
                    gi_node.name = gi_name
                    value_map[out_node] = gi_node

        else:
            # Inline region
            _emit_inline(region['nodes'])

    # 5c. Create output node
    output_nodes = [n for n in flat_graph.nodes if n.op == 'output']
    if output_nodes:
        orig_output = output_nodes[0]
        try:
            new_output_args = _remap_all(orig_output.args[0])
        except (KeyError, TypeError):
            # Fallback: find the last call_module node as output
            last_call = None
            for n in reversed(list(new_graph.nodes)):
                if n.op == 'call_module':
                    last_call = n
                    break
            new_output_args = (last_call,) if last_call else ()
        new_graph.output(new_output_args)

    # ── Step 6: Create GraphModule ──
    # torch.export (strict=False) may create synthetic submodules (submod_0,
    # submod_1, ...) for @torch.no_grad() wrappers.  These exist on flat_gm
    # but not on original_model.  Temporarily register them so that
    # GraphModule.__init__ can copy the attributes via _copy_attr.
    _synthetic = []
    for node in new_graph.nodes:
        if node.op == 'get_attr':
            first = node.target.split('.')[0]
            if not hasattr(original_model, first) and hasattr(flat_gm, first):
                original_model.add_module(first, getattr(flat_gm, first))
                _synthetic.append(first)

    new_graph.lint()
    gm = GraphModule(original_model, new_graph)

    # Clean up: remove synthetic modules from original_model
    for name in _synthetic:
        delattr(original_model, name)

    # Report skipped nodes summary
    if _skipped_nodes_info and int(os.environ.get("RANK", "0")) == 0:
        print(f">> [IR] WARNING: {len(_skipped_nodes_info)} inline node(s) were "
              f"SKIPPED during graph reconstruction!")
        for name, tgt, err in _skipped_nodes_info:
            print(f">>   SKIPPED: {name} (target={tgt}): {err}")
    elif int(os.environ.get("RANK", "0")) == 0:
        print(f">> [IR] Graph reconstruction: 0 nodes skipped (all inline ops preserved)")

    return gm


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

logging.basicConfig(level=logging.ERROR)

class IR_Anal(Enum):
    SINGLE = 1      # Experimental
    PARALLEL = 2
    SEQUENTIAL = 3


#huggingface_model_class = [
#        GPT2LMHeadModel, 
#        GPTNeoForCausalLM, 
#        BertLMHeadModel, 
#        GPTJForCausalLM, 
#        BartForCausalLM,
#        MBartForCausalLM, 
#        OPTForCausalLM,
#        WhisperForCausalLM,
#        GPT2ForSequenceClassification,
#        LlamaForCausalLM,
#        # TODO
#        ]

_OTHER_MODELS = [
        "WhisperForCausalLM",
        "Qwen2MoeForCausalLM",
        ]

class IR(object):
    def __init__(self, model: nn.Module, optimus):

        self.gm = None
        self.model_ir = []
        self.metadata_range = []

        self.optimus = optimus

        self.special_nodes: Dict[str, Tuple[int, int]] = {}  # { node_name : {stage#, needed-by-stage#),}


    def retrieve_IR(self, model: nn.Module, use_kv_cache: bool = False, dynamo_capture: bool = False, for_training: bool = True):

        self.dynamo_capture = dynamo_capture

        if dynamo_capture:
            return self._retrieve_IR_export(model, use_kv_cache, for_training=for_training)
        else:
            return self._retrieve_IR_trace(model, use_kv_cache)


    def _retrieve_IR_trace(self, model: nn.Module, use_kv_cache: bool = False):
        """Existing HFTracer / symbolic_trace path (unchanged)."""

        ##
        if model.__class__.__name__ in [ "ViTForImageClassification" ]:
            input_names = ['pixel_values']
            print(f" vit >> input_names: {input_names}")

            sig = inspect.signature(model.forward)
            concrete_args = {
                p.name: p.default
                for p in sig.parameters.values()
                if p.name not in input_names
            }

            tracer = hf_fx.HFTracer()

            traced_graph = tracer.trace(model, concrete_args=concrete_args)
            self.gm = torch.fx.GraphModule(model, traced_graph)
            return self.optimus.model2type["vt"]

        elif model.__class__.__name__ in _SUPPORTED_MODELS or model.__class__.__name__ in _OTHER_MODELS:
            input_names = list(model.dummy_inputs.keys())
            if use_kv_cache:
                if 'position_ids' not in input_names:
                    input_names.append('position_ids')

            sig = inspect.signature(model.forward)
            concrete_args = {
                p.name: p.default
                for p in sig.parameters.values()
                if p.name not in input_names
            }

            tracer = hf_fx.HFTracer()

            traced_graph = tracer.trace(model, concrete_args=concrete_args)
            self.gm = torch.fx.GraphModule(model, traced_graph)
            return self.optimus.model2type["hf"]

        elif isinstance(model, nn.Module):
            self.gm = fx.symbolic_trace(model)
            return self.optimus.model2type["sy"]

        else:
            print(f"Not supported model!")
            sys.exit(1)


    def _build_dynamic_shapes(self, input_names: List[str], example_inputs: dict,
                              model: nn.Module = None):
        """Build dynamic_shapes spec so torch.export doesn't bake in example sizes.

        Without this, inline call_function nodes (view, reshape, etc.) would
        contain literal shape values from the example inputs, causing shape
        mismatches at runtime with different sequence lengths.

        Returns a dict keyed by input name (matching kwargs passed to export()).
        """
        from torch.export import Dim

        batch_dim = Dim("batch", min=1, max=256)

        # Cap seq_len max from the model's position embedding table size.
        # Use max_position_embeddings - 1 because models like BERT generate a
        # Ne(max_pos, seq_len) guard — the exact boundary triggers
        # specialization in torch.export.
        seq_max = 16384
        if model is not None and hasattr(model, 'config'):
            mpe = getattr(model.config, 'max_position_embeddings', None)
            if mpe is not None:
                seq_max = mpe - 1
        seq_dim = Dim("seq_len", min=1, max=seq_max)

        # Text-model inputs that share the seq_len dimension
        _SEQ_INPUTS = {'input_ids', 'position_ids', 'attention_mask', 'token_type_ids'}

        specs = {}
        for name in input_names:
            tensor = example_inputs.get(name)
            if tensor is None:
                continue
            ndim = tensor.dim()
            if name in _SEQ_INPUTS and ndim >= 2:
                specs[name] = {0: batch_dim, 1: seq_dim}
            elif name == 'pixel_values' and ndim >= 1:
                specs[name] = {0: batch_dim}
            elif ndim >= 1:
                specs[name] = {0: batch_dim}
        return specs

    def _retrieve_IR_export(self, model: nn.Module, use_kv_cache: bool = False, for_training: bool = True):
        """torch.export.export() path — bypasses unflatten() entirely."""

        if not _TORCH_EXPORT_AVAILABLE:
            print(f"[IR] torch.export.export() requires PyTorch >= 2.4.0. "
                  f"Current version: {torch.__version__}")
            print(f"[IR] Please upgrade PyTorch or use dynamo_capture=False.")
            sys.exit(1)

        from torch.export import export

        # For training (use_kv_cache=False), disable use_cache to simplify the
        # export graph.  With use_cache=True, past_key_values are computed and
        # included in the output, creating a complex tuple that confuses the
        # pipeline split and loss computation.  Models like OPT load from
        # pretrained with use_cache=True by default, unlike LLaMA examples
        # which explicitly set use_cache=False.
        _orig_use_cache = None
        if (not use_kv_cache
                and hasattr(model, 'config')
                and getattr(model.config, 'use_cache', False)):
            _orig_use_cache = model.config.use_cache
            model.config.use_cache = False

        # Disable layerdrop before export.  Models like Whisper use
        # data-dependent control flow (`random.uniform() < layerdrop`) inside
        # an `if self.training:` guard.  torch.export cannot handle
        # data-dependent branches (even with layerdrop=0.0, random.uniform
        # creates an unbacked symbolic value during tracing).
        # Fix: after model.train(), set training=False on modules with
        # layerdrop so the `if self.training:` guard is never entered.
        _layerdrop_modules = []
        for mod in model.modules():
            if hasattr(mod, 'layerdrop'):
                _layerdrop_modules.append(mod)
        # Determine input names and create example inputs
        input_names = self._get_export_input_names(model, use_kv_cache)
        example_inputs = self._create_example_inputs(model, input_names)

        # Mark batch and seq_len dimensions as dynamic so the exported graph
        # doesn't hardcode the example-input shapes into view/reshape ops.
        dynamic_shapes = self._build_dynamic_shapes(input_names, example_inputs, model)

        # Export with strict=False (TorchDynamo non-strict mode)
        # — supports more Python constructs (e.g., @torch.no_grad in LLaMA RoPE)
        # IMPORTANT: Pass inputs as kwargs (not positional args) so each tensor
        # binds to the correct parameter name in forward().  Positional args
        # would mis-bind (e.g., position_ids going to attention_mask).
        #
        # Always export in TRAINING mode for training.  This is critical
        # because TorchDynamo bakes the model.training flag into the ATen
        # graph:
        #   - Dropout: eval-mode drops all ATen ops → dropout never applied
        #   - _update_causal_mask (LLaMA): eval returns explicit 4D mask →
        #     SDPA uses math kernel instead of Flash Attention → different
        #     backward gradients → training divergence
        # For inference (for_training=False), keep eval mode.
        _has_active_dropout = False
        if for_training:
            for m in model.modules():
                if isinstance(m, nn.Dropout) and m.p > 0:
                    _has_active_dropout = True
                    break

        _was_training = model.training
        if for_training:
            model.train()  # Training path: bake training-mode behavior
            if int(os.environ.get("RANK", "0")) == 0:
                if _has_active_dropout:
                    print(f">> [IR] Exporting in TRAINING mode (has active Dropout)")
                else:
                    print(f">> [IR] Exporting in TRAINING mode (for correct SDPA kernel selection)")

        # After model.train(), disable training on modules with layerdrop
        # so `if self.training:` guard around random.uniform is never entered.
        if _layerdrop_modules:
            for mod in _layerdrop_modules:
                mod.training = False
            if int(os.environ.get("RANK", "0")) == 0:
                print(f">> [IR] Disabled training on {len(_layerdrop_modules)} "
                      f"module(s) with layerdrop to avoid data-dependent control flow")

        # Patch _update_causal_mask to return None during training-mode export.
        # Models like GPT-2 compute an explicit causal mask at the top of the
        # model (in _update_causal_mask) and pass it to ALL attention layers,
        # creating a global skip connection.  After pipeline split, this mask
        # ends up in one stage but is needed by all stages — the schedule
        # cannot propagate it correctly.  By returning None, each attention
        # layer uses its own internal self.bias buffer for causal masking,
        # keeping the mask local to each layer (no skip connection).
        # LLaMA already returns None in training mode, so this is a no-op
        # for LLaMA.  For inference (for_training=False), skip the patch.
        # NOTE: Some models (e.g., OPT) return a tuple from _update_causal_mask
        # and rely on the external mask for causal attention (no internal
        # self.bias fallback).  For these, the patch would break unpacking
        # and remove causal masking.  We detect this via try/except: if export
        # fails with TypeError (tuple unpack), we revert and retry.
        _orig_update_causal_mask = None
        if for_training:
            for mod in model.modules():
                if hasattr(mod, '_update_causal_mask') and callable(mod._update_causal_mask):
                    _orig_update_causal_mask = (mod, mod._update_causal_mask)
                    mod._update_causal_mask = lambda *a, **kw: None
                    if int(os.environ.get("RANK", "0")) == 0:
                        print(f">> [IR] Patched _update_causal_mask → None "
                              f"(avoid global mask skip connection)")
                    break

        try:
            exported_program = export(
                model, args=(), kwargs=example_inputs, strict=False,
                dynamic_shapes=dynamic_shapes,
            )
        except TypeError as e:
            if _orig_update_causal_mask is not None and (
                    'unpack' in str(e) or 'NoneType' in str(e)):
                # _update_causal_mask returns a tuple (e.g., OPT) — revert patch
                mod_ref, orig_fn = _orig_update_causal_mask
                mod_ref._update_causal_mask = orig_fn
                _orig_update_causal_mask = None
                if int(os.environ.get("RANK", "0")) == 0:
                    print(f">> [IR] _update_causal_mask returns tuple; "
                          f"reverted patch and retrying export")
                exported_program = export(
                    model, args=(), kwargs=example_inputs, strict=False,
                    dynamic_shapes=dynamic_shapes,
                )
            else:
                raise
        model.train(_was_training)  # Restore original mode
        # Restore original use_cache setting
        if _orig_use_cache is not None:
            model.config.use_cache = _orig_use_cache
        # Restore training mode on layerdrop modules
        for mod in _layerdrop_modules:
            mod.training = True
        # Restore _update_causal_mask
        if _orig_update_causal_mask is not None:
            mod_ref, orig_fn = _orig_update_causal_mask
            mod_ref._update_causal_mask = orig_fn

        # Reconstruct module-level graph from nn_module_stack metadata
        # (bypasses unflatten() which is broken for LLaMA — PyTorch #147348)
        self.gm = build_module_graph_from_export(exported_program, model, input_names)

        if int(os.environ.get("RANK", "0")) == 0:
            n_call_module = sum(1 for n in self.gm.graph.nodes if n.op == 'call_module')
            n_call_function = sum(1 for n in self.gm.graph.nodes if n.op == 'call_function')
            n_get_attr = sum(1 for n in self.gm.graph.nodes if n.op == 'get_attr')
            print(f">> [IR] torch.export path: {n_call_module} call_module, "
                  f"{n_call_function} call_function, {n_get_attr} get_attr nodes")


        # Determine model type
        if model.__class__.__name__ in ["ViTForImageClassification"]:
            return self.optimus.model2type["vt"]
        elif (model.__class__.__name__ in _SUPPORTED_MODELS
              or model.__class__.__name__ in _OTHER_MODELS):
            return self.optimus.model2type["hf"]
        elif isinstance(model, nn.Module):
            return self.optimus.model2type["sy"]
        else:
            print(f"Not supported model!")
            sys.exit(1)




    def _get_export_input_names(self, model: nn.Module, use_kv_cache: bool) -> List[str]:
        """Determine input names for torch.export based on model type."""
        if model.__class__.__name__ in ["ViTForImageClassification"]:
            return ['pixel_values']
        elif (model.__class__.__name__ in _SUPPORTED_MODELS
              or model.__class__.__name__ in _OTHER_MODELS):
            input_names = list(model.dummy_inputs.keys())
            if use_kv_cache:
                if 'position_ids' not in input_names:
                    input_names.append('position_ids')
            return input_names
        else:
            # Generic model: try to infer from forward signature
            sig = inspect.signature(model.forward)
            return [
                p.name for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty and p.name != 'self'
            ]


    def _create_example_inputs(self, model: nn.Module, input_names: List[str]) -> dict:
        """Create example input tensors for torch.export.export().

        Returns a dict keyed by parameter name so they can be passed as
        kwargs to export(), ensuring correct parameter-name matching
        regardless of positional ordering in the model's forward().
        """
        inputs = {}

        # Try model.dummy_inputs first (HuggingFace models)
        if hasattr(model, 'dummy_inputs'):
            dummy = model.dummy_inputs

            # Avoid using seq_len == max_position_embeddings as the example.
            # torch.export specializes on boundary values, generating guards
            # like Ne(max_pos, seq_len) that conflict with dynamic_shapes.
            _max_pos = None
            if hasattr(model, 'config'):
                _max_pos = getattr(model.config, 'max_position_embeddings', None)

            for name in input_names:
                if name in dummy:
                    t = dummy[name]
                    if (_max_pos is not None
                            and t.dim() >= 2 and t.size(1) == _max_pos):
                        t = t[:, :min(8, _max_pos)]
                    inputs[name] = t
                elif name == 'position_ids':
                    # Generate position_ids matching both batch and seq_len of input_ids
                    if 'input_ids' in dummy:
                        batch_size = dummy['input_ids'].size(0)
                        seq_len = dummy['input_ids'].size(1)
                    else:
                        batch_size, seq_len = 1, 8
                    inputs[name] = (
                        torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
                    )
                else:
                    print(f"[IR] Warning: input '{name}' not found in dummy_inputs")

            # torch.export specializes batch=1 as a constant.  Expand to
            # batch>=2 so the batch dimension is treated as truly dynamic.
            for name in list(inputs.keys()):
                t = inputs[name]
                if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.size(0) == 1:
                    inputs[name] = t.expand(2, *(-1,) * (t.dim() - 1)).contiguous()

            if inputs:
                return inputs

        # Fallback: synthesize minimal inputs (batch=2 to avoid specialization)
        # Try to infer input dimension from the first Linear layer
        _first_linear_in = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                _first_linear_in = m.in_features
                break

        for name in input_names:
            if name == 'input_ids':
                inputs[name] = torch.randint(0, 1000, (2, 8))
            elif name == 'attention_mask':
                inputs[name] = torch.ones(2, 8, dtype=torch.long)
            elif name == 'position_ids':
                inputs[name] = torch.arange(8).unsqueeze(0).expand(2, -1).contiguous()
            elif name == 'pixel_values':
                inputs[name] = torch.randn(2, 3, 224, 224)
            else:
                # Generic tensor input — use first Linear's in_features if available
                in_dim = _first_linear_in or 64
                inputs[name] = torch.randn(2, in_dim)
        return inputs


    def split_IR(self, model: nn.Module, method, num_stage):

        if method not in [ "simple", "llama-tp-split", ]:
            print(f"Not supported split method!")
            sys.exit(1)

        # TODO: TO DELETE
        #if int(os.environ.get("RANK", "0")) == 0:
        #    print(f">> ------------------ FX graph (pre) --------------------------------")
        #    for n in self.gm.graph.nodes:
        #        print(f"n.op:{n.op}, n.name:{n.name}, n.target:{n.target}, n.args:{n.args}, n.all_input_nodes:{n.all_input_nodes}")
        #    print(f">> ------------------------------------------------------------")

        if method == "simple":
            submods = self.simple_split(model, num_stage)

        elif method == "llama-tp-split":
            submods = self.llama_tp_split(model, num_stage)

        # TODO: add new split method
        #elif method == ...
        #

            
        self.check_last_submods(submods, num_stage)

        self.model_ir.append(submods)

        if int(os.environ.get("RANK", "0")) == 0:
            print(f">> ------------------ FX graph (split) --------------------------------")
            n_split_cm = 0
            n_split_cf = 0
            n_split_ga = 0
            for n in self.model_ir[0].graph.nodes:
                print(f"n.op:{n.op}, n.name:{n.name}, n.target:{n.target}, n.args:{n.args}, n.all_input_nodes:{n.all_input_nodes}")
                if n.op == 'call_module':
                    n_split_cm += 1
                elif n.op == 'call_function':
                    n_split_cf += 1
                elif n.op == 'get_attr':
                    n_split_ga += 1
            print(f">> Split graph summary: {n_split_cm} call_module, "
                  f"{n_split_cf} call_function, {n_split_ga} get_attr")
            if n_split_cf > 0 or n_split_ga > 0:
                print(f">> WARNING: {n_split_cf} call_function and {n_split_ga} get_attr "
                      f"nodes at top level of split graph (not inside submodules)")
            print(f">> ------------------------------------------------------------")



    def simple_split(self, module, num_stage):
        length = self.gm.graph.nodes.__len__()

        modcnt = 0
        for n in self.gm.graph.nodes:
            if n.op == 'call_module':
                modcnt = modcnt + 1
        
        segment = modcnt // num_stage
        print(f"length:{length}, modcnt:{modcnt}, num_stage:{num_stage}, segment:{segment}")


        ## simple assert
        assert length >= num_stage, f"Model length:{length} is smaller than # of workers:{num_stage}"
        
        self.last_flag = False

        def part_fn(node):
            last_idx, last_name = self.metadata_range[-1]
            
            if self.last_flag == True:
                idx = last_idx
                #print(f" part_fn:  node.name:{node.name}, --> {idx}")
                return idx
            
            idx = 0
            
            cur = node
            while cur.name != last_name:
                for i, m_name in self.metadata_range:
                    if cur.name == m_name:
                        idx = i
                        #print(f" part_fn:  node.name:{node.name}, m_name:{m_name}, --> {idx}")
                        return idx
            
                cur = cur._next
            
            if cur.name == last_name:
                idx = last_idx
                self.last_flag = True
            
            #print(f" part_fn:  node.name:{node.name}, --> {idx}")
            return idx
            

        k, cnt = 0, 0
        for n in self.gm.graph.nodes:
            if n.op == 'call_module':
                cnt = cnt + 1
        
            if cnt == segment:
                self.metadata_range.append((k, n.name))
                k = k + 1
                cnt = 0
        
            if k > num_stage - 1:
                break
        
        if len(self.metadata_range) <  num_stage:
            self.metadata_range.append((k, n.name))

        if int(os.environ.get("RANK", "0")) == 0:
            print(f" ------------------------------------------------------------")
            print(f"  rank:{self.optimus.tpl.rank},  first metadata_range: {self.metadata_range}")
            print(f" ------------------------------------------------------------")

        submodules = split_module(self.gm, module, part_fn, keep_original_order=True)


        def move_parameters(split_graph_module, user_target, parameter_value, use_index, _buffer):

            assert isinstance(parameter_value, torch.Tensor), f"Not torch.Tensor but {type(parameter_value)} received."

            target = split_graph_module.get_submodule(user_target)
            new_parameter_name = f"moved_{node.target.replace('.', '_')}"

            assert not hasattr(target, new_parameter_name), f"{user_target} has parameter[{new_parameter_name}]"

            if _buffer:
                target.register_buffer(new_parameter_name, parameter_value)
            else:
                setattr(target, new_parameter_name, parameter_value)

            placeholder_cnt = 0
            for snode in target.graph.nodes:
                if snode.op == "placeholder":
                    if placeholder_cnt == use_index:
                        with target.graph.inserting_before(snode):
                            get_attr = target.graph.get_attr(new_parameter_name)
                            snode.replace_all_uses_with(get_attr)
                            target.graph.erase_node(snode)
                    placeholder_cnt += 1

            target.graph.lint()
            target.recompile()

            return get_attr

                    
        def remove_reference(node, user, delete_node=True):
            assert len(user.kwargs) == 0
            use_indices = [i for i, arg in enumerate(user.args) if arg == node]
            assert len(use_indices) == 1
            args_copy = list(user.args)
            args_copy.pop(use_indices[0])
            user.args = tuple(args_copy)
            if delete_node:
                node.graph.erase_node(node)

            return use_indices[0]


        # Move parameters from get_attr nodes into submodules.
        # With torch.export path, get_attr nodes for buffers used by inline
        # call_function ops (e.g., RoPE inv_freq) should be skipped.
        has_get_attr = any(n.op == "get_attr" for n in submodules.graph.nodes)
        if has_get_attr:
            remove_candidates = list()
            for node in submodules.graph.nodes:
                if node.op == "get_attr" and len(node.users) == 1:
                    user = list(node.users)[0]
                    if user.op == "call_function" or user.op == "call_method":
                        # Inline op using a buffer — no submodule to move into
                        continue
                    assert user.op == "call_module", \
                        f"get_attr '{node.target}' consumed by unexpected op: {user.op}"

                    atoms = node.target.split(".")
                    module_itr = submodules
                    for atom in atoms[:-1]:
                        module_itr = getattr(module_itr, atom)
                    parameter_value = getattr(module_itr, atoms[-1])

                    # Skip synthetic modules (e.g., submod_N from torch.export)
                    if not isinstance(parameter_value, torch.Tensor):
                        continue

                    _buffer = atoms[-1] in module_itr._buffers
                    use_index = remove_reference(node, user)

                    move_parameters(submodules, user.target, parameter_value, use_index, _buffer)

                    remove_candidates.append((module_itr, atoms))

            for module_itr, atoms in remove_candidates:
                delattr(module_itr, atoms[-1])
            submodules.graph.lint()
            submodules.recompile()

        self.metadata_range = []

        cnt = 0
        for n in submodules.graph.nodes:
            if n.op == 'call_module':
                self.metadata_range.append((cnt, n.name))
                cnt = cnt + 1

        print(f" ------------------------------------------------------------")
        print(f"  rank:{self.optimus.tpl.rank},  second metadata_range: {self.metadata_range}")
        print(f" ------------------------------------------------------------")

        assert len(self.metadata_range) == num_stage

        return submodules


    def llama_tp_split(self, module, num_stage):

        assert module.__class__.__name__.startswith("Llama")

        num_layers = len(module.model.layers)
        #num_blocks = len(module.model.layers) + 2
        num_blocks = num_layers + 2
        last_layer = num_layers - 1

        assert num_layers >= num_stage, f"# of layers[{num_layers}] is smaller than # of stages:{num_stage}"

        layers_per_stage = [ (i * num_blocks) // num_stage for i in range(num_stage + 1) ]

        stage_layers = [ list(range(layers_per_stage[i], layers_per_stage[i + 1])) for i in range(num_stage) ]
        if num_stage > 2:
            #if len(stage_layers[-1]) > len(stage_layers[-2]):
            if len(stage_layers[-1]) > len(stage_layers[-2]) and stage_layers[-1][0] != last_layer:
                stage_layers[-2] = stage_layers[-2] + [stage_layers[-1][0]]
                stage_layers[-1] = stage_layers[-1][1:]

        if self.optimus.is_first_stage():
            print(f" num_stage: {num_stage}")
            print(f" num_blocks = {num_blocks}")
            print(f" layers_per_stage = {layers_per_stage}")
            print(f" stage_layers = {stage_layers}")

            # Example :
            #   num_stage = 4
            #   layers_per_stage = [0, 3, 6, 9, 12]
            #   stage_layers = [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]

        
        node_p = None
        layer_id = 0
        status = 0
        k = 0
        for n in self.gm.graph.nodes:
            if n.op == 'call_module' and isinstance(n.target, str):
                parts = n.target.split('.')

                if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
                    layer_id = int(parts[2])
                    if layer_id in stage_layers[k]:
                        status = 1
                        node_p = n
                    else:
                        status = 2


            if status == 2:
                self.metadata_range.append((k, node_p.name))
                k = k + 1
                status = 0

            if k > num_stage - 1:
                break

        if len(self.metadata_range) <  num_stage:
            self.metadata_range.append((k, node_p.name))

        self.last_flag = False

        def part_fn(node):
            last_idx, last_name = self.metadata_range[-1]

            if self.last_flag == True:
                idx = last_idx
                #print(f" part_fn:  node.name:{node.name}, --> {idx}")
                return idx

            idx = 0

            cur = node
            while cur.name != last_name:
                for i, m_name in self.metadata_range:
                    if cur.name == m_name:
                        idx = i
                        #print(f" part_fn:  node.name:{node.name}, m_name:{m_name}, --> {idx}")
                        return idx

                cur = cur._next

            if cur.name == last_name:
                idx = last_idx
                self.last_flag = True

            #print(f" part_fn:  node.name:{node.name}, --> {idx}")
            return idx

        if int(os.environ.get("RANK", "0")) == 0:
            print(f" ------------------------------------------------------------")
            print(f"  rank:{self.optimus.tpl.rank},  first metadata_range: {self.metadata_range}")
            print(f" ------------------------------------------------------------")

        submodules = split_module(self.gm, module, part_fn, keep_original_order=True)

        def move_parameters(split_graph_module, user_target, parameter_value, use_index, _buffer):

            assert isinstance(parameter_value, torch.Tensor), f"Not torch.Tensor but {type(parameter_value)} received."

            target = split_graph_module.get_submodule(user_target)
            new_parameter_name = f"moved_{node.target.replace('.', '_')}"

            assert not hasattr(target, new_parameter_name), f"{user_target} has parameter[{new_parameter_name}]"

            if _buffer:
                target.register_buffer(new_parameter_name, parameter_value)
            else:
                setattr(target, new_parameter_name, parameter_value)

            placeholder_cnt = 0
            for snode in target.graph.nodes:
                if snode.op == "placeholder":
                    if placeholder_cnt == use_index:
                        with target.graph.inserting_before(snode):
                            get_attr = target.graph.get_attr(new_parameter_name)
                            snode.replace_all_uses_with(get_attr)
                            target.graph.erase_node(snode)
                    placeholder_cnt += 1

            target.graph.lint()
            target.recompile()

            return get_attr

        def remove_reference(node, user, delete_node=True):
            assert len(user.kwargs) == 0
            use_indices = [i for i, arg in enumerate(user.args) if arg == node]
            assert len(use_indices) == 1
            args_copy = list(user.args)
            args_copy.pop(use_indices[0])
            user.args = tuple(args_copy)
            if delete_node:
                node.graph.erase_node(node)

            return use_indices[0]

        has_get_attr = any(n.op == "get_attr" for n in submodules.graph.nodes)
        if has_get_attr:
            remove_candidates = list()
            for node in submodules.graph.nodes:
                if node.op == "get_attr" and len(node.users) == 1:
                    user = list(node.users)[0]
                    if user.op == "call_function" or user.op == "call_method":
                        # Inline op using a buffer — no submodule to move into
                        continue
                    assert user.op == "call_module", \
                        f"get_attr '{node.target}' consumed by unexpected op: {user.op}"

                    atoms = node.target.split(".")
                    module_itr = submodules
                    for atom in atoms[:-1]:
                        module_itr = getattr(module_itr, atom)
                    parameter_value = getattr(module_itr, atoms[-1])

                    # Skip synthetic modules (e.g., submod_N from torch.export)
                    if not isinstance(parameter_value, torch.Tensor):
                        continue

                    _buffer = atoms[-1] in module_itr._buffers
                    use_index = remove_reference(node, user)

                    move_parameters(submodules, user.target, parameter_value, use_index, _buffer)

                    remove_candidates.append((module_itr, atoms))

            for module_itr, atoms in remove_candidates:
                delattr(module_itr, atoms[-1])
            submodules.graph.lint()
            submodules.recompile()

        self.metadata_range = []

        cnt = 0
        for n in submodules.graph.nodes:
            if n.op == 'call_module':
                self.metadata_range.append((cnt, n.name))
                cnt = cnt + 1

        print(f" ------------------------------------------------------------")
        print(f"  rank:{self.optimus.tpl.rank},  second metadata_range: {self.metadata_range}")
        print(f" ------------------------------------------------------------")

        assert len(self.metadata_range) == num_stage

        return submodules


        
    def check_last_submods(self, submods, num_stage):
        gmodule_cnt = 0
        mod_cnt = 0
        for submod in submods.modules():
            if isinstance(submod, fx.GraphModule):
                gmodule_cnt = gmodule_cnt + 1
                last_submod = submod
                continue


        assert gmodule_cnt > num_stage, f"GraphModule #:[{gmodule_cnt}] must have more than the number of stages #:[{num_stage}]"

        for node in last_submod.graph.nodes:
            #print(f"-- node.op:{node.op}, node.name:{node.name}, node.target:{node.target}, node.all_input_nodes:{node.all_input_nodes}")
            if node.op == 'call_module' and node.target != 'loss_fn':
                mod_cnt = mod_cnt + 1

        #print(f">>> GraphModule cnt:{gmodule_cnt},  Last GraphModule's  mod_cnt ==> {mod_cnt}")

        assert mod_cnt > 0, f"Last partition has {mod_cnt} modules. It must have more than 0 modules"



    # analyze IR graph and find the cross-layer referenced nodes
    def cross_reference_analyze(self, stage, g:fx.Graph):
    
        if stage == 0:
            return
    
        from_, to_ = self.get_range(stage, g)
    
        #logging.debug(f" ***** stage:{stage} >>  from_:{from_.name}, to_:{to_.name}")
        print(f" ***** stage:{stage} >>  from_:{from_.name}, to_:{to_.name}")
    
        cur = to_
        while (cur != from_) or (stage > 0 and cur == from_):
    
            # in process check - backward direction
    
            #for _, target_ in enumerate(cur.all_input_nodes):
            for i, target_ in enumerate(cur.all_input_nodes):
                if cur.name == "loss_fn" and i > 0:
                    break
                referenced_in = False
                referenced_out = False
    
                inner = cur._prev
                if inner != from_._prev:
                    while (inner != from_) or (stage > 0 and inner == from_):
                        if inner.name == target_.name:
                            #logging.debug(f" [cross_reference_analyze] ({target_.name}) referenced in current stage:{stage} !")
                            referenced_in = True
                            break
    
                        if inner == from_:
                            break
    
                        inner = inner._prev
    
                if referenced_in == True:
                    continue
    
                if referenced_in == False:
    
                    # output process check - forward direction
    
                    stage_ = 0
                    split_node_name = self.metadata_range[stage_][1]
    
                    for k in g.nodes:
                        first_node = k
                        break
    
                    outer = first_node
                    while outer != from_: 
                        # DEBUG
                        if outer.name == target_.name:
                            logging.info(f" [cross_reference_analyze] ({target_.name}) referenced in outer stage:{stage_} !!")
    
                            if target_.name not in self.special_nodes:
                                self.special_nodes[target_.name] = (stage_, stage)  # { node_name : {stage#, needed-by-stage#),}
                            referenced_out = True
                            break
    
                        if outer.name == split_node_name:
                            stage_ = stage_ + 1
                            split_node_name = self.metadata_range[stage_][1]
    
                        outer = outer._next
    
    
                if referenced_out == False:
                    logging.critical(f"[Error] cannot handle this case: {target_.name} !!!")
                    sys.exit(1)
    
            if cur == from_:
                break
    
            cur = cur._prev



    def get_range(self, stage, g:fx.Graph) -> (Node, Node):
     
        if stage == 0:
            from_node_name = "-1"
            for n in g.nodes:
                if n.op == 'placeholder':
                    from_node_name = n.name
                    logging.debug(f">>>> get_range: n.op == 'placeholder' --> from_node_name:{from_node_name}")
                break
        else:
            from_node_name = self.metadata_range[stage-1][1]
     
        to_node_name = self.metadata_range[stage][1]
    
        for n in g.nodes:
            if from_node_name == "-1":
                from_node = n
                break
     
            if n.name == from_node_name:
                from_node = n
                break
     
        for n in reversed(g.nodes):
            if n.name == to_node_name :
                to_node = n
                break
     
        if stage == 0:
            return (from_node, to_node)
        else:
            return (from_node._next, to_node)


    def setup_submod(self, stage, rank):

        for n, m in self.model_ir[0].named_children():
            if n == f"submod_{stage}" and isinstance(m, GraphModule):
                self.optimus.run_info.name = n
                self.optimus.run_info.submod = m
                break

        if self.optimus.run_info.name is None:
            print(f"ERROR: Not found name(submod_{stage})")
            sys.exit(0)

        #print(f" ## Rank:{rank}, name:{self.name}")

        for n in self.model_ir[0].graph.nodes:
            if n.name == self.optimus.run_info.name:
                self.optimus.run_info.node = n
                break

        if self.optimus.run_info.node is None:
            print(f"ERROR: Not found node({self.name})")
            sys.exit(0)


    def build_getitem_dic(self):
        for node in self.model_ir[0].graph.nodes:
            if node.op == 'call_function' and node.name.startswith("getitem"):
                self.optimus.run_info.getitem_dic[node.name] = (node.args[0].name, node.args[1])

    #def print_graph(self, ir, rank):
    def print_graph(self, rank):
        print(f" # rank = {rank}, metadata_range:{self.metadata_range}")
        for node in self.model_ir[0].graph.nodes:
            print(f"-- node.op:{node.op}, node.name:{node.name}, node.target:{node.target}, node.args:{node.args}, node.all_input_nodes:{node.all_input_nodes}")


    def get_output_node(self):
        for n in reversed(self.model_ir[0].graph.nodes):
            if n.op == 'output':
                return n


    def delete_param(self, mod, name):
        for param_name, param in mod.named_parameters():
            t = getattr(mod, param_name)
            setattr(mod, param_name, None)
            del t

    def has_child(self, mod):
        num_children = 0
        if mod is not None:
            num_children = len(list(mod.children()))
        return num_children

    def partially_matched(self, name):
        for string in self.reference:
            if name in string:
                return True
        return False

    def delete_module(self, root, module_node):

        module_name = module_node.name
        module_target_name = str(module_node.target)
        if len(module_name) > len(module_target_name):
            self.reference.append(module_target_name)

        get_attr_flag = False
        if len(module_name) == len(module_target_name) and not self.partially_matched(module_target_name):
            target_atoms = module_target_name.split('.')
            attr_itr = root
            if module_node.op == 'get_attr':
                if target_atoms[-1] == "weight" or target_atoms[-1] == "bias":
                    target_atoms = target_atoms[:-1]
                    get_attr_flag = True
                    if self.partially_matched(str('.'.join(target_atoms))):
                        return
                else:
                    return

            for i , atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(\
                        f"Node referenced nonexistant target {'.'.join(target_atoms[:i])} ... [{target_atoms}]")
                parent_ = attr_itr
                attr_itr = getattr(attr_itr, atom)

            #if self.has_child(attr_itr) == 0:
            #if get_attr_flag == True or self.has_child(attr_itr) == 0:
            if True:
                self.delete_param(attr_itr, atom)
                delattr(parent_, atom)
                del attr_itr
                #print(f">>> rank:{self.optimus.tpl.rank}, delete module:target:{module_target_name} ## atom:{atom},op:{module_node.op}")
                self.reference.append(module_target_name)
            else:
                print(f">>> delete_module:{module_name} requested, but {module_name} has child")



    def delete_intermediate_module(self, module_name):
        del_submod = self.model_ir[0].get_submodule(module_name)

        for n in del_submod.graph.nodes:
            if n.op == 'call_module':
                self.delete_module(del_submod, n)

    def print_module_shape_before_parallel(self):
        if self.optimus.get_rank() == 0:
            file_name = f"tensor_info_before.txt"
            with open(file_name, "w") as f:
                for stage in range(self.optimus.tpl.num_stage):
                    param_info = []
                    mod_name = f"submod_{stage}"
                    mod = self.model_ir[0].get_submodule(mod_name)

                    for name, param in mod.named_parameters():
                        param_info.append(f"Name: {name}, Shape: {param.shape}")
                    f.write("\n".join(param_info))
            print(f"================================================")

    def print_module_shape_after_parallel(self):
        param_info = []
        file_name = f"tensor_info_rank_{self.optimus.get_rank()}.txt"

        for name, param in self.optimus.run_info.submod.named_parameters():
            param_info.append(f"Name: {name}, Shape: {param.shape}")
        with open(file_name, "w") as f:
                f.write("\n".join(param_info))
        print(f"================================================")


    def clean_module_memory(self):
        self.reference = []

        for stage in reversed(range(self.optimus.tpl.num_stage)):
            if stage != self.optimus.tpl.get_stage():
                del_name = f"submod_{stage}"
                self.delete_intermediate_module(del_name)

        #print(f"[rank:{self.optimus.tpl.rank}] reference:{self.reference}")
        self.reference = None

        self.model_ir = []
        self.gm = None

        gc.collect()
        torch.cuda.empty_cache()


