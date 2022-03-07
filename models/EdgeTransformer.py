import torch
import torch.nn as nn
from typing import Optional, Any, Tuple

from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules import Linear, Module, MultiheadAttention, LayerNorm, Dropout, ModuleList
import copy

from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import _LinearWithBias

import warnings
from torch.nn.functional import linear, softmax, dropout, pad


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoderEdge(Module):

    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoderEdge, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, restrict_tgt: Optional[Tensor]=None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, self_relative_matrix: Optional[Tensor]=None,
                cross_relative_matrix: Optional[Tensor]=None, relative_style = "add"):
        output = tgt

        for mod in self.layers:
            if restrict_tgt is None:
                output = mod(output, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask,
                             self_relative_matrix = self_relative_matrix,
                             cross_relative_matrix = cross_relative_matrix,
                             relative_style = relative_style)
            else:
                output = mod(output, memory, tgt_mask=tgt_mask,
                             memory_mask=memory_mask, restrict_tgt=restrict_tgt,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             memory_key_padding_mask=memory_key_padding_mask,
                             self_relative_matrix=self_relative_matrix,
                             cross_relative_matrix=cross_relative_matrix,
                             relative_style=relative_style)
                
        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderEdge(Module):

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoderEdge, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                src_relative_matrix: Optional[Tensor] = None, relative_style = "add") -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask,
                         src_relative_matrix=src_relative_matrix, relative_style=relative_style)

        if self.norm is not None:
            output = self.norm(output)

        return output


class RelativeMultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(RelativeMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(RelativeMultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                relative_matrix: Optional[Tensor] = None, relative_style = "add"
                ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shapes for inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
          source sequence length.

          If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
          length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
          the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

    Shapes for outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return self.multi_head_attention_forward_graph(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, relative_matrix=relative_matrix, relative_style=relative_style)
        else:
            return self.multi_head_attention_forward_graph(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, relative_matrix=relative_matrix, relative_style=relative_style)

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
    
    def multi_head_attention_forward_graph(self, query: Tensor, key: Tensor,
                value: Tensor,
                embed_dim_to_check: int,
                num_heads: int,
                in_proj_weight: Tensor,
                in_proj_bias: Tensor,
                bias_k: Optional[Tensor],
                bias_v: Optional[Tensor],
                add_zero_attn: bool,
                dropout_p: float,
                out_proj_weight: Tensor,
                out_proj_bias: Tensor,
                training: bool = True,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[Tensor] = None,
                use_separate_proj_weight: bool = False,
                q_proj_weight: Optional[Tensor] = None,
                k_proj_weight: Optional[Tensor] = None,
                v_proj_weight: Optional[Tensor] = None,
                static_k: Optional[Tensor] = None,
                static_v: Optional[Tensor] = None,
                relative_matrix: Optional[Tensor] = None,
                relative_style="add", register_hook=False
        ):
            tgt_len, bsz, embed_dim = query.size()
    
            head_dim = embed_dim // num_heads
            scaling = float(head_dim) ** -0.5
    
            if not use_separate_proj_weight:
                if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
                    # self-attention
                    q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
        
                elif key is value or torch.equal(key, value):
                    # encoder-decoder attention
                    # This is inline in_proj function with in_proj_weight and in_proj_bias
                    _b = in_proj_bias
                    _start = 0
                    _end = embed_dim
                    _w = in_proj_weight[_start:_end, :]
                    if _b is not None:
                        _b = _b[_start:_end]
                    q = linear(query, _w, _b)
            
                    if key is None:
                        assert value is None
                        k = None
                        v = None
                    else:
                
                        # This is inline in_proj function with in_proj_weight and in_proj_bias
                        _b = in_proj_bias
                        _start = embed_dim
                        _end = None
                        _w = in_proj_weight[_start:, :]
                        if _b is not None:
                            _b = _b[_start:]
                        k, v = linear(key, _w, _b).chunk(2, dim=-1)
        
                else:
                    # This is inline in_proj function with in_proj_weight and in_proj_bias
                    _b = in_proj_bias
                    _start = 0
                    _end = embed_dim
                    _w = in_proj_weight[_start:_end, :]
                    if _b is not None:
                        _b = _b[_start:_end]
                    q = linear(query, _w, _b)
            
                    # This is inline in_proj function with in_proj_weight and in_proj_bias
                    _b = in_proj_bias
                    _start = embed_dim
                    _end = embed_dim * 2
                    _w = in_proj_weight[_start:_end, :]
                    if _b is not None:
                        _b = _b[_start:_end]
                    k = linear(key, _w, _b)
            
                    # This is inline in_proj function with in_proj_weight and in_proj_bias
                    _b = in_proj_bias
                    _start = embed_dim * 2
                    _end = None
                    _w = in_proj_weight[_start:, :]
                    if _b is not None:
                        _b = _b[_start:]
                    v = linear(value, _w, _b)
            else:
                q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
                len1, len2 = q_proj_weight_non_opt.size()
                assert len1 == embed_dim and len2 == query.size(-1)
        
                k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
                len1, len2 = k_proj_weight_non_opt.size()
                assert len1 == embed_dim and len2 == key.size(-1)
        
                v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
                len1, len2 = v_proj_weight_non_opt.size()
                assert len1 == embed_dim and len2 == value.size(-1)
        
                if in_proj_bias is not None:
                    q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
                    k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim: (embed_dim * 2)])
                    v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
                else:
                    q = linear(query, q_proj_weight_non_opt, in_proj_bias)
                    k = linear(key, k_proj_weight_non_opt, in_proj_bias)
                    v = linear(value, v_proj_weight_non_opt, in_proj_bias)
            q = q * scaling
    
            if attn_mask is not None:
                assert (
                        attn_mask.dtype == torch.float32
                        or attn_mask.dtype == torch.float64
                        or attn_mask.dtype == torch.float16
                        or attn_mask.dtype == torch.uint8
                        or attn_mask.dtype == torch.bool
                ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
                if attn_mask.dtype == torch.uint8:
                    warnings.warn(
                        "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                    attn_mask = attn_mask.to(torch.bool)
        
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                    if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                        raise RuntimeError("The size of the 2D attn_mask is not correct.")
                elif attn_mask.dim() == 3:
                    if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                        raise RuntimeError("The size of the 3D attn_mask is not correct.")
                else:
                    raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
                # attn_mask's dim is 3 now.
    
            # convert ByteTensor key_padding_mask to bool
            if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
                )
                key_padding_mask = key_padding_mask.to(torch.bool)
    
            if bias_k is not None and bias_v is not None:
                if static_k is None and static_v is None:
                    k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
                    v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
                    if attn_mask is not None:
                        attn_mask = pad(attn_mask, (0, 1))
                    if key_padding_mask is not None:
                        key_padding_mask = pad(key_padding_mask, (0, 1))
                else:
                    assert static_k is None, "bias cannot be added to static key."
                    assert static_v is None, "bias cannot be added to static value."
            else:
                assert bias_k is None
                assert bias_v is None
    
            q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
            if k is not None:
                k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
            if v is not None:
                v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    
            if static_k is not None:
                assert static_k.size(0) == bsz * num_heads
                assert static_k.size(2) == head_dim
                k = static_k
    
            if static_v is not None:
                assert static_v.size(0) == bsz * num_heads
                assert static_v.size(2) == head_dim
                v = static_v
    
            src_len = k.size(1)
    
            if key_padding_mask is not None:
                assert key_padding_mask.size(0) == bsz
                assert key_padding_mask.size(1) == src_len
    
            if add_zero_attn:
                src_len += 1
                k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
                v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
                if attn_mask is not None:
                    attn_mask = pad(attn_mask, (0, 1))
                if key_padding_mask is not None:
                    key_padding_mask = pad(key_padding_mask, (0, 1))
    
            attn_output_weights = torch.bmm(q, k.transpose(1, 2))
            
            dot_product_weights = attn_output_weights.clone()
            
            if relative_matrix is not None:
                if relative_style == "add":
                    attn_output_weights = attn_output_weights + relative_matrix
                elif relative_style == "mul":
                    attn_output_weights = attn_output_weights * relative_matrix
    
            assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
            
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_output_weights.masked_fill_(attn_mask, float("-inf"))
                else:
                    attn_output_weights += attn_mask
    
            if key_padding_mask is not None:
                attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
                attn_output_weights = attn_output_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float("-inf"),
                )
                attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
    
            attn_output_weights = softmax(attn_output_weights, dim=-1)
            attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)
    
            attn_output = torch.bmm(attn_output_weights, v)
            assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
            
            if need_weights:
                # average attention weights over heads
                attn = dot_product_weights.view(bsz, num_heads, tgt_len, src_len)
                attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
                return attn_output, attn_output_weights.sum(dim=1) / num_heads, attn.sum(dim=1) / num_heads
            else:
                return attn_output, None, None


class EdgeTransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(EdgeTransformerEncoderLayer, self).__init__()
        self.self_attn = RelativeMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(EdgeTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                            src_relative_matrix: Optional[Tensor] = None, relative_style="add"
                ) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, relative_matrix=src_relative_matrix,
                              relative_style=relative_style
                              )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class EdgeTransformerLightDecoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", viz_weight=False):
        super(EdgeTransformerLightDecoderLayer, self).__init__()
        
        self.multihead_attn = RelativeMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)
        
        self.viz_weight = viz_weight
    
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(EdgeTransformerLightDecoderLayer, self).__setstate__(state)

    
    def forward(self, tgt: Tensor, memory: Tensor, tgt_key_padding_mask = None, tgt_mask = None,
                memory_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                cross_relative_matrix: Optional[Tensor] = None, self_relative_matrix = None,
                relative_style="add"):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2, cross_attn_weight, cross_attn = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                                                  key_padding_mask=memory_key_padding_mask,
                                                                  relative_matrix=cross_relative_matrix,
                                                                  relative_style=relative_style)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class EdgeTransformerRestrictDecoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", viz_weight=False):
        super(EdgeTransformerRestrictDecoderLayer, self).__init__()
        self.self_attn = RelativeMultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = RelativeMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)
        
        self.viz_weight = viz_weight
    
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(EdgeTransformerRestrictDecoderLayer, self).__setstate__(state)
    
    def forward(self, tgt: Tensor, memory: Tensor,
                restrict_tgt: Tensor = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                self_relative_matrix: Optional[Tensor] = None, cross_relative_matrix: Optional[Tensor] = None,
                relative_style="add"):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        import pdb
        pdb.set_trace()
        tgt2, self_attn_weight, self_attn = self.self_attn(tgt, restrict_tgt, restrict_tgt, attn_mask=tgt_mask,
                                                           key_padding_mask=tgt_key_padding_mask,
                                                           relative_matrix=self_relative_matrix,
                                                           relative_style=relative_style)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, cross_attn_weight, cross_attn = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                                                  key_padding_mask=memory_key_padding_mask,
                                                                  relative_matrix=cross_relative_matrix,
                                                                  relative_style=relative_style)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        if self.viz_weight:
            return tgt, self_attn_weight, cross_attn_weight, self_attn, cross_attn
        else:
            return tgt



class EdgeTransformerDecoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", viz_weight=False):
        super(EdgeTransformerDecoderLayer, self).__init__()
        self.self_attn = RelativeMultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = RelativeMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
        self.viz_weight = viz_weight

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(EdgeTransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                self_relative_matrix: Optional[Tensor]=None, cross_relative_matrix: Optional[Tensor]=None,
                relative_style="add"):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2, self_attn_weight, self_attn = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              relative_matrix = self_relative_matrix,
                              relative_style=relative_style)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, cross_attn_weight, cross_attn = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   relative_matrix = cross_relative_matrix,
                                   relative_style=relative_style)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
