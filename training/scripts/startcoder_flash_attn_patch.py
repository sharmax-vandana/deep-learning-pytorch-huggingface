import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from einops import rearrange
from typing import Optional, Tuple, Union
import logging 
import transformers
from torch import nn

def unsqueeze_and_expand(proj, unsqueeze_dim=2, num_head=12):
    proj = torch.unsqueeze(proj, unsqueeze_dim)
    b, sq, _, hn = proj.shape
    proj = proj.expand((b, sq, num_head, hn))
    return proj

def forward(
    self,
    hidden_states: torch.Tensor,
    layer_past: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[
    Tuple[torch.Tensor, Optional[torch.Tensor]],
    Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]],
]:
    """Implements the multihead softmax attention.
    Arguments
    ---------
        q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
    """
    if encoder_hidden_states is not None:
        if not hasattr(self, "q_attn") or not self.is_cross_attention:
            raise ValueError(
                "If class is used as cross attention, the weights `q_attn` have to be defined. "
                "Please make sure to instantiate class with `GPTBigCodeAttention(..., is_cross_attention=True)`."
            )

        query = self.q_attn(hidden_states)
        key_value = self.c_attn(encoder_hidden_states)
        attention_mask = encoder_attention_mask
    elif self.multi_query:
        query, key_value = self.c_attn(hidden_states).split((self.embed_dim, 2 * self.kv_dim), dim=2)
    else:
        # Note: We split as (self.num_heads, 3, self.head_dim) instead of (3, self.num_heads, self.head_dim),
        # i.e., the memory layout is not the same as GPT2.
        # This makes the concatenation with past_key_value more efficient.
        query, key_value = (
            self.c_attn(hidden_states)
            .view(*hidden_states.shape[:2], self.num_heads, 3 * self.head_dim)
            .transpose(1, 2)
            .split((self.head_dim, 2 * self.head_dim), dim=3)
        )

    if layer_past is not None:
        key_value = torch.cat((layer_past, key_value), dim=-2)
    present = key_value if use_cache else None

    key, value = key_value.split((self.head_dim, self.head_dim), dim=-1)

    if self.multi_query:
        batch_size, query_length, _ = query.shape
        query = query.reshape(batch_size, query_length, self.num_heads, self.head_dim)
        key, value = [unsqueeze_and_expand(x, unsqueeze_dim=2, num_head=self.num_heads) for x in [key, value]]
    else:
        query, key, value = [rearrange(x, "b h s d -> b s h d") for x in [query, key, value]]
    query, key, value = [x.to(torch.bfloat16) for x in [query, key, value]]
    # print(f"{query.shape=} {key.shape=} {value.shape=}")
    attn_output = core_attention_flash(query, key, value,training=self.training,dropout_p=0.1) # copied from https://huggingface.co/bigcode/starcoder/blob/main/config.json#L9
    attn_weights = None
    attn_output = self.c_proj(attn_output.reshape(hidden_states.shape))
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        raise NotImplementedError("`output_attentions` is not supported when `use_flash_attn` is True")

    return outputs  # a, present, (attentions)
      
      
def core_attention_flash(q, k, v,training=None,causal=True,dropout_p=None,softmax_scale=None):
    """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
    """
    assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q, k, v)))
    assert all((i.is_cuda for i in (q, k, v)))

    batch_size, seqlen_q = q.shape[0], q.shape[1]
    seqlen_k = k.shape[1]

    q, k, v = [rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v]]
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)

    if training:
        # during training q,k,v always have same seqlen
        assert seqlen_k == seqlen_q

        is_causal = causal
        cu_seqlens_k = cu_seqlens_q
        dropout_p = dropout_p
    else:
        # turn off FA causal mask after first inference autoregressive iteration
        # only on first autoregressive step q,k,v have same seqlen
        is_causal = causal and (seqlen_q == seqlen_k)
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=q.device
        )
        dropout_p = 0

    output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlen_q,
        seqlen_k,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=is_causal,
    )

    output = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
    return output

# alternative https://github.com/huggingface/transformers/commit/ee81bf5aee0d65f005d157c013777e3d27d8d6bf
# alternative https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py
def replace_starcoder_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        logging.warning(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    print("Replacing starcoder attention with flash attention")
    transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeAttention.forward = forward
    
  

from transformers.models.gpt_bigcode.modeling_gpt_bigcode import upcast_softmax,upcast_masked_softmax

class FlashSelfAttention(torch.nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.mask_value = None
        self.multi_query = config.multi_query
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.kv_heads = 1 if self.multi_query else self.num_heads
        self.kv_dim = self.kv_heads * self.head_dim
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        self.layer_idx = layer_idx
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = (
            config.scale_attention_softmax_in_fp32 and config.attention_softmax_in_fp32
        )
        
        #### FLASH ATTENTION CHANGE
        assert flash_attn_varlen_func is not None, (
            "Please install FlashAttention first, " "e.g., with pip install flash-attn"
        )
        assert rearrange is not None, "Please install einops first, e.g., with pip install einops"
        self.causal = True
        self.softmax_scale = None
        self.dropout_p = config.attn_pdrop
        

        if self.is_cross_attention:
            if self.multi_query:
                raise NotImplementedError("Multi-Query Attention not supported for cross_attention")

            self.c_attn = nn.Linear(self.embed_dim, 2 * self.embed_dim)
            self.q_attn = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = nn.Linear(self.embed_dim, self.embed_dim + 2 * self.kv_dim)

        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _get_mask_value(self, device, dtype):
        # torch.where expects a tensor. We use a cache to avoid recreating it every time.
        if self.mask_value is None or self.mask_value.dtype != dtype or self.mask_value.device != device:
            self.mask_value = torch.full([], torch.finfo(dtype).min, dtype=dtype, device=device)
        return self.mask_value

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        dtype = query.dtype
        softmax_dtype = torch.float32 if self.attention_softmax_in_fp32 else dtype
        upcast = dtype != softmax_dtype

        unscale = self.layer_idx + 1 if self.scale_attention_softmax_in_fp32 and upcast else 1
        scale_factor = unscale**-1
        if self.scale_attn_weights:
            scale_factor /= self.head_dim**0.5

        # MQA models: (batch_size, query_length, num_heads * head_dim)
        # MHA models: (batch_size, num_heads, query_length, head_dim)
        query_shape = query.shape
        batch_size = query_shape[0]
        key_length = key.size(-1)
        if self.multi_query:
            # (batch_size, query_length, num_heads, head_dim) x (batch_size, head_dim, key_length)
            # -> (batch_size, query_length, num_heads, key_length)
            query_length = query_shape[1]
            attn_shape = (batch_size, query_length, self.num_heads, key_length)
            attn_view = (batch_size, query_length * self.num_heads, key_length)
            # No copy needed for MQA 2, or when layer_past is provided.
            query = query.reshape(batch_size, query_length * self.num_heads, self.head_dim)
        else:
            # (batch_size, num_heads, query_length, head_dim) x (batch_size, num_heads, head_dim, key_length)
            # -> (batch_size, num_heads, query_length, key_length)
            query_length = query_shape[2]
            attn_shape = (batch_size, self.num_heads, query_length, key_length)
            attn_view = (batch_size * self.num_heads, query_length, key_length)
            # Always copies
            query = query.reshape(batch_size * self.num_heads, query_length, self.head_dim)
            # No copy when layer_past is provided.
            key = key.reshape(batch_size * self.num_heads, self.head_dim, key_length)

        attn_weights = torch.empty(attn_view, device=query.device, dtype=query.dtype)
        if query.device.type == "cpu":
            # This is needed because of a bug in pytorch https://github.com/pytorch/pytorch/issues/80588.
            # The bug was fixed in https://github.com/pytorch/pytorch/pull/96086,
            # but the fix has not been released as of pytorch version 2.0.0.
            attn_weights.zero_()
            beta = 1
        else:
            beta = 0
        attn_weights = torch.baddbmm(attn_weights, query, key, beta=beta, alpha=scale_factor).view(attn_shape)

        if upcast:
            # Use a fused kernel to prevent a large overhead from casting and scaling.
            # Sub-optimal when the key length is not a multiple of 8.
            if attention_mask is None:
                attn_weights = upcast_softmax(attn_weights, unscale, softmax_dtype)
            else:
                mask_value = self._get_mask_value(attn_weights.device, softmax_dtype)
                attn_weights = upcast_masked_softmax(attn_weights, attention_mask, mask_value, unscale, softmax_dtype)
        else:
            if attention_mask is not None:
                mask_value = self._get_mask_value(attn_weights.device, softmax_dtype)

                # The fused kernel is very slow when the key length is not a multiple of 8, so we skip fusion.
                attn_weights = torch.where(attention_mask, attn_weights, mask_value)

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            if self.multi_query:
                head_mask = head_mask.transpose(1, 2)
            attn_weights = attn_weights * head_mask

        if self.multi_query:
            attn_output = torch.bmm(attn_weights.view(attn_view), value).view(query_shape)
        else:
            attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def core_attention_flash(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q, k, v)))
        assert all((i.is_cuda for i in (q, k, v)))

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        q, k, v = [rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)

        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q

            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
            dropout_p = self.dropout_p
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = self.causal and (seqlen_q == seqlen_k)
            cu_seqlens_k = torch.arange(
                0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=q.device
            )
            dropout_p = 0

        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            dropout_p,
            softmax_scale=self.softmax_scale,
            causal=is_causal,
        )

        output = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]],
    ]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn") or not self.is_cross_attention:
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPTBigCodeAttention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key_value = self.c_attn(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        elif self.multi_query:
            query, key_value = self.c_attn(hidden_states).split((self.embed_dim, 2 * self.kv_dim), dim=2)
        else:
            # Note: We split as (self.num_heads, 3, self.head_dim) instead of (3, self.num_heads, self.head_dim),
            # i.e., the memory layout is not the same as GPT2.
            # This makes the concatenation with past_key_value more efficient.
            query, key_value = (
                self.c_attn(hidden_states)
                .view(*hidden_states.shape[:2], self.num_heads, 3 * self.head_dim)
                .transpose(1, 2)
                .split((self.head_dim, 2 * self.head_dim), dim=3)
            )

        if layer_past is not None:
            key_value = torch.cat((layer_past, key_value), dim=-2)
        present = key_value if use_cache else None

        key, value = key_value.split((self.head_dim, self.head_dim), dim=-1)

        if self.multi_query:
            batch_size, query_length, _ = query.shape
            query = query.reshape(batch_size, query_length, self.num_heads, self.head_dim)
            key, value = [unsqueeze_and_expand(x, unsqueeze_dim=2, num_head=self.num_heads) for x in [key, value]]
        else:
            query, key, value = [rearrange(x, "b h s d -> b s h d") for x in [query, key, value]]
        query, key, value = [x.to(torch.bfloat16) for x in [query, key, value]]
        # print(f"{query.shape=} {key.shape=} {value.shape=}")
        attn_output = self.core_attention_flash(query, key, value)
        attn_weights = None
        attn_output = self.c_proj(attn_output.reshape(hidden_states.shape))
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            raise NotImplementedError("`output_attentions` is not supported when `use_flash_attn` is True")
        return outputs  # a, present, (attentions)
      
      
def starcoder_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        logging.warning(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    print("Replacing starcoder attention with flash attention")
    transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeAttention = FlashSelfAttention
    