import torch
from flash_attn.flash_attn_interface import flash_attn_unpadded_func
from einops import rearrange
from typing import Optional, Tuple, Union
import logging 
import transformers

def unsqueeze_and_expand(proj, unsqueeze_dim=2, num_head=12):
    proj = torch.unsqueeze(proj, unsqueeze_dim)
    b, sq, _, hn = proj.shape
    proj = proj.expand((b, sq, num_head, hn))
    return proj

# def forward(self, q, k, v):
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
        if self.use_flash_attn:
            raise NotImplementedError("`output_attentions` is not supported when `use_flash_attn` is True")
        if self.multi_query:
            # Transpose to return weights in the usual format (batch_size, num_heads, query_length, key_length)
            attn_weights = attn_weights.transpose(1, 2)
        outputs += (attn_weights,)

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

    output = flash_attn_unpadded_func(
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
    logging.info("Replacing starcoder attention with flash attention")
    transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeAttention.forward = forward