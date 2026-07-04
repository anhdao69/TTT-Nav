#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
True-streaming Spatial-TTT / TTT-Nav evaluation for Habitat R2R/RxR.

Local layout defaults:
  JANUS_ROOT       = /storage/anhdh35/JanusVLN
  CHECKPOINT_ROOT  = /storage/anhdh35/JanusVLN/checkpoints/spatial_ttt_nav_train_episode
  SPATIAL_TTT_ROOT = /storage/anhdh35/TTT-Nav/qwen-vl-finetune/models

What is different from the old evaluator:
  Old act(): rebuild whole conversation every step, re-encode all history images,
             create a fresh LaCTCache + DynamicCache in generate_with_spatial_ttt.
  This file: initialize cache once per episode, append only the new user turn/image,
             keep DynamicCache + LaCTCache across Habitat steps.

Important limitation:
  The LaCT decode path in the provided attention layer supports seq_len == 1 once
  a LaCTCache exists. Therefore after the first prompt prefill, this evaluator
  streams the newly appended prompt tokens one-by-one. It still encodes only the
  new frame once and avoids re-prefilling the whole episode.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make the local JanusVLN modules importable before Habitat extension registration.
DEFAULT_JANUS_ROOT = "/storage/anhdh35/JanusVLN"
DEFAULT_SPATIAL_TTT_ROOT = "/storage/anhdh35/TTT-Nav/qwen-vl-finetune/models"
for _p in (
    DEFAULT_JANUS_ROOT,
    os.path.join(DEFAULT_JANUS_ROOT, "src"),
    DEFAULT_SPATIAL_TTT_ROOT,
    os.path.join(DEFAULT_SPATIAL_TTT_ROOT, "models"),
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from PIL import Image

import habitat
from habitat import Env
from habitat.config.default import get_agent_config
from habitat.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat_extensions import measures  # noqa: F401  # registers OracleSuccess, etc.

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
from transformers.cache_utils import DynamicCache


# -----------------------------------------------------------------------------
# Paths / distributed helpers
# -----------------------------------------------------------------------------
def _insert_path(path: str):
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def _import_dist_helpers():
    try:
        from utils.dist import init_distributed_mode, get_rank, get_world_size  # type: ignore
        return init_distributed_mode, get_rank, get_world_size
    except Exception:
        def init_distributed_mode(args):
            args.rank = int(os.environ.get("RANK", "0"))
            args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            args.local_rank = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", 0)))
            args.distributed = args.world_size > 1
            if args.distributed and not dist.is_initialized():
                torch.cuda.set_device(args.local_rank)
                dist.init_process_group(backend="nccl", init_method="env://")

        def get_rank():
            return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        def get_world_size():
            return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        return init_distributed_mode, get_rank, get_world_size


# -----------------------------------------------------------------------------
# Prompt format: must match create_data/claude_build_long.py
# -----------------------------------------------------------------------------
FIRST_TURN = (
    "You are a visual-language navigation agent moving through an indoor "
    "environment. Follow the instruction below. At every step you receive one "
    "new observation image and must output exactly one action.\n"
    "Instruction: {instruction}\n"
    "Action space: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP.\n"
    "Observation: <image>"
)
NEXT_TURN = "Observation: <image>"

VALID_ACTIONS = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")
ACTION_RE = re.compile(r"\b(MOVE_FORWARD|TURN_LEFT|TURN_RIGHT|STOP)\b", re.IGNORECASE)


@dataclass
class EpisodeStepOutput:
    action_text: str
    raw_text: str
    prompt_tokens: int
    new_segment_tokens: int
    num_frames_seen: int


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def reset_peak_memory_stats(device: str):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def read_peak_memory_stats(device: str) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"peak_allocated_mb": 0.0, "peak_reserved_mb": 0.0}
    dev = torch.device(device)
    torch.cuda.synchronize(dev)
    return {
        "peak_allocated_mb": torch.cuda.max_memory_allocated(dev) / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(dev) / 1024**2,
    }


def parse_action(text: str, fallback: str = "STOP") -> str:
    normalized = text.upper().replace(" ", "_")
    m = ACTION_RE.search(normalized)
    return m.group(1).upper() if m else fallback


def _content_with_one_image(turn_text: str, image: Image.Image) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    used = False
    for seg in re.split(r"(<image>)", turn_text):
        if seg == "<image>":
            if used:
                raise ValueError("Each TTT-Nav turn must contain exactly one <image> placeholder.")
            content.append({"type": "image", "image": image})
            used = True
        elif seg.strip():
            content.append({"type": "text", "text": seg})
    if not used:
        raise ValueError(f"Missing <image> placeholder in turn: {turn_text}")
    return content


def build_user_turn_message(instruction: str, image: Image.Image, is_first_step: bool) -> List[Dict[str, Any]]:
    turn_text = FIRST_TURN.format(instruction=instruction) if is_first_step else NEXT_TURN
    return [{"role": "user", "content": _content_with_one_image(turn_text, image)}]


def _checkpoint_sort_key(path: Path) -> Tuple[int, str]:
    m = re.search(r"checkpoint[-_](\d+)", path.name)
    return (int(m.group(1)) if m else -1, path.name)


def resolve_safetensors(checkpoint_path: str) -> str:
    p = Path(checkpoint_path)
    if p.is_file():
        return str(p)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint_path does not exist: {checkpoint_path}")

    direct = p / "model.safetensors"
    if direct.exists():
        return str(direct)

    index = p / "model.safetensors.index.json"
    if index.exists():
        return str(index)

    shards = sorted(p.glob("*.safetensors"))
    if len(shards) == 1:
        return str(shards[0])

    checkpoint_dirs = sorted([x for x in p.glob("checkpoint-*") if x.is_dir()], key=_checkpoint_sort_key)
    if checkpoint_dirs:
        latest = checkpoint_dirs[-1]
        print(f"[load] checkpoint_path is a run folder. Auto-select latest: {latest}")
        return resolve_safetensors(str(latest))

    raise FileNotFoundError(
        f"Could not find model.safetensors, model.safetensors.index.json, "
        f"or checkpoint-* subfolders in {checkpoint_path}"
    )


# -----------------------------------------------------------------------------
# Qwen3-VL visual feature compatibility helpers
# -----------------------------------------------------------------------------
def _is_tensor_list(x):
    return isinstance(x, (list, tuple)) and len(x) > 0 and all(torch.is_tensor(t) for t in x)


def _unpack_qwen3_visual_features(features):
    """
    Support multiple Qwen3-VL signatures:
      old: (embeds, deepstack_embeds)
      new: (embeds, grid_thw, deepstack_embeds)
    """
    if not isinstance(features, (tuple, list)):
        return features, None
    if len(features) == 2:
        return features[0], features[1]
    if len(features) >= 3:
        deepstack = None
        for item in reversed(features[1:]):
            if _is_tensor_list(item):
                deepstack = item
                break
        if deepstack is None:
            for item in reversed(features[1:]):
                is_grid = torch.is_tensor(item) and item.ndim >= 2 and item.shape[-1] == 3
                if not is_grid:
                    deepstack = item
                    break
        return features[0], deepstack
    raise ValueError(f"Unexpected visual feature output: {type(features)} len={len(features)}")


def _cat_visual_embeds(embeds):
    if isinstance(embeds, (list, tuple)):
        return torch.cat(list(embeds), dim=0)
    return embeds


def _truncate_dynamic_cache(past_key_values, window_size: int, keep_full_layers: Optional[set] = None):
    """Bound KV memory for the LaCT / sliding-window layers only.

    In the hybrid architecture (75% LaCT layers + 25% full-attention "anchor"
    layers), the anchor layers must keep their ENTIRE KV history so that global
    context stays visible -- in VLN that is the instruction given at the very
    start of the episode. During training the anchor layers use full causal
    attention over the whole sequence (no sliding_window in the config), so
    windowing them at eval would silently break train/eval alignment and make
    the model "forget" the task on long episodes. Their indices are passed via
    keep_full_layers and skipped here; only LaCT/SWA layers are truncated.
    """
    if past_key_values is None or window_size <= 0:
        return
    keep_full_layers = keep_full_layers or set()

    if hasattr(past_key_values, "layers"):
        for idx, layer in enumerate(past_key_values.layers):
            if idx in keep_full_layers:
                continue
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if torch.is_tensor(k) and torch.is_tensor(v) and k.ndim >= 3 and k.shape[2] > window_size:
                layer.keys = k[:, :, -window_size:, :].contiguous()
                layer.values = v[:, :, -window_size:, :].contiguous()
        return

    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if key_cache is None or value_cache is None:
        return
    for i in range(min(len(key_cache), len(value_cache))):
        if i in keep_full_layers:
            continue
        k, v = key_cache[i], value_cache[i]
        if torch.is_tensor(k) and torch.is_tensor(v) and k.ndim >= 3 and k.shape[2] > window_size:
            key_cache[i] = k[:, :, -window_size:, :].contiguous()
            value_cache[i] = v[:, :, -window_size:, :].contiguous()


# -----------------------------------------------------------------------------
# Model loading: mirrors training flags
# -----------------------------------------------------------------------------
def load_spatial_ttt_model_aligned(
    model_path: str,
    checkpoint_path: str,
    device: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    num_lact_heads: int = 4,
    lact_chunk_size: int = 1024,
    window_size: int = 2048,
    lact_layers: str = "0/1/2/4/5/6/8/9/10/12/13/14/16/17/18/20/21/22/24/25/26",
    use_muon: bool = True,
    use_momentum: bool = True,
    use_conv_layer: bool = False,
    w0_w2_low_rank: int = 0,
    learnable_ttt_scale: bool = True,
    use_fused_kernel: bool = False,
):
    try:
        from models.spatial_ttt import SpatialTTTForConditionalGeneration, wrap_model_with_lact
    except Exception:
        from spatial_ttt import SpatialTTTForConditionalGeneration, wrap_model_with_lact

    config = AutoConfig.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)

    model = wrap_model_with_lact(
        model,
        num_lact_heads=num_lact_heads,
        w0_w2_low_rank=w0_w2_low_rank,
        use_fused_kernel=use_fused_kernel,
        lact_chunk_size=lact_chunk_size,
        window_size=window_size,
        use_conv_layer=use_conv_layer,
        lact_layers=lact_layers,
        use_muon=use_muon,
        use_momentum=use_momentum,
        learnable_ttt_scale=learnable_ttt_scale,
    )

    ckpt = resolve_safetensors(checkpoint_path)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[load] base_model     : {model_path}")
        print(f"[load] checkpoint     : {ckpt}")
        print(f"[load] device         : {device}")
        print(f"[load] lact_chunk     : {lact_chunk_size}")
        print(f"[load] window_size    : {window_size}")
        print(f"[load] lact_layers    : {lact_layers}")
        print(f"[load] muon/momentum  : {use_muon}/{use_momentum}")
        print(f"[load] fused/low_rank : {use_fused_kernel}/{w0_w2_low_rank}")

    # tie_weights removes the warning and is harmless if already tied.
    try:
        model.tie_weights()
    except Exception:
        pass

    load_checkpoint_and_dispatch(model, ckpt, device_map={"": device}, dtype=torch_dtype)
    return SpatialTTTForConditionalGeneration(model).eval()


# -----------------------------------------------------------------------------
# True streaming policy
# -----------------------------------------------------------------------------
class TrueStreamingSpatialTTTNavPolicy:
    """
    Persistent-cache episode policy.

    It keeps two state objects across Habitat steps:
      - self.past_key_values: sliding-window KV cache for attention
      - self.lact_cache:      fast-weight / pending-token state for LaCT TTT

    Each act(image) appends only:
      previous assistant close marker, current user Observation:<image>, assistant prefix,
    then greedily decodes the next action tokens.
    """

    def __init__(self, args: argparse.Namespace, device: str):
        _insert_path(args.spatial_ttt_root)
        _insert_path(os.path.join(args.spatial_ttt_root, "models"))

        # Import after path insertion.
        try:
            from models.causal_swa_lact import LaCTCache, Qwen3VLLaCTSWIGLULayer
            from models.causal_swa_lact_streaming_chunked import Qwen3VLLaCTSWIGLULayerStreamingChunked
        except Exception:
            from causal_swa_lact import LaCTCache, Qwen3VLLaCTSWIGLULayer
            from causal_swa_lact_streaming_chunked import Qwen3VLLaCTSWIGLULayerStreamingChunked

        self.LaCTCache = LaCTCache
        self.Qwen3VLLaCTSWIGLULayer = Qwen3VLLaCTSWIGLULayer
        self.Qwen3VLLaCTSWIGLULayerStreamingChunked = Qwen3VLLaCTSWIGLULayerStreamingChunked

        self.args = args
        self.device = device
        self.ttt_model = load_spatial_ttt_model_aligned(
            model_path=args.model_path,
            checkpoint_path=args.checkpoint_path,
            device=device,
            torch_dtype=torch.bfloat16,
            num_lact_heads=args.num_lact_heads,
            lact_chunk_size=args.lact_chunk_size,
            window_size=args.window_size,
            lact_layers=args.lact_layers,
            use_muon=args.use_muon,
            use_momentum=args.use_momentum,
            use_conv_layer=args.use_conv_layer,
            w0_w2_low_rank=args.w0_w2_low_rank,
            learnable_ttt_scale=args.learnable_ttt_scale,
            use_fused_kernel=args.use_fused_kernel,
        )
        self.qwen_model = self.ttt_model._model

        # Hybrid architecture: the layers NOT in lact_layers are full-attention
        # "anchor" layers (25%). Their KV must never be windowed so global
        # context (the instruction) stays visible for the whole episode.
        try:
            num_layers = len(self.qwen_model.model.language_model.layers)
        except Exception:
            cfg = self.qwen_model.config
            num_layers = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
        lact_set = {int(x) for x in str(args.lact_layers).split("/") if x != ""}
        self.anchor_layers = {i for i in range(num_layers) if i not in lact_set}
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[load] anchor (full-attn) layers kept un-windowed: {sorted(self.anchor_layers)}")

        self.processor = AutoProcessor.from_pretrained(
            args.model_path,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            padding_side="left",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.bos_id = self.tokenizer.bos_token_id

        self.instruction: Optional[str] = None
        self.num_frames_seen = 0
        self.seq_len = 0
        self.next_position_ids: Optional[torch.Tensor] = None  # [3, 1, 1]
        self.past_key_values: Optional[DynamicCache] = None
        self.lact_cache = None
        self.need_assistant_close = False

    def reset_episode(self, instruction: str):
        self.instruction = instruction
        self.num_frames_seen = 0
        self.seq_len = 0
        self.next_position_ids = None
        self.past_key_values = DynamicCache()
        self.lact_cache = self.LaCTCache()
        self.need_assistant_close = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _apply_chat_template_for_new_turn(self, image: Image.Image, is_first_step: bool) -> Dict[str, torch.Tensor]:
        assert self.instruction is not None
        messages = build_user_turn_message(self.instruction, image, is_first_step=is_first_step)
        try:
            batch = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                do_sample_frames=False,
            )
        except TypeError:
            batch = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )

        out: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device)

        # Avoid inserting a BOS token in the middle of a streaming episode.
        ids = out["input_ids"]
        if self.seq_len > 0 and self.bos_id is not None and ids.shape[1] > 0 and int(ids[0, 0]) == int(self.bos_id):
            out["input_ids"] = ids[:, 1:]

        # After decoding the previous action, the cache contains the assistant prefix
        # and action text. Close that assistant message before the next user turn.
        if self.need_assistant_close:
            close_ids = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
            close = torch.tensor(close_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            out["input_ids"] = torch.cat([close, out["input_ids"]], dim=1)

        for k in ("pixel_values", "pixel_values_videos"):
            if k in out:
                out[k] = out[k].to(device=self.device, dtype=torch.bfloat16)
        return out

    def _compute_segment_position_ids(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Local multimodal positions for this newly appended segment.
        local_pos, _ = self.qwen_model.model.get_rope_index(input_ids, image_grid_thw, video_grid_thw)
        local_pos = local_pos.to(self.device)
        if self.next_position_ids is None:
            return local_pos
        # Shift the whole local segment after the previous cached token.
        # Shape: [3, batch, segment_len] + [3, batch, 1]
        return local_pos + self.next_position_ids.to(local_pos.device)

    def _prepare_segment_embeds(self, inputs: Dict[str, torch.Tensor]):
        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        pixel_values_videos = inputs.get("pixel_values_videos")
        video_grid_thw = inputs.get("video_grid_thw")

        inputs_embeds = self.qwen_model.model.language_model.embed_tokens(input_ids)
        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        if pixel_values is not None:
            image_features = self.qwen_model.model.get_image_features(pixel_values, image_grid_thw)
            image_embeds, deepstack_image_embeds = _unpack_qwen3_visual_features(image_features)
            image_embeds = _cat_visual_embeds(image_embeds).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.qwen_model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_features = self.qwen_model.model.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds, deepstack_video_embeds = _unpack_qwen3_visual_features(video_features)
            video_embeds = _cat_visual_embeds(video_embeds).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.qwen_model.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask_1d = image_mask[..., 0]
            video_mask_1d = video_mask[..., 0]
            visual_pos_masks = image_mask_1d | video_mask_1d
            deepstack_visual_embeds = []
            if deepstack_image_embeds is not None and deepstack_video_embeds is not None:
                image_mask_joint = image_mask_1d[visual_pos_masks]
                video_mask_joint = video_mask_1d[visual_pos_masks]
                for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                    embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            deepstack_visual_embeds = deepstack_video_embeds

        position_ids = self._compute_segment_position_ids(input_ids, image_grid_thw, video_grid_thw)
        return input_ids, inputs_embeds, position_ids, visual_pos_masks, deepstack_visual_embeds, video_mask, video_grid_thw

    def _select_deepstack_for_slice(
        self,
        deepstack_visual_embeds,
        full_visual_mask: Optional[torch.Tensor],
        token_start: int,
        token_end: int,
    ):
        if deepstack_visual_embeds is None or full_visual_mask is None:
            return None, None
        mask_slice = full_visual_mask[:, token_start:token_end]
        if not bool(mask_slice.any().item()):
            return mask_slice, None

        # Count how many visual tokens came before token_start in this segment.
        before = int(full_visual_mask[:, :token_start].sum().item())
        count = int(mask_slice.sum().item())
        if count <= 0:
            return mask_slice, None

        if isinstance(deepstack_visual_embeds, (list, tuple)):
            return mask_slice, [x[before: before + count] for x in deepstack_visual_embeds]
        return mask_slice, deepstack_visual_embeds[before: before + count]

    def _forward_embeds(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds=None,
        video_mask: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward already-prepared embeddings and update persistent caches."""
        batch_size, segment_len, _ = inputs_embeds.shape
        assert batch_size == 1, "True streaming evaluator supports batch_size=1."

        language_model = self.qwen_model.model.language_model
        rotary_emb = language_model.rotary_emb
        norm = language_model.norm
        lm_head = self.ttt_model.lm_head

        hidden_states = inputs_embeds
        position_embeddings = rotary_emb(hidden_states, position_ids)
        cache_position = torch.arange(self.seq_len, self.seq_len + segment_len, device=self.device, dtype=torch.long)

        for layer_idx, layer in enumerate(language_model.layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            if isinstance(layer.self_attn, (self.Qwen3VLLaCTSWIGLULayer, self.Qwen3VLLaCTSWIGLULayerStreamingChunked)):
                hidden_states, _ = layer.self_attn(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    past_key_values=self.past_key_values,
                    cache_position=cache_position,
                    lact_cache=self.lact_cache,
                    video_mask=video_mask,
                    video_grid_thw=video_grid_thw,
                )
            else:
                hidden_states, _ = layer.self_attn(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    past_key_values=self.past_key_values,
                    cache_position=cache_position,
                    attention_mask=None,
                )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            if deepstack_visual_embeds is not None and visual_pos_masks is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = language_model._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = norm(hidden_states)
        logits = lm_head(hidden_states[:, -1:, :])

        # Advance global position/counter and bound all KV caches.
        self.seq_len += segment_len
        self.next_position_ids = position_ids[:, :, -1:].detach() + 1
        _truncate_dynamic_cache(self.past_key_values, self.args.window_size, keep_full_layers=self.anchor_layers)
        return logits

    def _append_segment(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, int]:
        (
            input_ids,
            inputs_embeds,
            position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
            video_mask,
            video_grid_thw,
        ) = self._prepare_segment_embeds(inputs)

        segment_len = input_ids.shape[1]
        if segment_len == 0:
            raise RuntimeError("Empty streaming segment.")

        # First episode prompt can use fast streaming prefill. After LaCTCache exists,
        # the provided decode path asserts seq_len == 1, so append token-by-token.
        if self.seq_len == 0:
            logits = self._forward_embeds(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                video_mask=video_mask,
                video_grid_thw=video_grid_thw,
            )
            return logits, segment_len

        logits = None
        for i in range(segment_len):
            mask_i, deep_i = self._select_deepstack_for_slice(deepstack_visual_embeds, visual_pos_masks, i, i + 1)
            video_mask_i = video_mask[:, i:i+1] if torch.is_tensor(video_mask) and video_mask.dim() >= 2 else None
            logits = self._forward_embeds(
                input_ids=input_ids[:, i:i+1],
                inputs_embeds=inputs_embeds[:, i:i+1, :],
                position_ids=position_ids[:, :, i:i+1],
                visual_pos_masks=mask_i,
                deepstack_visual_embeds=deep_i,
                video_mask=video_mask_i,
                video_grid_thw=video_grid_thw,
            )
        assert logits is not None
        return logits, segment_len

    def _append_token(self, token_id: torch.Tensor) -> torch.Tensor:
        token_id = token_id.to(self.device)
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(0)
        inputs_embeds = self.qwen_model.model.language_model.embed_tokens(token_id)
        if self.next_position_ids is None:
            pos = torch.zeros((3, 1, 1), dtype=torch.long, device=self.device)
        else:
            pos = self.next_position_ids.to(self.device)
        return self._forward_embeds(
            input_ids=token_id,
            inputs_embeds=inputs_embeds,
            position_ids=pos,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            video_mask=None,
            video_grid_thw=None,
        )

    @torch.inference_mode()
    def act(self, image: Image.Image) -> EpisodeStepOutput:
        if self.instruction is None:
            raise RuntimeError("Call reset_episode(instruction) before act().")

        is_first_step = self.num_frames_seen == 0
        inputs = self._apply_chat_template_for_new_turn(image, is_first_step=is_first_step)
        logits, new_segment_tokens = self._append_segment(inputs)

        generated_tokens: List[torch.Tensor] = []
        raw_text = ""
        action = self.args.invalid_action_fallback
        generated_im_end = False

        for _ in range(self.args.max_new_tokens):
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            logits = self._append_token(next_token)

            token_value = int(next_token.item())
            if self.im_end_id is not None and token_value == int(self.im_end_id):
                generated_im_end = True
                break
            if self.tokenizer.eos_token_id is not None and token_value == int(self.tokenizer.eos_token_id):
                generated_im_end = True
                break

            joined = torch.cat(generated_tokens, dim=1)
            raw_text = self.processor.batch_decode(
                joined,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            parsed = parse_action(raw_text, fallback="")
            if parsed in VALID_ACTIONS:
                action = parsed
                break

        if not raw_text and generated_tokens:
            joined = torch.cat(generated_tokens, dim=1)
            raw_text = self.processor.batch_decode(
                joined,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            action = parse_action(raw_text, fallback=self.args.invalid_action_fallback)

        # Usually we stop immediately after ACTION, before the model emits <|im_end|>.
        # Therefore the next step must first append the assistant close marker.
        self.need_assistant_close = not generated_im_end
        self.num_frames_seen += 1

        return EpisodeStepOutput(
            action_text=action,
            raw_text=raw_text,
            prompt_tokens=self.seq_len,
            new_segment_tokens=new_segment_tokens,
            num_frames_seen=self.num_frames_seen,
        )


# -----------------------------------------------------------------------------
# Habitat evaluator
# -----------------------------------------------------------------------------
class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str,
        env_num: int,
        output_path: str,
        policy: TrueStreamingSpatialTTTNavPolicy,
        args: argparse.Namespace,
    ):
        self.args = args
        self.device = policy.device
        self.split = split
        self.env_num = env_num
        self.output_path = output_path
        self.policy = policy
        os.makedirs(output_path, exist_ok=True)

        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)

        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = self.split
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        self.actions2idx = OrderedDict(
            {
                "STOP": 0,
                "MOVE_FORWARD": 1,
                "TURN_LEFT": 2,
                "TURN_RIGHT": 3,
            }
        )

    def config_env(self) -> Env:
        return Env(config=self.config)

    def _rank_result_path(self, rank: int) -> str:
        return os.path.join(self.output_path, f"result_rank{rank}.jsonl")

    def _load_done_keys(self) -> set:
        done = set()
        for path in glob.glob(os.path.join(self.output_path, "result_rank*.jsonl")):
            with open(path, "r") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if "scene_id" in r and "episode_id" in r:
                        done.add((str(r["scene_id"]), str(r["episode_id"]), r.get("episode_instruction", "")))
        return done

    def eval_action(self, rank: int) -> Dict[str, List[float]]:
        env = self.config_env()

        selected_scenes = None
        if self.args.scene_names:
            selected_scenes = {s.strip() for s in self.args.scene_names.split(",") if s.strip()}
            if rank == 0:
                print(f"[eval] selected scenes: {sorted(selected_scenes)}")

        scene_episode_dict: Dict[str, list] = {}
        for ep in env.episodes:
            scene_id = ep.scene_id.split("/")[-2]
            if selected_scenes is not None and scene_id not in selected_scenes:
                continue
            scene_episode_dict.setdefault(ep.scene_id, []).append(ep)

        metrics_acc = {
            "success": [],
            "spl": [],
            "oracle_success": [],
            "distance_to_goal": [],
            "steps": [],
            "peak_cuda_mb": [],
            "peak_reserved_mb": [],
        }
        done_keys = self._load_done_keys() if self.args.resume else set()
        result_path = self._rank_result_path(rank)

        for scene in sorted(scene_episode_dict.keys()):
            scene_id = scene.split("/")[-2]
            episodes = scene_episode_dict[scene]
            shard = episodes[rank :: self.env_num]
            if rank == 0:
                print(f"[eval] scene={scene_id}, total={len(episodes)}, this_rank={len(shard)}")

            pbar = tqdm.tqdm(shard, desc=f"rank {rank} scene {scene_id}", disable=(rank != 0))
            for episode in pbar:
                episode_instruction = (
                    episode.instruction.instruction_text
                    if "objectnav" not in self.config_path.lower()
                    else episode.object_category
                )
                episode_id = str(episode.episode_id)
                done_key = (scene_id, episode_id, episode_instruction)
                if done_key in done_keys:
                    continue

                env.current_episode = episode
                observations = env.reset()
                self.policy.reset_episode(episode_instruction)

                vis_frames = []
                should_save_video = self.args.save_video and (random.random() < self.args.save_video_ratio)
                if should_save_video:
                    os.makedirs(os.path.join(self.output_path, "vis"), exist_ok=True)

                reset_peak_memory_stats(self.device)
                ep_peak_alloc = 0.0
                ep_peak_reserved = 0.0
                step_id = 0
                last_raw = ""
                last_action = ""
                last_prompt_tokens = 0

                while not env.episode_over:
                    rgb = observations["rgb"]
                    image = Image.fromarray(rgb).convert("RGB")
                    info = env.get_metrics()

                    out = self.policy.act(image)
                    last_raw = out.raw_text
                    last_action = out.action_text
                    last_prompt_tokens = out.prompt_tokens

                    mem = read_peak_memory_stats(self.device)
                    ep_peak_alloc = max(ep_peak_alloc, mem["peak_allocated_mb"])
                    ep_peak_reserved = max(ep_peak_reserved, mem["peak_reserved_mb"])

                    if info.get("top_down_map") is not None and should_save_video:
                        vis_frames.append(observations_to_image({"rgb": observations["rgb"]}, info))

                    action_idx = self.actions2idx.get(out.action_text, self.actions2idx[self.args.invalid_action_fallback])
                    if step_id >= self.args.max_steps:
                        action_idx = self.actions2idx["STOP"]

                    observations = env.step(action_idx)
                    step_id += 1

                metrics = env.get_metrics()
                if should_save_video:
                    images_to_video(
                        vis_frames,
                        os.path.join(self.output_path, "vis"),
                        f"{scene_id}_{episode_id}",
                        fps=6,
                        quality=9,
                    )

                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "episode_instruction": episode_instruction,
                    "success": float(metrics["success"]),
                    "spl": float(metrics["spl"]),
                    "os": float(metrics["oracle_success"]),
                    "ne": float(metrics["distance_to_goal"]),
                    "steps": int(step_id),
                    "peak_cuda_mb": round(ep_peak_alloc, 2),
                    "peak_reserved_mb": round(ep_peak_reserved, 2),
                    "last_action": last_action,
                    "last_raw": last_raw,
                    "last_prompt_tokens": int(last_prompt_tokens),
                    "streaming": True,
                }
                with open(result_path, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                metrics_acc["success"].append(result["success"])
                metrics_acc["spl"].append(result["spl"])
                metrics_acc["oracle_success"].append(result["os"])
                metrics_acc["distance_to_goal"].append(result["ne"])
                metrics_acc["steps"].append(result["steps"])
                metrics_acc["peak_cuda_mb"].append(result["peak_cuda_mb"])
                metrics_acc["peak_reserved_mb"].append(result["peak_reserved_mb"])

                pbar.set_postfix(
                    sr=np.mean(metrics_acc["success"]) if metrics_acc["success"] else 0.0,
                    spl=np.mean(metrics_acc["spl"]) if metrics_acc["spl"] else 0.0,
                    ne=np.mean(metrics_acc["distance_to_goal"]) if metrics_acc["distance_to_goal"] else 0.0,
                    tok=last_prompt_tokens,
                )

        env.close()
        return metrics_acc


# -----------------------------------------------------------------------------
# Summary / args
# -----------------------------------------------------------------------------
def _gather_variable_tensor(x: torch.Tensor, world_size: int, device: str) -> torch.Tensor:
    if world_size == 1:
        return x
    n = torch.tensor([x.numel()], dtype=torch.long, device=device)
    ns = [torch.zeros_like(n) for _ in range(world_size)]
    dist.all_gather(ns, n)
    sizes = [int(t.item()) for t in ns]
    max_n = max(sizes) if sizes else 0
    padded = torch.zeros(max_n, dtype=x.dtype, device=device)
    if x.numel() > 0:
        padded[: x.numel()] = x
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return torch.cat([g[: sizes[i]] for i, g in enumerate(gathered)], dim=0)


def summarize_and_write(local_metrics: Dict[str, List[float]], args: argparse.Namespace, device: str, rank: int, world_size: int):
    keys = ["success", "spl", "oracle_success", "distance_to_goal", "steps", "peak_cuda_mb", "peak_reserved_mb"]
    gathered: Dict[str, torch.Tensor] = {}
    for k in keys:
        t = torch.tensor(local_metrics[k], dtype=torch.float32, device=device)
        gathered[k] = _gather_variable_tensor(t, world_size, device)

    if rank != 0:
        return

    n = int(gathered["success"].numel())
    if n == 0:
        summary = {
            "sucs_all": 0.0,
            "spls_all": 0.0,
            "oss_all": 0.0,
            "ones_all": 0.0,
            "steps_mean": 0.0,
            "peak_cuda_mb_mean": 0.0,
            "peak_cuda_mb_max": 0.0,
            "peak_reserved_mb_mean": 0.0,
            "length": 0,
            "streaming": True,
        }
    else:
        summary = {
            "sucs_all": gathered["success"].mean().item(),
            "spls_all": gathered["spl"].mean().item(),
            "oss_all": gathered["oracle_success"].mean().item(),
            "ones_all": gathered["distance_to_goal"].mean().item(),
            "steps_mean": gathered["steps"].mean().item(),
            "peak_cuda_mb_mean": gathered["peak_cuda_mb"].mean().item(),
            "peak_cuda_mb_max": gathered["peak_cuda_mb"].max().item(),
            "peak_reserved_mb_mean": gathered["peak_reserved_mb"].mean().item(),
            "length": n,
            "streaming": True,
        }
    print("[summary]", json.dumps(summary, indent=2))
    with open(os.path.join(args.output_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def str2bool(x: str) -> bool:
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x in ("1", "true", "yes", "y"):
        return True
    if x in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool: {x}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Paths
    p.add_argument("--janus_root", type=str, default=DEFAULT_JANUS_ROOT)
    p.add_argument("--spatial_ttt_root", type=str, default=DEFAULT_SPATIAL_TTT_ROOT)
    p.add_argument(
        "--model_path",
        type=str,
        default="/storage/anhdh35/JanusVLN/checkpoints/Qwen3-VL-2B-Instruct",
        help="Base Qwen3-VL model dir used for training.",
    )
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default="/storage/anhdh35/JanusVLN/checkpoints/spatial_ttt_nav_train_episode",
        help="Trained TTT checkpoint dir, run folder, or model.safetensors.",
    )
    p.add_argument("--habitat_config_path", type=str, default="config/vln_r2r.yaml")
    p.add_argument("--eval_split", type=str, default="val_unseen")
    p.add_argument("--output_path", type=str, default="/storage/anhdh35/JanusVLN/results/spatial_ttt_nav_stream/val_unseen")

    # Generation / context
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--invalid_action_fallback", type=str, default="STOP", choices=list(VALID_ACTIONS))
    # Cap near the training horizon (episodes were capped at MAX_FRAMES=128,
    # mean ~73). Running far past this pushes the LaCT state / anchor context
    # into an untrained regime. 200 leaves headroom while avoiding 400-step drift.
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--min_pixels", type=int, default=50176)
    p.add_argument("--max_pixels", type=int, default=168960)

    # Spatial-TTT flags: defaults mirror training
    p.add_argument("--num_lact_heads", type=int, default=4)
    p.add_argument("--lact_chunk_size", type=int, default=1024)
    p.add_argument("--window_size", type=int, default=2048)
    p.add_argument("--lact_layers", type=str, default="0/1/2/4/5/6/8/9/10/12/13/14/16/17/18/20/21/22/24/25/26")
    p.add_argument("--use_muon", type=str2bool, default=True)
    p.add_argument("--use_momentum", type=str2bool, default=True)
    p.add_argument("--use_conv_layer", type=str2bool, default=False)
    p.add_argument("--w0_w2_low_rank", type=int, default=0)
    p.add_argument("--learnable_ttt_scale", type=str2bool, default=True)
    p.add_argument("--use_fused_kernel", type=str2bool, default=False)

    # Eval misc
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--save_video_ratio", type=float, default=0.05)
    p.add_argument("--scene_names", type=str, default=None)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)

    # Distributed args compatible with utils.dist / torchrun.
    p.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    p.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    p.add_argument("--dist_url", default="env://")
    return p.parse_args()


def main():
    args = parse_args()
    _insert_path(args.janus_root)
    _insert_path(os.path.join(args.janus_root, "src"))
    _insert_path(args.spatial_ttt_root)
    _insert_path(os.path.join(args.spatial_ttt_root, "models"))

    init_distributed_mode, get_rank, get_world_size = _import_dist_helpers()
    set_seed(args.seed)
    init_distributed_mode(args)

    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
        print("[args]", json.dumps(vars(args), indent=2))

    policy = TrueStreamingSpatialTTTNavPolicy(args=args, device=device)
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        policy=policy,
        args=args,
    )
    local_metrics = evaluator.eval_action(rank)
    if world_size > 1:
        dist.barrier()
    summarize_and_write(local_metrics, args, device, rank, world_size)
    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    main()