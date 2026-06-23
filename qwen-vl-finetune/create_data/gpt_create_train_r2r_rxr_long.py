#!/usr/bin/env python3
"""
Create long-history VLN training data for Spatial-TTT / Qwen3-VL.

Input:
    Existing JanusVLN-style JSON:
        /mnt/data/vmo-ai-task/anhdh35/JanusVLN/train_r2r_rxr.json

Trajectory image root:
        /mnt/data/vmo-ai-task/anhdh35/JanusVLN/data/trajectory_data

Output:
    New long-prefix video dataset:
        /mnt/data/vmo-ai-task/anhdh35/JanusVLN/train_r2r_rxr_long.json

New format:
    {
      "id": "R2R/train/1/step_0038_STOP",
      "dataset": "R2R",
      "split": "train",
      "episode_id": "1",
      "step_id": 38,
      "target_action": "STOP",
      "history_num_frames": 39,
      "num_video_frames": 39,
      "sampling": "full_prefix",
      "video": [
        "data/trajectory_data/R2R/train/1/step_0000_TURN_RIGHT.png",
        ...
        "data/trajectory_data/R2R/train/1/step_0038_STOP.png"
      ],
      "conversations": [
        {"from": "human", "value": "<video>\\n..."},
        {"from": "gpt", "value": "STOP"}
      ]
    }

Important:
    - For each step t, the video contains only frames up to t.
    - No future frames are included.
    - If prefix length > max_frames, use global-local sampling.
"""

import argparse
import concurrent.futures as futures
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


VALID_ACTIONS = {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"}
ACTION_RE = re.compile(r"step_(\d+)_(MOVE_FORWARD|TURN_LEFT|TURN_RIGHT|STOP)\.png$")
PATH_RE = re.compile(
    r"(?:^|/)data/trajectory_data/(R2R|RxR)/([^/]+)/([^/]+)/step_(\d+)_(MOVE_FORWARD|TURN_LEFT|TURN_RIGHT|STOP)\.png$"
)


def normalize_rel_path(path: str) -> str:
    """
    Normalize to path relative to JanusVLN data_path:
        data/trajectory_data/R2R/train/1/step_0000_TURN_RIGHT.png
    """
    path = path.strip()
    path = path.replace("\\", "/")

    idx = path.find("data/trajectory_data/")
    if idx >= 0:
        return path[idx:]

    # If somehow absolute path is passed.
    idx = path.find("/trajectory_data/")
    if idx >= 0:
        return "data" + path[idx:]

    return path


def parse_frame_path(path: str) -> Optional[Dict[str, Any]]:
    """
    Parse:
        data/trajectory_data/R2R/train/1/step_0000_TURN_RIGHT.png
    """
    rel = normalize_rel_path(path)
    m = PATH_RE.search(rel)
    if not m:
        return None

    dataset, split, episode_id, step_id, action = m.groups()
    return {
        "dataset": dataset,
        "split": split,
        "episode_id": episode_id,
        "step_id": int(step_id),
        "action": action,
        "rel_path": rel,
    }


def parse_step_action_from_filename(filename: str) -> Optional[Tuple[int, str]]:
    m = ACTION_RE.search(os.path.basename(filename))
    if not m:
        return None
    step_id, action = m.groups()
    return int(step_id), action


def extract_instruction(text: str) -> str:
    """
    Extract instruction from old prompt:

        Your task is to ...
        You should take one of the following actions:
    """
    patterns = [
        r"Your task is to\s*(.*?)\n\s*You should take one of the following actions:",
        r"Instruction:\s*(.*?)\n",
        r"Task:\s*(.*?)\n",
    ]

    for p in patterns:
        m = re.search(p, text, flags=re.S | re.I)
        if m:
            return " ".join(m.group(1).strip().split())

    # Fallback: remove visual placeholders and action list if regex fails.
    text = text.replace("<image>", " ").replace("<video>", " ")
    text = re.sub(
        r"You should take one of the following actions:.*",
        "",
        text,
        flags=re.S | re.I,
    )
    text = " ".join(text.split())
    return text


def build_prompt(
    instruction: str,
    previous_actions: List[str],
    max_prev_actions_in_prompt: int = 32,
) -> str:
    """
    Good prompt for one-video VLN action prediction.

    The video is always oldest -> newest.
    The last frame is the current observation.
    The target is the expert next action at the final/current frame.
    """
    if previous_actions:
        shown_actions = previous_actions[-max_prev_actions_in_prompt:]
        prev_text = ", ".join(shown_actions)
        if len(previous_actions) > max_prev_actions_in_prompt:
            prev_text = f"... , {prev_text}"
    else:
        prev_text = "None"

    prompt = (
        "<video>\n"
        "You are a visual language navigation model.\n"
        "The video frames are ordered from oldest to newest.\n"
        "The last frame is the current observation. Earlier frames are historical observations.\n"
        "Use the instruction, the trajectory history and the current observation to infer your progress.\n\n"
        f"Instruction:\n{instruction}\n\n"
        "Choose exactly one next action from:\n"
        "MOVE_FORWARD\n"
        "TURN_LEFT\n"
        "TURN_RIGHT\n"
        "STOP\n\n"
        "Answer with only the action name."
    )
    return prompt


def sample_vln_history(
    frames: List[str],
    actions: Optional[List[str]],
    max_frames: int = 128,
    local_frames: int = 32,
    keep_first: bool = True,
    keep_action_changes: bool = True,
) -> Tuple[List[str], str]:
    """
    Use full prefix if possible.
    If prefix is too long, use global-local sampling.

    Always:
        - no future frames
        - keep current frame
        - keep recent local frames densely
        - keep action-change anchors
        - fill remaining with global uniform history
    """
    n = len(frames)

    if n <= 0:
        raise ValueError("Empty frame prefix")

    # Qwen video path is safer with at least 2 frames.
    if n == 1:
        return [frames[0], frames[0]], "duplicate_single_frame"

    if n <= max_frames:
        return frames, "full_prefix"

    selected = set()

    if keep_first:
        selected.add(0)

    # Always keep current frame.
    selected.add(n - 1)

    # Dense recent local window.
    local_start = max(0, n - local_frames)
    for i in range(local_start, n):
        selected.add(i)

    # Keep action transition anchors.
    if keep_action_changes and actions is not None:
        usable = min(len(actions), n)
        for i in range(1, usable):
            if actions[i] != actions[i - 1]:
                selected.add(i - 1)
                selected.add(i)

    # Fill remaining slots with global uniform samples before local window.
    remaining = max_frames - len(selected)
    if remaining > 0:
        candidate_end = max(0, local_start - 1)
        if candidate_end > 0:
            global_idx = np.linspace(0, candidate_end, remaining, dtype=int)
            for i in global_idx:
                selected.add(int(i))

    # If action-change anchors made it too large, trim optional older frames.
    if len(selected) > max_frames:
        must_keep = {n - 1}
        if keep_first:
            must_keep.add(0)

        # Keep recent local frames as strongly preferred.
        for i in range(local_start, n):
            must_keep.add(i)

        optional = sorted(i for i in selected if i not in must_keep)
        keep_budget = max_frames - len(must_keep)

        if keep_budget > 0 and optional:
            opt_idx = np.linspace(0, len(optional) - 1, keep_budget, dtype=int)
            optional_keep = {optional[i] for i in opt_idx}
        else:
            optional_keep = set()

        selected = must_keep | optional_keep

    # Absolute fallback: if still too many, keep first + latest max_frames-1.
    if len(selected) > max_frames:
        latest = set(range(max(0, n - (max_frames - 1)), n))
        selected = latest
        if keep_first:
            selected.add(0)

        if len(selected) > max_frames:
            selected = set(sorted(selected)[-max_frames:])

    selected = sorted(selected)
    return [frames[i] for i in selected], "global_local_prefix"


def discover_episode_frames(
    trajectory_root: Path,
    dataset: str,
    split: str,
    episode_id: str,
) -> List[Tuple[int, str, str]]:
    """
    Return:
        [(step_id, action, rel_path), ...]

    rel_path format:
        data/trajectory_data/R2R/train/1/step_0000_TURN_RIGHT.png
    """
    ep_dir = trajectory_root / dataset / split / episode_id
    if not ep_dir.exists():
        return []

    out = []
    for p in ep_dir.glob("*.png"):
        parsed = parse_step_action_from_filename(p.name)
        if parsed is None:
            continue
        step_id, action = parsed
        rel = f"data/trajectory_data/{dataset}/{split}/{episode_id}/{p.name}"
        out.append((step_id, action, rel))

    out.sort(key=lambda x: x[0])
    return out


def should_keep_step(
    step_idx: int,
    actions: List[str],
    mode: str = "all",
    forward_stride: int = 4,
    keep_last_k: int = 5,
) -> bool:
    """
    mode='all':
        keep every step.

    mode='balanced':
        keep all STOP, all turns, all action changes, last K steps,
        and every Nth MOVE_FORWARD.

    For first long training, all is correct but huge.
    Balanced is usually more practical.
    """
    if mode == "all":
        return True

    if mode != "balanced":
        raise ValueError(f"Unknown keep mode: {mode}")

    n = len(actions)
    action = actions[step_idx]

    if step_idx == 0:
        return True

    if action == "STOP":
        return True

    if action in {"TURN_LEFT", "TURN_RIGHT"}:
        return True

    if step_idx >= max(0, n - keep_last_k):
        return True

    if actions[step_idx] != actions[step_idx - 1]:
        return True

    if step_idx + 1 < n and actions[step_idx] != actions[step_idx + 1]:
        return True

    if action == "MOVE_FORWARD" and step_idx % forward_stride == 0:
        return True

    return False


def process_episode(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Worker function. Creates all new training items for one episode.
    """
    trajectory_root = Path(task["trajectory_root"])
    dataset = task["dataset"]
    split = task["split"]
    episode_id = task["episode_id"]
    instruction = task["instruction"]

    max_frames = task["max_frames"]
    local_frames = task["local_frames"]
    keep_mode = task["keep_mode"]
    forward_stride = task["forward_stride"]
    keep_last_k = task["keep_last_k"]
    max_prev_actions_in_prompt = task["max_prev_actions_in_prompt"]

    # Prefer scanning the actual trajectory directory.
    discovered = discover_episode_frames(
        trajectory_root=trajectory_root,
        dataset=dataset,
        split=split,
        episode_id=episode_id,
    )

    # Fallback to annotation-derived frames if directory scan fails.
    if not discovered:
        discovered = task["fallback_frames"]

    if not discovered:
        return []

    discovered = sorted(discovered, key=lambda x: x[0])

    frames = [x[2] for x in discovered]
    actions = [x[1] for x in discovered]
    steps = [x[0] for x in discovered]

    items = []

    for i in range(len(frames)):
        target_action = actions[i]
        if target_action not in VALID_ACTIONS:
            continue

        if not should_keep_step(
            step_idx=i,
            actions=actions,
            mode=keep_mode,
            forward_stride=forward_stride,
            keep_last_k=keep_last_k,
        ):
            continue

        prefix_frames = frames[: i + 1]
        prefix_actions = actions[: i + 1]

        sampled_frames, sampling_type = sample_vln_history(
            frames=prefix_frames,
            actions=prefix_actions,
            max_frames=max_frames,
            local_frames=local_frames,
            keep_first=True,
            keep_action_changes=True,
        )

        previous_actions = actions[:i]
        prompt = build_prompt(
            instruction=instruction,
            previous_actions=previous_actions,
            max_prev_actions_in_prompt=max_prev_actions_in_prompt,
        )

        step_id = steps[i]
        new_id = f"{dataset}/{split}/{episode_id}/step_{step_id:04d}_{target_action}"

        item = {
            "id": new_id,
            "dataset": dataset,
            "split": split,
            "episode_id": episode_id,
            "step_id": step_id,
            "target_action": target_action,
            "history_num_frames": len(prefix_frames),
            "num_video_frames": len(sampled_frames),
            "sampling": sampling_type,
            "video": sampled_frames,
            "conversations": [
                {
                    "from": "human",
                    "value": prompt,
                },
                {
                    "from": "gpt",
                    "value": target_action,
                },
            ],
        }

        items.append(item)

    return items


def build_episode_tasks(
    input_json: Path,
    trajectory_root: Path,
    max_frames: int,
    local_frames: int,
    keep_mode: str,
    forward_stride: int,
    keep_last_k: int,
    max_prev_actions_in_prompt: int,
) -> List[Dict[str, Any]]:
    """
    Read old annotations and build one task per episode.
    """
    print(f"[INFO] Loading input json: {input_json}")
    with open(input_json, "r") as f:
        data = json.load(f)

    print(f"[INFO] Loaded {len(data):,} old samples")

    episodes = {}

    bad = 0

    for item in tqdm(data, desc="Grouping old samples"):
        images = item.get("images") or item.get("image") or []
        if isinstance(images, str):
            images = [images]

        if not images:
            bad += 1
            continue

        # The last image in old format should be current observation.
        cur_path = normalize_rel_path(images[-1])
        parsed = parse_frame_path(cur_path)

        # Fallback: try all images.
        if parsed is None:
            for p in reversed(images):
                parsed = parse_frame_path(p)
                if parsed is not None:
                    cur_path = parsed["rel_path"]
                    break

        if parsed is None:
            bad += 1
            continue

        dataset = parsed["dataset"]
        split = parsed["split"]
        episode_id = parsed["episode_id"]

        old_prompt = item["conversations"][0]["value"]
        instruction = extract_instruction(old_prompt)

        gpt_action = item["conversations"][1]["value"].strip()
        file_action = parsed["action"]
        action = gpt_action if gpt_action in VALID_ACTIONS else file_action

        key = (dataset, split, episode_id)

        if key not in episodes:
            episodes[key] = {
                "dataset": dataset,
                "split": split,
                "episode_id": episode_id,
                "instruction": instruction,
                "fallback_by_step": {},
            }

        # Keep the first non-empty instruction.
        if not episodes[key]["instruction"] and instruction:
            episodes[key]["instruction"] = instruction

        episodes[key]["fallback_by_step"][parsed["step_id"]] = (
            parsed["step_id"],
            action,
            cur_path,
        )

    tasks = []
    for (dataset, split, episode_id), ep in episodes.items():
        fallback_frames = sorted(ep["fallback_by_step"].values(), key=lambda x: x[0])

        task = {
            "dataset": dataset,
            "split": split,
            "episode_id": episode_id,
            "instruction": ep["instruction"],
            "fallback_frames": fallback_frames,
            "trajectory_root": str(trajectory_root),
            "max_frames": max_frames,
            "local_frames": local_frames,
            "keep_mode": keep_mode,
            "forward_stride": forward_stride,
            "keep_last_k": keep_last_k,
            "max_prev_actions_in_prompt": max_prev_actions_in_prompt,
        }

        tasks.append(task)

    print(f"[INFO] Built {len(tasks):,} episode tasks")
    print(f"[INFO] Bad / skipped old samples: {bad:,}")

    return tasks


def write_json_array_stream(
    output_json: Path,
    episode_results_iter: Iterable[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Stream output as a valid JSON array.
    Avoids holding all new items in memory.
    """
    output_json.parent.mkdir(parents=True, exist_ok=True)

    total_items = 0
    action_counter = Counter()
    frame_counter = Counter()
    sampling_counter = Counter()
    dataset_counter = Counter()

    first = True

    with open(output_json, "w") as f:
        f.write("[\n")

        for items in episode_results_iter:
            for item in items:
                if not first:
                    f.write(",\n")
                json.dump(item, f, ensure_ascii=False)
                first = False

                total_items += 1
                action_counter[item["target_action"]] += 1
                dataset_counter[item["dataset"]] += 1
                sampling_counter[item["sampling"]] += 1
                frame_counter[item["num_video_frames"]] += 1

        f.write("\n]\n")

    stats = {
        "total_items": total_items,
        "actions": dict(action_counter),
        "datasets": dict(dataset_counter),
        "sampling": dict(sampling_counter),
        "min_num_video_frames": min(frame_counter.keys()) if frame_counter else None,
        "max_num_video_frames": max(frame_counter.keys()) if frame_counter else None,
    }

    return stats


def save_stats(stats_path: Path, stats: Dict[str, Any]) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        type=str,
        default="/mnt/data/vmo-ai-task/anhdh35/JanusVLN/train_r2r_rxr.json",
    )
    parser.add_argument(
        "--trajectory_root",
        type=str,
        default="/mnt/data/vmo-ai-task/anhdh35/JanusVLN/data/trajectory_data",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/mnt/data/vmo-ai-task/anhdh35/JanusVLN/train_r2r_rxr_long.json",
    )
    parser.add_argument(
        "--stats_json",
        type=str,
        default="/mnt/data/vmo-ai-task/anhdh35/JanusVLN/train_r2r_rxr_long_stats.json",
    )

    parser.add_argument("--max_frames", type=int, default=128)
    parser.add_argument("--local_frames", type=int, default=32)

    parser.add_argument(
        "--keep_mode",
        type=str,
        default="all",
        choices=["all", "balanced"],
        help="all = keep every step. balanced = keep turns/stops/action changes + sparse forward.",
    )
    parser.add_argument("--forward_stride", type=int, default=4)
    parser.add_argument("--keep_last_k", type=int, default=5)
    parser.add_argument("--max_prev_actions_in_prompt", type=int, default=32)

    parser.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--shuffle_episodes", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_json = Path(args.input_json)
    trajectory_root = Path(args.trajectory_root)
    output_json = Path(args.output_json)
    stats_json = Path(args.stats_json)

    assert input_json.exists(), f"Input json does not exist: {input_json}"
    assert trajectory_root.exists(), f"Trajectory root does not exist: {trajectory_root}"

    tasks = build_episode_tasks(
        input_json=input_json,
        trajectory_root=trajectory_root,
        max_frames=args.max_frames,
        local_frames=args.local_frames,
        keep_mode=args.keep_mode,
        forward_stride=args.forward_stride,
        keep_last_k=args.keep_last_k,
        max_prev_actions_in_prompt=args.max_prev_actions_in_prompt,
    )

    if args.shuffle_episodes:
        random.shuffle(tasks)

    print(f"[INFO] Processing with {args.num_workers} workers")
    print(f"[INFO] Output: {output_json}")

    def result_iterator():
        with futures.ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            iterator = ex.map(process_episode, tasks, chunksize=8)
            for items in tqdm(iterator, total=len(tasks), desc="Processing episodes"):
                yield items

    stats = write_json_array_stream(
        output_json=output_json,
        episode_results_iter=result_iterator(),
    )

    save_stats(stats_json, stats)

    print("[DONE]")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"[INFO] Saved dataset: {output_json}")
    print(f"[INFO] Saved stats:   {stats_json}")


if __name__ == "__main__":
    main()