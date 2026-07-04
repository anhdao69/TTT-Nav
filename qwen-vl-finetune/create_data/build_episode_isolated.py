#!/usr/bin/env python3
"""
Build ACTION-ISOLATED per-episode training data for Spatial-TTT VLN.

Converts train_r2r_rxr_episode.json (interleaved frames + oracle action text)
into train_r2r_rxr_episode_isolated.json where the action text is REMOVED from
the token stream. Each step becomes:

    <|im_start|>user\n[Observation: <image>]<|im_end|>\n
    <|im_start|>assistant\n<|im_end|>\n          <- empty assistant "readout" turn

Supervision: the label at the <|im_end|> token right after "assistant\n" is
overridden with a SINGLE-TOKEN action word (forward / left / right / stop),
so the prediction made FROM the "assistant\n" position is trained, while the
action never appears as an input token -> no action leakage, and the training
stream is byte-identical to the inference stream.

Token facts (Qwen3-VL tokenizer):
    "forward"=13435  "left"=2359  "right"=1291  "stop"=9495   (all single tokens)
    "<|im_start|>assistant\n" = [151644, 77091, 198]

KEEP IN SYNC WITH INFERENCE: at eval, append the user turn + generation prompt
("<|im_start|>assistant\n"), read logits once (argmax over the 4 action ids),
then append "<|im_end|>\n" before the next user turn. Never append action text.
"""
import argparse
import json
import re
from collections import Counter

# ----------------------------------------------------------------------
# Prompt templates (KEEP IN SYNC WITH INFERENCE!)
# ----------------------------------------------------------------------
FIRST_TURN = (
    "You are a visual-language navigation agent moving through an indoor "
    "environment. Follow the instruction below. At every step you receive one "
    "new observation image and must output exactly one action word.\n"
    "Instruction: {instruction}\n"
    "Action space: forward, left, right, stop.\n"
    "Observation: <image>"
)
NEXT_TURN = "Observation: <image>"

# env action name -> single-token action word
ACTION_MAP = {
    "MOVE_FORWARD": "forward",
    "TURN_LEFT": "left",
    "TURN_RIGHT": "right",
    "STOP": "stop",
}

# instruction extractor for the old episode json's first human turn
OLD_INSTR_RE = re.compile(r"Instruction: (.*?)\nAction space:", re.DOTALL)


def convert_sample(sample):
    convs = sample["conversations"]
    human_turns = [t for t in convs if t["from"] == "human"]
    gpt_turns = [t for t in convs if t["from"] == "gpt"]
    images = sample["images"]

    if not (len(human_turns) == len(gpt_turns) == len(images)):
        return None, "count_mismatch"

    m = OLD_INSTR_RE.search(human_turns[0]["value"])
    if not m:
        return None, "no_instruction"
    instruction = m.group(1).strip()

    actions = []
    for t in gpt_turns:
        act = ACTION_MAP.get(t["value"].strip())
        if act is None:
            return None, "bad_action"
        actions.append(act)

    new_convs = []
    for j in range(len(images)):
        human = FIRST_TURN.format(instruction=instruction) if j == 0 else NEXT_TURN
        new_convs.append({"from": "human", "value": human})
        # empty assistant turn -> "<|im_start|>assistant\n<|im_end|>\n" readout scaffold
        new_convs.append({"from": "gpt", "value": ""})

    return {
        "id": sample["id"],
        "conversations": new_convs,
        "images": images,
        "actions": actions,
    }, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src_json",
        default="/mnt/data/vmo-ai-task/anhdh35/JanusVLN/data/labels/train_r2r_rxr_episode.json",
    )
    ap.add_argument(
        "--out_json",
        default="/mnt/data/vmo-ai-task/anhdh35/JanusVLN/data/labels/train_r2r_rxr_episode_isolated.json",
    )
    args = ap.parse_args()

    print(f"[1/2] Reading {args.src_json} ...")
    with open(args.src_json, "r") as f:
        data = json.load(f)
    print(f"      {len(data)} samples")

    out, status, act_dist = [], Counter(), Counter()
    for sample in data:
        new_sample, st = convert_sample(sample)
        status[st] += 1
        if new_sample is not None:
            out.append(new_sample)
            act_dist.update(new_sample["actions"])

    total_actions = sum(act_dist.values())
    print("\n================ SUMMARY ================")
    print(f"status                  : {dict(status)}")
    print(f"output samples          : {len(out)}")
    print(
        "action distribution     : "
        + ", ".join(f"{k}={v} ({100 * v / total_actions:.1f}%)" for k, v in act_dist.most_common())
    )
    print(f"[2/2] writing -> {args.out_json}")
    with open(args.out_json, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print("done.")


if __name__ == "__main__":
    main()
