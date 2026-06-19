import json
from pathlib import Path

ann = Path("/mnt/data/vmo-ai-task/anhdh35/JanusVLN/train_r2r_rxr.json")

if ann.suffix == ".jsonl":
    data = [json.loads(x) for x in ann.read_text().splitlines() if x.strip()]
else:
    data = json.load(open(ann))

bad = []
for i, x in enumerate(data):
    text = "\n".join(c["value"] for c in x.get("conversations", []))

    has_video = ("video" in x) or ("<video>" in text)
    has_image_key = "image" in x
    has_images_key = "images" in x
    n_tok = text.count("<image>")

    if has_video or not has_image_key or has_images_key:
        bad.append((i, x.keys(), n_tok, has_video, has_image_key, has_images_key))
        continue

    n_img = len(x["image"]) if isinstance(x["image"], list) else 1
    if n_img != n_tok:
        bad.append((i, x.keys(), n_tok, "n_img=" + str(n_img)))

print("total:", len(data))
print("bad:", len(bad))
print("first 20 bad:")
for b in bad[:20]:
    print(b)