import re

# Spatial-TTT-Data-97k: mini high-quality spatial dataset from Spatial-TTT, ~97k samples (download from THU-SI/Spatial-TTT-Data-97k on Hugging Face)
# SPATIAL_TTT_DATA_97K = {
#     "annotation_path": "PATH_TO_SPATIAL_TTT_97K_ANNOTATION",
#     "data_path": "PATH_TO_SPATIAL_TTT_97K_DATA",
# }

# data_dict = {
#     "spatial_ttt_data_97k": SPATIAL_TTT_DATA_97K,
# }

TRAIN_R2R_RxR = {
    "annotation_path": "/mnt/data/vmo-ai-task/anhdh35/JanusVLN/data/labels/train_r2r_rxr_episode.json",
    "data_path": "/mnt/data/vmo-ai-task/anhdh35/JanusVLN",
    "tag": "train_r2r_rxr"
}

# Action-isolated episodes: actions removed from the token stream, supervised
# via single-token readout labels (see create_data/build_episode_isolated.py).
TRAIN_R2R_RxR_ISOLATED = {
    "annotation_path": "/mnt/data/vmo-ai-task/anhdh35/JanusVLN/data/labels/train_r2r_rxr_episode_isolated.json",
    "data_path": "/mnt/data/vmo-ai-task/anhdh35/JanusVLN",
    "tag": "train_r2r_rxr_episode_isolated"
}

data_dict = {
    # "cambrian_737k": CAMBRIAN_737K,
    # "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    # "mp_doc": MP_DOC,
    # "clevr_mc": CLEVR_MC,
    # "videochatgpt": VIDEOCHATGPT,
    "train_r2r_rxr": TRAIN_R2R_RxR,
    "train_r2r_rxr_episode_isolated": TRAIN_R2R_RxR_ISOLATED,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for name in dataset_names:
        sampling_rate = parse_sampling_rate(name)
        name = re.sub(r"%(\d+)$", "", name)
        if name not in data_dict:
            raise ValueError(f"Unknown dataset: {name}")
        config = data_dict[name].copy()
        config["sampling_rate"] = sampling_rate
        config_list.append(config)
    return config_list


if __name__ == "__main__":
    configs = data_list(["spatial_ttt_data_97k"])
    for c in configs:
        print(c)

