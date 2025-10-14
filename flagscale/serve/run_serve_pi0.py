import base64
import io
import logging
import sys
import time
import traceback

import numpy as np
import torch

from flask import Flask, jsonify, request
from flask_cors import CORS
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from PIL import Image

from flagscale import serve

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

serve.load_args()
TASK_CONFIG = serve.task_config
ENGINE_CONFIG = TASK_CONFIG.serve[0].engine_args

model = None


def load_model():
    global model
    try:
        lerobot_dataset = LeRobotDataset(
            ENGINE_CONFIG.dataset_repo_id,
            episodes=ENGINE_CONFIG.episodes,
            video_backend=ENGINE_CONFIG.video_backend,
        )
        cfg = PreTrainedConfig.from_pretrained(ENGINE_CONFIG.model)
        cfg.repo_id = ENGINE_CONFIG.tokenizer_path
        cfg.pretrained_path = ENGINE_CONFIG.model
        policy = make_policy(cfg, ds_meta=lerobot_dataset.meta)
        if torch.cuda.is_available():
            policy = policy.cuda()
            logger.info(f"Load model to GPU-{torch.cuda.get_device_name()} done. ")
        else:
            logger.info("Load model to CPU done.")
        if ENGINE_CONFIG.compile_model:
            policy = torch.compile(policy, mode=ENGINE_CONFIG.compile_mode)
        model = policy
        return True
    except Exception as e:
        logger.error(f"Load model error: {e}")
        logger.error(traceback.format_exc())
        return False


def decode_image_base64(image_base64):
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        # shape to: [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        raise ValueError(f"Image decode error: {e}")


def process_images(images_json):
    # images_json: List[Dict[str, base64]]
    processed = []
    for i, sample in enumerate(images_json):
        try:
            sample_dict = {}
            for k, v in sample.items():
                sample_dict[k] = decode_image_base64(v)
            processed.append(sample_dict)
        except Exception as e:
            logger.error(f"Image[{i}] decode error: {e}")
            raise ValueError(f"Image[{i}] decode error: {e}")
    return processed


@app.route('/infer', methods=['POST'])
def infer_api():
    start_time = time.time()

    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 503
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Request format error"}), 400
    if 'qpos' not in data:
        return jsonify({"success": False, "error": "Request requires: qpos"}), 400
    if 'eef_pose' not in data:
        return jsonify({"success": False, "error": "Request requires: eef_pose"}), 400
    try:
        qpos = torch.tensor(data['qpos']).cuda()
        eef_pose = torch.tensor(data['eef_pose']).cuda()
        instruction = data.get('instruction')
        images = data.get('images')
    except Exception as e:
        return (
            jsonify({"success": False, "error": f"State parameters processing error: {e}"}),
            400,
        )
    if instruction is None:
        return jsonify({"success": False, "error": "Request requires instruction"}), 400
    images_tensor = None
    if images is not None:
        try:
            images_tensor = process_images(images)
            if torch.cuda.is_available():
                for sample in images_tensor:
                    for key in sample:
                        sample[key] = sample[key].cuda()
        except Exception as e:
            return jsonify({"success": False, "error": f"image process failed: {e}"}), 400

    with torch.no_grad():
        batch = {
            # [1, 3, 480, 640]
            "observation.images.cam_high": sample["cam_high"][None,],
            "observation.images.cam_left_wrist": sample["cam_left_wrist"][None,],
            "observation.images.cam_right_wrist": sample["cam_right_wrist"][None,],
            # [1,14]
            "observation.state": eef_pose,
            # [1,14]
            "action": qpos,
            # [str]
            "task": [instruction],
        }
        images, img_masks = model.prepare_images(batch)
        state = model.prepare_state(batch)
        lang_tokens, lang_masks = model.prepare_language(batch)
        actions = model.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=None
        )

    processing_time = time.time() - start_time
    logger.info(f"Inference cost: {processing_time:.2f}s")
    return jsonify({"success": True, "qpos": actions.tolist(), "processing_time": processing_time})


def main():
    if not load_model():
        logger.error("Model load failed, exiting.")
        sys.exit(1)

    logger.info(f"Serve URL: http://{ENGINE_CONFIG['host']}:{ENGINE_CONFIG['port']}")
    logger.info(f"Available API:")
    logger.info(f"  - POST /infer   - inference api")
    app.run(host=ENGINE_CONFIG['host'], port=ENGINE_CONFIG['port'], debug=False, threaded=True)


if __name__ == "__main__":
    main()
