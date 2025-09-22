import base64
import io
import logging
import os
import sys
import time
import traceback

import h5py
import numpy as np
import torch

from flask import Flask, jsonify, request
from flask_cors import CORS
from models.robobrain_robotics_dit import RoboBrainRobotics, RoboBrainRoboticsConfig
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
ROBOT_CONST = ENGINE_CONFIG.robot_const

model = None


def load_model():
    global model
    try:
        logger.info("Start load model.")
        config = RoboBrainRoboticsConfig.from_pretrained(ENGINE_CONFIG.model)
        config.training = False
        device_id = os.environ.get("EGL_DEVICE_ID", "0")
        device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        model = RoboBrainRobotics.from_pretrained(
            ENGINE_CONFIG.model,
            config=config,
            torch_dtype=torch.float32,
            # attn_implementation="flash_attention_2",
            device_map=device,
        ).to(torch.float32)
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()
            logger.info(f"Load model to GPU-{torch.cuda.get_device_name()} done. ")
        else:
            logger.info("Load model to CPU done.")
        return True
    except Exception as e:
        logger.error(f"Load model error: {e}")
        logger.error(traceback.format_exc())
        return False


def transform(x, scale, offset, clip=True):
    x_norm = x * scale + offset
    if clip:
        np.clip(x_norm, -1, 1, out=x_norm)
    return x_norm


def inverse_transform(x_norm, scale, offset):
    x_norm = np.asarray(x_norm)
    return (x_norm - offset) / scale


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


@app.route('/health', methods=['GET'])
def health_check():
    try:
        if model is None:
            return (
                jsonify(
                    {"status": "error", "message": "model not loaded", "timestamp": time.time()}
                ),
                503,
            )

        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                "memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB",
            }
        else:
            gpu_info = {"available": False}

        return jsonify(
            {
                "status": "healthy",
                "model_loaded": True,
                "gpu_info": gpu_info,
                "timestamp": time.time(),
            }
        )
    except Exception as e:
        logger.error(f"health check failed: {e}")
        return jsonify({"status": "error", "message": str(e), "timestamp": time.time()}), 500


@app.route('/info', methods=['GET'])
def service_info():
    return jsonify(
        {
            "service_name": "RoboBrain Robotics API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "info": "/info",
                "replay": "/replay",
                "infer": "/infer",
            },
            "model_info": {
                "model_path": ENGINE_CONFIG.model,
                "config_path": ENGINE_CONFIG.model,
                "action_dim": (
                    getattr(model.config, 'action_dim', 'unknown') if model else 'unknown'
                ),
                "action_horizon": (
                    getattr(model.config, 'action_horizon', 'unknown') if model else 'unknown'
                ),
            },
            "timestamp": time.time(),
        }
    )


@app.route('/replay', methods=['GET'])
def replay_api():
    try:
        with h5py.File(ENGINE_CONFIG.replay_file, 'r') as f:
            action = f['action'][:]
            qpos = f['qpos'][:]

        return jsonify({"success": True, "action": action.tolist(), "qpos": qpos.tolist()})
    except Exception as e:
        logger.error(f"Access ground truth qpos failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/infer', methods=['POST'])
def infer_api():
    start_time = time.time()
    try:
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
            state = np.array(data['qpos'])
            eef_pose = np.array(data['eef_pose'])
            scale = np.array(ROBOT_CONST['qpos']['scale_'])
            offset = np.array(ROBOT_CONST['qpos']['offset_'])
            state_norm = transform(state, scale, offset)
            state_tensor = torch.tensor(state_norm, dtype=torch.float32)
            instruction = data.get('instruction')
            images = data.get('images')
            if torch.cuda.is_available():
                state_tensor = state_tensor.cuda()
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
        logger.info(
            f"Inference start. state.shape: {state_tensor.shape}, instruction: {instruction}, num_images: {len(images_tensor) if images_tensor else 0}"
        )
        with torch.no_grad():
            result = model.get_action(
                state=state_tensor, instruction=[instruction], image=images_tensor
            )
        delta_actions = result.data['action_pred'].squeeze(0).cpu().numpy()
        if delta_actions is not None:
            try:
                scale = np.array(ROBOT_CONST['action']['scale_'])
                offset = np.array(ROBOT_CONST['action']['offset_'])
                eef_pose_norm = transform(eef_pose, scale, offset)
                actions = []
                current_eef_pose = eef_pose_norm
                for i in range(10):
                    current_eef_pose = current_eef_pose + delta_actions[i]
                    actions.append(current_eef_pose.tolist())

                actions_denorm = inverse_transform(np.array(actions), scale, offset)
                gt_delta_actions = [actions_denorm[0] - eef_pose_norm]
                for i in range(1, actions_denorm.shape[0]):
                    gt_delta_actions.append(actions_denorm[i] - actions_denorm[i - 1])
                gt_delta_actions = np.array(gt_delta_actions)
                for i in range(1, actions_denorm.shape[0]):
                    for col in [6, 13]:
                        actions_denorm[i, col] = actions_denorm[0, col]

            except Exception as e:
                logger.error(f"Action de-norm failed: {e}")

        processing_time = time.time() - start_time
        logger.info(f"Inference cost: {processing_time:.2f}s")
        return jsonify(
            {
                "success": True,
                "result": actions_denorm.tolist(),
                "delta_actions": gt_delta_actions.tolist(),
                "processing_time": processing_time,
            }
        )

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Inference error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e), "processing_time": processing_time}), 500


@app.errorhandler(404)
def not_found(error):
    return (
        jsonify(
            {
                "success": False,
                "error": "API not exist.",
                "available_endpoints": ["/health", "/info", "/infer", "/replay"],
            }
        ),
        404,
    )


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error."}), 500


def main():
    if not load_model():
        logger.error("Model load failed, exiting.")
        sys.exit(1)

    logger.info(f"Serve URL: http://{ENGINE_CONFIG['host']}:{ENGINE_CONFIG['port']}")
    logger.info(f"Available API:")
    logger.info(f"  - GET  /health  - health check")
    logger.info(f"  - GET  /info    - service info")
    logger.info(f"  - GET  /replay  - get replay ground truth qpos")
    logger.info(f"  - POST /infer   - inference api")

    app.run(
        host=ENGINE_CONFIG['host'],
        port=ENGINE_CONFIG['port'],
        debug=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
