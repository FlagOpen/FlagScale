

import os
import sys
import torch
import h5py
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from models.robobrain_robotics_dit import RoboBrainRobotics, RoboBrainRoboticsConfig
import numpy as np
import time
import traceback
import base64
from PIL import Image
import io

from flagscale import serve

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log',  # 指定日志输出文件
    filemode='a'
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
        logger.info("Start load model.")
        config = RoboBrainRoboticsConfig.from_pretrained(ENGINE_CONFIG.model)
        config.training = False
        devide_id = os.environ.get("EGL_DEVICE_ID", "0")
        device = f"cuda:{devide_id}" if torch.cuda.is_available() else "cpu"
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

     
ACTION_QUANTILES_LOW = {
    "action": {
        "quantiles_low_": [-0.008952002273872495, -0.0075089819729328156, -0.008938997983932495, -0.09321874380111694, -0.04934048652648926, -0.08562564849853516, -0.0356999933719635, -0.008957445621490479, -0.00789663940668106, -0.008612995967268944, -0.08561980724334717, -0.04220205545425415, -0.0927119106054306, -0.021585147362202406],
        "quantiles_high_": [0.009830005466938019, 0.008996007964015007, 0.012891992926597595, 0.08754575252532959, 0.04377281665802002, 0.09669125080108643, 0.04218482971191406, 0.00984799861907959, 0.007583707571029663, 0.01218000054359436, 0.08659172058105469, 0.040928006172180176, 0.0775449275970459, 0.03149998188018799],
        "scale_": [106.4848318002756, 121.17540185642973, 91.61284029974587, 11.064118950366046, 21.47920562161597, 10.969909525578773, 1.0, 106.3521239324965, 129.19598707443794, 96.1861863966185, 11.613623716892482, 24.058682506464063, 11.74695773541818, 1.0],
        "offset_": [-0.04674754359100264, -0.09009609189717949, -0.181073005258242, 0.03138326981925532, 0.0597944555730352, -0.06069438290205942, 0.0, -0.047356633144646554, 0.020214122717063576, -0.17154876445894596, -0.005643775962900666, 0.01532585329409053, 0.08908289545186188, 0.0]
    },
    "qpos": {
        "quantiles_low_": [-0.019729137420654297, -0.04380178451538086, -0.03898739814758301, -0.04760468006134033, -0.03961634635925293, -0.0476837158203125, -0.0356999933719635, -0.0247955322265625, -0.042725563049316406, -0.03492289036512375, -0.041962623596191406, -0.040958523750305176, -0.035777658224105835, -0.021585147362202406],
        "quantiles_high_": [0.028228759765625, 0.05067610740661621, 0.0396728515625, 0.045777320861816406, 0.04193645715713501, 0.04615795612335205, 0.04218482971191406, 0.020218849182128906, 0.04816293716430664, 0.03733110427856445, 0.04163992404937744, 0.039668649435043335, 0.038447558879852295, 0.03149998188018799],
        "scale_": [41.70323763777994, 21.168971360638828, 25.425799601620582, 21.417401277059486, 24.523985301844952, 21.31249098029433, 25.678940569162076, 44.430235251694164, 22.00498165608864, 27.680126656823465, 23.922713088264057, 24.805529859604945, 26.94501691721127, 37.675327380566195],
        "offset_": [-0.17723109375803703, -0.07276127804902932, -0.008714227710960976, 0.01956853553975879, -0.028449304172882384, 0.01625876332732834, -0.08326199188186922, 0.10167133001713591, -0.05982476885373389, -0.033329971471014797, 0.0038598047225062437, 0.015997883893533293, -0.035970393893265395, -0.18677250657125954]
    }
}


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
            return jsonify({
                "status": "error",
                "message": "model not loaded",
                "timestamp": time.time()
            }), 503
        
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                "memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB"
            }
        else:
            gpu_info = {"available": False}
        
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "gpu_info": gpu_info,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"health check failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/info', methods=['GET'])
def service_info():
    return jsonify({
        "service_name": "RoboBrain Robotics API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "info": "/info", 
            "replay": "/replay",
            "infer": "/infer"
        },
        "model_info": {
            "model_path": ENGINE_CONFIG.model,
            "config_path": ENGINE_CONFIG.model,
            "action_dim": getattr(model.config, 'action_dim', 'unknown') if model else 'unknown',
            "action_horizon": getattr(model.config, 'action_horizon', 'unknown') if model else 'unknown'
        },
        "timestamp": time.time()
    })

@app.route('/replay', methods=['GET'])
def replay_api():
    try: 
        # /share/project/section/RoboBrain_Robotic_AGX/videos/train/1/1_cam_right_wrist.mp4
        # /share/project/section/RoboBrain_Robotic_AGX/videos/train/1/1.hdf5
        with h5py.File('/share/project/section/RoboBrain_Robotic_newdragon/videos/train/0/0.hdf5', 'r') as f:
            action = f['action'][:]
            qpos = f['qpos'][:]

        return jsonify({
            "success": True, 
            "action": action.tolist(),
            "qpos": qpos.tolist(),
        })
    except Exception as e:
        logger.error(f"Access ground truth qpos failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/infer', methods=['POST'])
def infer_api():
    start_time = time.time()
    try:
        if model is None:
            return jsonify({
                "success": False,
                "error": "model not loaded"
            }), 503
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "request format error"
            }), 400
        if 'qpos' not in data:
            return jsonify({
                "success": False,
                "error": "request requires: qpos"
            }), 400
        if 'eef_pose' not in data:
            return jsonify({
                "success": False,
                "error": "request requires: eef_pose"
            }), 400
        try:
            state = np.array(data['qpos'])
            eef_pose = np.array(data['eef_pose'])
            scale = np.array(ACTION_QUANTILES_LOW['qpos']['scale_'])
            offset = np.array(ACTION_QUANTILES_LOW['qpos']['offset_'])
            state_norm = transform(state, scale, offset)
            state_tensor = torch.tensor(state_norm, dtype=torch.float32)
            instruction = data.get('instruction')
            images = data.get('images')
            if torch.cuda.is_available():
                state_tensor = state_tensor.cuda()
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"state parameters process error: {e}"
            }), 400
        if instruction is None:
            return jsonify({
                "success": False,
                "error": "request requires instruction"
            }), 400
        images_tensor = None
        if images is not None:
            try:
                images_tensor = process_images(images)
                if torch.cuda.is_available():
                    for sample in images_tensor:
                        for key in sample:
                            sample[key] = sample[key].cuda()
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"image process failed: {e}"
                }), 400
        logger.info(f"Inference start. state.shape: {state_tensor.shape}, instruction: {instruction}, num_images: {len(images_tensor) if images_tensor else 0}")
        with torch.no_grad():
            result = model.get_action(
                state=state_tensor,
                instruction=[instruction],
                image=images_tensor,
            )
        delta_actions = result.data['action_pred'].squeeze(0).cpu().numpy()
        if delta_actions is not None:
            try:
                scale = np.array(ACTION_QUANTILES_LOW['action']['scale_'])
                offset = np.array(ACTION_QUANTILES_LOW['action']['offset_'])
                eef_pose_norm = transform(eef_pose, scale, offset)
                actions = []
                current_eef_pose = eef_pose_norm
                for i in range(10):
                    current_eef_pose = current_eef_pose + delta_actions[i]
                    actions.append(current_eef_pose.tolist())

                actions_denorm = inverse_transform(
                    np.array(actions), scale, offset
                )
                gt_delta_actions = [actions_denorm[0] - eef_pose_norm]
                for i in range(1, actions_denorm.shape[0]):
                    gt_delta_actions.append(
                        actions_denorm[i] - actions_denorm[i - 1]
                    )
                gt_delta_actions = np.array(gt_delta_actions)
                # actions_denorm shape: [10, action_dim]
                for i in range(1, actions_denorm.shape[0]):
                    for col in [6, 13]:
                        actions_denorm[i, col] = actions_denorm[0, col]

                    # for col in [3, 4, 5, 10, 11, 12]:
                    #     actions_denorm[i, col] = actions_denorm[0, col]
                # horizon = 10
                
            except Exception as e:
                logger.error(f"动作数据反归一化失败: {e}")
                
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        logger.info(f"Inference cost: {processing_time:.2f}s")
        return jsonify({
            "success": True, 
            "result": actions_denorm.tolist(),
            "delta_actions": gt_delta_actions.tolist(),
            "processing_time": processing_time
        })
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"推理失败: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "processing_time": processing_time
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "API not exist.",
        "available_endpoints": ["/health", "/info", "/infer", "/replay"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error."
    }), 500


def main():
    import pdb; pdb.set_trace()
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
        debug=ENGINE_CONFIG['flask_debug'],
        threaded=ENGINE_CONFIG['flask_threaded']
    ) 


if __name__ == "__main__":
    main()
