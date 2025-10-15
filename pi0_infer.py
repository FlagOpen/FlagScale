import torch
import time

from flagscale.models.pi0.modeling_pi0 import PI0Policy, PI0PolicyConfig


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/models/lerobot/pi0"
TOKENIZER_PATH = "/models/google/paligemma-3b-pt-224"
STAT_PATH = "/models/lerobot/aloha_mobile_cabinet/meta/stats.json"
NUM_INFER = 100


def main():
    batch = {
        # 图像张量：4维 (1, 3, H, W)，float32（H=480, W=640 匹配常见图像尺寸，确保维度正确）
        "observation.images.camera0": torch.randn(1, 3, 480, 640, dtype=torch.float32),
        "observation.images.camera1": torch.randn(1, 3, 480, 640, dtype=torch.float32),
        "observation.images.camera2": torch.randn(1, 3, 480, 640, dtype=torch.float32),
        
        # 特征向量：2维 (1, 14)，float32
        "observation.state": torch.randn(1, 14, dtype=torch.float32),
        "observation.effort": torch.randn(1, 14, dtype=torch.float32),
        "action": torch.randn(1, 14, dtype=torch.float32),
        
        # 整数标量：1维 (1,)，int64
        "episode_index": torch.randint(0, 100, (1,), dtype=torch.int64),
        "frame_index": torch.randint(0, 1500, (1,), dtype=torch.int64),
        "index": torch.randint(0, 127500, (1,), dtype=torch.int64),
        "task_index": torch.randint(0, 10, (1,), dtype=torch.int64),
        
        # 浮点标量：1维 (1,)，float32
        "timestamp": torch.randn(1, dtype=torch.float32),
        
        # 布尔标量：1维 (1,)，bool（固定为False匹配目标）
        "next.done": torch.tensor([False], dtype=torch.bool),
        
        # 任务文本：固定字符串列表（完全匹配目标）
        "task": ["Open the top cabinet, store the pot inside it then close the cabinet."]
    }
    config = PI0PolicyConfig.from_pretrained(MODEL_PATH)
    t_s = time.time()
    policy = PI0Policy.from_pretrained(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        stat_path=STAT_PATH,
        config=config)
    policy = policy.eval().to(DEVICE)
    print(f"pi0_load: {(time.time() - t_s):.2f} s")
    
    t_s = time.time()
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens, lang_masks = policy.prepare_language(batch)
    print(f"feature_process: {((time.time() - t_s)*1000):.2f} ms")

    images = [i.to(DEVICE) for i in images]
    img_masks = [i.to(DEVICE) for i in img_masks]
    state = state.to(DEVICE)
    lang_tokens = lang_tokens.to(DEVICE)
    lang_masks = lang_masks.to(DEVICE)

    for _ in range(NUM_INFER):
        t_s = time.time()
        with torch.no_grad():
            actions = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None)
        print(f"sample_actions: {((time.time() - t_s)*1000):.2f} ms")
        original_action_dim = policy.config.action_feature["shape"][0]
        actions = actions[:, :10, :original_action_dim]
    print("actions: ", actions.shape)


if __name__ == "__main__":
    main()
