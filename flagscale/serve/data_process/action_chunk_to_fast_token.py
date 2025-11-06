#!/usr/bin/env python3
"""
Action Chunk to Fast Token 转换程序
"""

import os
import sys

from typing import Dict, List

import numpy as np

from scipy.fft import idct

# from action_token.tokenizer import FASTTokenizer
from transformers import AutoProcessor
from flagscale.runner.utils import logger

script_dir = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(script_dir, "fast")


class ActionChunkProcessor:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = tokenizer_path):
        self.max_len = max_len
        self.fast_tokenizer = AutoProcessor.from_pretrained(
            fast_tokenizer_path, trust_remote_code=True
        )

    def create_dummy_batch(
        self, batch_size: int = 2, action_horizon: int = 50, action_dim: int = 32
    ) -> Dict:
        dummy_batch = {
            "prompt": ["Pick up the red block from the table", "Move the blue object to the left"][
                :batch_size
            ],
            "state": np.random.randn(batch_size, action_dim).astype(np.float32),
            "actions": np.random.randn(batch_size, action_horizon, action_dim).astype(np.float32),
        }
        return dummy_batch

    def process_action_chunk_to_fast_token(self, action_chunk: np.ndarray) -> List[List[int]]:
        fast_tokens = self.fast_tokenizer(action_chunk)
        return fast_tokens

    def extract_actions_from_tokens(
        self, action_tokens: List[List[int]], action_horizon: int, action_dim: int
    ) -> np.ndarray:
        assert (
            action_horizon is not None and action_dim is not None
        ), "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."

        decoded_actions = []
        output_dims = []
        for token in action_tokens:
            try:
                decoded_tokens = self.fast_tokenizer.bpe_tokenizer.decode(token)
                decoded_dct_coeff = (
                    np.array(list(map(ord, decoded_tokens))) + self.fast_tokenizer.min_token
                )
                output_dim = len(decoded_dct_coeff)
                output_dims.append(output_dim)
                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, action_dim)
                assert decoded_dct_coeff.shape == (
                    action_horizon,
                    action_dim,
                ), f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({action_horizon}, {action_dim})"
            except Exception as e:
                logger.info(f"Error decoding tokens: {e}")
                logger.info(f"Tokens: {token}")
                decoded_dct_coeff = np.zeros((action_horizon, action_dim))
            decoded_actions.append(
                idct(decoded_dct_coeff / self.fast_tokenizer.scale, axis=0, norm="ortho")
            )
        return np.stack(decoded_actions), output_dims

    def _extract_actions_from_tokens(
        self, action_tokens: List[List[int]], action_horizon: int, action_dim: int
    ) -> np.ndarray:
        assert (
            action_horizon is not None and action_dim is not None
        ), "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."

        decoded_actions = []
        output_dims = []
        for token in action_tokens:
            try:
                decoded_tokens = self.fast_tokenizer.bpe_tokenizer.decode(token)
                decoded_dct_coeff = (
                    np.array(list(map(ord, decoded_tokens))) + self.fast_tokenizer.min_token
                )
                output_dim = len(decoded_dct_coeff)
                output_dims.append(output_dim)
                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, action_dim)
                assert decoded_dct_coeff.shape == (
                    action_horizon,
                    action_dim,
                ), f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({action_horizon}, {action_dim})"
            except Exception as e:
                logger.info(f"Error decoding tokens: {e}")
                logger.info(f"Tokens: {token}")
                if len(decoded_dct_coeff.shape) == 2:
                    results = []
                    output_horizon = decoded_dct_coeff.shape[0]
                    needed_rows = 30
                    zero_rows_needed = max(0, needed_rows - output_horizon)
                    results.extend(decoded_dct_coeff[:needed_rows])
                    zero_vector = [0] * action_dim
                    results.extend([zero_vector for _ in range(zero_rows_needed)])
                    decoded_dct_coeff = np.array(results)
                else:
                    decoded_dct_coeff = np.zeros((action_horizon, action_dim))
            decoded_actions.append(
                idct(decoded_dct_coeff / self.fast_tokenizer.scale, axis=0, norm="ortho")
            )
        return np.stack(decoded_actions), output_dims

    def _extract_actions_from_tokens_v2(
        self, action_tokens: List[List[int]], action_horizon: int, action_dim: int
    ) -> np.ndarray:
        assert (
            action_horizon is not None and action_dim is not None
        ), "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."

        decoded_actions = []
        output_dims = []
        zero_nums = []
        for token in action_tokens:
            try:
                decoded_tokens = self.fast_tokenizer.bpe_tokenizer.decode(token)
                decoded_dct_coeff = (
                    np.array(list(map(ord, decoded_tokens))) + self.fast_tokenizer.min_token
                )
                output_dim = len(decoded_dct_coeff)
                output_dims.append(output_dim)
                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, action_dim)
                assert decoded_dct_coeff.shape == (
                    action_horizon,
                    action_dim,
                ), f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({action_horizon}, {action_dim})"
            except Exception as e:
                logger.info(f"Error decoding tokens: {e}")
                logger.info(f"Tokens: {token}")
                if len(decoded_dct_coeff.shape) == 2:
                    decoded_dct_coeff = decoded_dct_coeff
                else:
                    decoded_dct_coeff = np.zeros((action_horizon, action_dim))
            tmp_actions = idct(decoded_dct_coeff / self.fast_tokenizer.scale, axis=0, norm="ortho")
            if tmp_actions.shape == (action_horizon, action_dim):
                decoded_actions.append(tmp_actions)
                zero_nums.append(0)
            else:
                if action_horizon > tmp_actions.shape[0]:
                    delta_size = action_horizon - tmp_actions.shape[0]
                    new_rows = np.zeros((delta_size, action_dim))
                    pad_actions = np.vstack((tmp_actions, new_rows))
                    decoded_actions.append(pad_actions)
                    zero_nums.append(delta_size)
                else:
                    truncate_actions = tmp_actions[:action_horizon, :]
                    decoded_actions.append(truncate_actions)
                    zero_nums.append(0)

            # decoded_actions.append(idct(decoded_dct_coeff / self.fast_tokenizer.scale, axis=0, norm="ortho"))
        return np.stack(decoded_actions), output_dims, zero_nums

    def validate_encode_decode(
        self, original_actions: np.ndarray, action_horizon: int, action_dim: int
    ) -> Dict:
        fast_tokens = self.fast_tokenizer(original_actions[None])[0]
        decoded_actions = self.fast_tokenizer.decode(
            [fast_tokens], time_horizon=action_horizon, action_dim=action_dim
        )[0]
        mse_error = np.mean((original_actions - decoded_actions) ** 2)
        mae_error = np.mean(np.abs(original_actions - decoded_actions))

        logger.info(f"  - MSE: {mse_error:.6f}")
        logger.info(f"  - MAE: {mae_error:.6f}")

        return {
            "original_shape": original_actions.shape,
            "decoded_shape": decoded_actions.shape,
            "mse_error": mse_error,
            "mae_error": mae_error,
            "fast_tokens_length": len(fast_tokens),
        }

    def process_batch(self, dummy_batch: Dict) -> Dict:
        batch_size = len(dummy_batch["prompt"])
        results = {"processed_samples": []}

        for i in range(batch_size):
            sample_result = {"sample_id": i}

            try:
                action_chunk = dummy_batch["actions"][i : i + 1]
                fast_tokens = self.process_action_chunk_to_fast_token(action_chunk)
                sample_result["fast_tokens"] = fast_tokens[0]
                sample_result["fast_tokens_length"] = len(fast_tokens[0])
                tokens, token_masks, ar_masks, loss_masks = self.tokenize_with_context(
                    dummy_batch["prompt"][i], dummy_batch["state"][i], dummy_batch["actions"][i]
                )
                sample_result["context_tokens"] = tokens
                extracted_actions = self.extract_actions_from_tokens(
                    tokens, dummy_batch["actions"].shape[1], dummy_batch["actions"].shape[2]
                )
                sample_result["extracted_actions"] = extracted_actions
                validation_result = self.validate_encode_decode(
                    dummy_batch["actions"][i],
                    dummy_batch["actions"].shape[1],
                    dummy_batch["actions"].shape[2],
                )
                sample_result["validation"] = validation_result

                sample_result["success"] = True

            except Exception as e:
                sample_result["error"] = str(e)
                sample_result["success"] = False

            results["processed_samples"].append(sample_result)

        return results

    def print_results_summary(self, results: Dict):
        successful_samples = 0
        total_fast_tokens = 0
        total_mse_error = 0
        total_mae_error = 0

        for sample in results["processed_samples"]:
            sample_id = sample["sample_id"]

            if sample.get("success", False):
                successful_samples += 1
                logger.info(f"Sample {sample_id + 1}: ✅")
                logger.info(f"  - Fast tokens length: {sample['fast_tokens_length']}")
                logger.info(f"  - context tokens shape: {sample['context_tokens'].shape}")
                logger.info(f"  - extracted actions shape: {sample['extracted_actions'].shape}")

                validation = sample["validation"]
                logger.info(f"  - MSE: {validation['mse_error']:.6f}")
                logger.info(f"  - MAE: {validation['mae_error']:.6f}")

                total_fast_tokens += sample['fast_tokens_length']
                total_mse_error += validation['mse_error']
                total_mae_error += validation['mae_error']
            else:
                logger.info(f"Sample {sample_id + 1}: ❌")

        logger.info(f"\n📊 Statistics: {successful_samples}/{len(results['processed_samples'])} success")
        if successful_samples > 0:
            logger.info(f"AVG Fast tokens len: {total_fast_tokens/successful_samples:.1f}")
            logger.info(f"AVG MSE: {total_mse_error/successful_samples:.6f}")
            logger.info(f"AVG MAE: {total_mae_error/successful_samples:.6f}")


def main():
    logger.info("🚀 Action Chunk to Fast Token")
    try:
        processor = ActionChunkProcessor(max_len=256)
        dummy_batch = processor.create_dummy_batch(batch_size=3, action_horizon=50, action_dim=32)
        results = processor.process_batch(dummy_batch)
        processor.print_results_summary(results)

        logger.info("\n🎉 Done!")

    except Exception as e:
        logger.info(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
