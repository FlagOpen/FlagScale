# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Inference API."""


import torch

from megatron.core import mpu
from .communication import broadcast_float_list

from .generation_single_thread import (
    score_and_return_on_first_stage,
    generate_tokens_probs_and_return_on_first_stage_stream_main_process,
    generate_tokens_probs_and_return_on_first_stage_stream_sub_process,
)
from .tokenization import (
    tokenize_prompts,
    detokenize_generations)

def generate_and_post_process_single_thread(model,
                              prompts=None,
                              tokens_to_generate=0,
                              return_output_log_probs=False,
                              top_k_sampling=0,
                              top_p_sampling=0.0,
                              top_p_decay=0.0,
                              top_p_bound=0.0,
                              temperature=1.0,
                              add_BOS=False,
                              use_eod_token_for_early_termination=True,
                              stop_on_double_eol=False,
                              stop_on_eol=False,
                              prevent_newline_after_colon=False,
                              random_seed=-1,
                              stream=False):
    """Run inference and post-process outputs, i.e., detokenize,
    move to cpu and convert to list."""

    if not stream:
    # Main inference.
        tokens, lengths, output_log_probs = generate(
            model,
            prompts=prompts,
            tokens_to_generate=tokens_to_generate,
            return_output_log_probs=return_output_log_probs,
            top_k_sampling=top_k_sampling,
            top_p_sampling=top_p_sampling,
            top_p_decay=top_p_decay,
            top_p_bound=top_p_bound,
            temperature=temperature,
            add_BOS=add_BOS,
            use_eod_token_for_early_termination=use_eod_token_for_early_termination,
            stop_on_double_eol=stop_on_double_eol,
            stop_on_eol=stop_on_eol,
            prevent_newline_after_colon=prevent_newline_after_colon,
            random_seed=random_seed,
            )

        return None
    else:
        tokens = generate_stream(
            model,
            prompts=prompts,
            tokens_to_generate=tokens_to_generate,
            return_output_log_probs=return_output_log_probs,
            top_k_sampling=top_k_sampling,
            top_p_sampling=top_p_sampling,
            top_p_decay=top_p_decay,
            top_p_bound=top_p_bound,
            temperature=temperature,
            add_BOS=add_BOS,
            use_eod_token_for_early_termination=use_eod_token_for_early_termination,
            stop_on_double_eol=stop_on_double_eol,
            stop_on_eol=stop_on_eol,
            prevent_newline_after_colon=prevent_newline_after_colon,
            random_seed=random_seed
            )

        # Only post-process on first stage.
        if mpu.is_pipeline_first_stage():
            print(f"device is {torch.cuda.current_device()}, stream function return is {tokens}")
            return tokens
          
        return None

def generate(model,
             prompts=None,
             tokens_to_generate=0,
             return_output_log_probs=False,
             top_k_sampling=0,
             top_p_sampling=0.0,
             top_p_decay=0.0,
             top_p_bound=0.0,
             temperature=1.0,
             add_BOS=False,
             use_eod_token_for_early_termination=True,
             stop_on_double_eol=False,
             stop_on_eol=False,
             prevent_newline_after_colon=False,
             random_seed=-1, 
             ):
    """Given prompts and input parameters, run inference and return:
       tokens: prompts plus the generated tokens.
       lengths: length of the prompt + generations. Note that we can
           discard tokens in the tokens tensor that are after the
           corresponding length.
       output_log_probs: log probs of the tokens.
    """

    # Make sure input params are avaialble to all ranks.
    values = [tokens_to_generate,
              return_output_log_probs,
              top_k_sampling, top_p_sampling, top_p_decay, top_p_bound,
              temperature, add_BOS, use_eod_token_for_early_termination,
              stop_on_double_eol,
              stop_on_eol,
              prevent_newline_after_colon,
              random_seed]
    values_float_tensor = broadcast_float_list(len(values), float_list=values)
    tokens_to_generate = int(values_float_tensor[0].item())
    return_output_log_probs = bool(values_float_tensor[1].item())
    top_k_sampling = int(values_float_tensor[2].item())
    top_p_sampling = values_float_tensor[3].item()
    top_p_decay = values_float_tensor[4].item()
    top_p_bound = values_float_tensor[5].item()
    temperature = values_float_tensor[6].item()
    add_BOS = bool(values_float_tensor[7].item())
    use_eod_token_for_early_termination = bool(values_float_tensor[8].item())
    stop_on_double_eol = bool(values_float_tensor[9].item())
    stop_on_eol = bool(values_float_tensor[10].item())
    prevent_newline_after_colon = bool(values_float_tensor[11].item())
    random_seed = int(values_float_tensor[12].item())

    if random_seed != -1:
        torch.random.manual_seed(random_seed)

    # Tokenize prompts and get the batch.
    # Note that these tensors are broadcaseted to all ranks.
    if torch.distributed.get_rank() == 0:
        assert prompts is not None
    
    context_tokens_tensor, context_length_tensor = tokenize_prompts(
        prompts=prompts, tokens_to_generate=tokens_to_generate, add_BOS=add_BOS)

    if tokens_to_generate == 0:
        return score_and_return_on_first_stage(
            model, context_tokens_tensor, context_length_tensor)
    
    # Main inference function.
    # Note that the outputs are available on the first stage.
    return generate_tokens_probs_and_return_on_first_stage_stream_sub_process(
        model, context_tokens_tensor, context_length_tensor,
        return_output_log_probs=return_output_log_probs,
        top_k=top_k_sampling,
        top_p=top_p_sampling,
        top_p_decay=top_p_decay,
        top_p_bound=top_p_bound,
        temperature=temperature,
        use_eod_token_for_early_termination=use_eod_token_for_early_termination,
        stop_on_double_eol=stop_on_double_eol,
        stop_on_eol=stop_on_eol,
        prevent_newline_after_colon=prevent_newline_after_colon,
        )

def generate_stream(model,
             prompts=None,
             tokens_to_generate=0,
             return_output_log_probs=False,
             top_k_sampling=0,
             top_p_sampling=0.0,
             top_p_decay=0.0,
             top_p_bound=0.0,
             temperature=1.0,
             add_BOS=False,
             use_eod_token_for_early_termination=True,
             stop_on_double_eol=False,
             stop_on_eol=False,
             prevent_newline_after_colon=False,
             random_seed=-1
             ):
    """Given prompts and input parameters, run inference and return:
       tokens: prompts plus the generated tokens.
       lengths: length of the prompt + generations. Note that we can
           discard tokens in the tokens tensor that are after the
           corresponding length.
       output_log_probs: log probs of the tokens.
    """

    # Make sure input params are avaialble to all ranks.
    values = [tokens_to_generate,
              return_output_log_probs,
              top_k_sampling, top_p_sampling, top_p_decay, top_p_bound,
              temperature, add_BOS, use_eod_token_for_early_termination,
              stop_on_double_eol,
              stop_on_eol,
              prevent_newline_after_colon,
              random_seed]
    values_float_tensor = broadcast_float_list(len(values), float_list=values)
    tokens_to_generate = int(values_float_tensor[0].item())
    return_output_log_probs = bool(values_float_tensor[1].item())
    top_k_sampling = int(values_float_tensor[2].item())
    top_p_sampling = values_float_tensor[3].item()
    top_p_decay = values_float_tensor[4].item()
    top_p_bound = values_float_tensor[5].item()
    temperature = values_float_tensor[6].item()
    add_BOS = bool(values_float_tensor[7].item())
    use_eod_token_for_early_termination = bool(values_float_tensor[8].item())
    stop_on_double_eol = bool(values_float_tensor[9].item())
    stop_on_eol = bool(values_float_tensor[10].item())
    prevent_newline_after_colon = bool(values_float_tensor[11].item())
    random_seed = int(values_float_tensor[12].item())

    if random_seed != -1:
        torch.random.manual_seed(random_seed)

    # Tokenize prompts and get the batch.
    # Note that these tensors are broadcaseted to all ranks.
    if torch.distributed.get_rank() == 0:
        assert prompts is not None
    
    context_tokens_tensor, context_length_tensor = tokenize_prompts(
        prompts=prompts, tokens_to_generate=tokens_to_generate, add_BOS=add_BOS)

    if tokens_to_generate == 0:
        return score_and_return_on_first_stage(
            model, context_tokens_tensor, context_length_tensor)
    
    # Main inference function.
    # Note that the outputs are available on the first stage.
    return generate_tokens_probs_and_return_on_first_stage_stream_main_process(
        model, context_tokens_tensor, context_length_tensor,
        return_output_log_probs=return_output_log_probs,
        top_k=top_k_sampling,
        top_p=top_p_sampling,
        top_p_decay=top_p_decay,
        top_p_bound=top_p_bound,
        temperature=temperature,
        use_eod_token_for_early_termination=use_eod_token_for_early_termination,
        stop_on_double_eol=stop_on_double_eol,
        stop_on_eol=stop_on_eol,
        prevent_newline_after_colon=prevent_newline_after_colon,
        )

