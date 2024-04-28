import os
import sys
import torch




# # Load Huggingface model.
# from transformers import MixtralForCausalLM

# model_path = ""
# model = MixtralForCausalLM.from_pretrained(model_path, device_map="cpu")

# for name, val in model.state_dict().items():
#     print("name:", name)
#     print("val:", torch.sum(val))




# # Load Megatron model.
# root_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.insert(0, os.path.join(root_path, "megatron"))
# from megatron.training.checkpointing import _load_base_checkpoint

# # load_dir=""
# state_dict, _, _ = _load_base_checkpoint(load_dir, rank0=True)

# # print(state_dict["args"])
# # print(state_dict['model'])

# print("state_dict.keys():", state_dict.keys())
# print("state_dict['model'].keys():", state_dict["model"].keys())
# # print("state_dict['model']:", state_dict['model'])

# for name, val in state_dict["model"].items():
#     if "_extra_state" in name:
#         continue
#     print("name:", name)
#     print("shape:", val.shape)
#     print("val:", torch.sum(val), val)
#     # if name == "decoder.layers.0.self_attention.linear_qkv.bias":
#     #     print("val:", val)
#     # else:    
#     #     print("val:", torch.sum(val))





# # Load Huggingface model.
# from transformers import MistralForCausalLM

# model_path = ""
# model = MistralForCausalLM.from_pretrained(model_path, device_map="cpu")

# for name, val in model.state_dict().items():
#     print("name:", name)
#     print("val:", torch.sum(val))




# # Load Huggingface model.
# from transformers import AutoModelForCausalLM

# # model_path = ""
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", trust_remote_code=True)

# for name, val in model.state_dict().items():
#     print("name:", name)
#     print("shape:", val.shape)
#     print("val:", torch.sum(torch.abs(val)), val)
#     # if "model.layers.0.self_attn." in name and "weight" in name:
#     #     print("val", val)
#     # else:
#     #     print("val:", torch.sum(val))








# # Load Megatron model.
# root_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.insert(0, os.path.join(root_path, "megatron"))
# from megatron.core import mpu
# from megatron.training.checkpointing import _load_base_checkpoint

# # load_dir=""
# tp_size = 1
# pp_size = 8
# ep_size = 1
# mpu.set_tensor_model_parallel_world_size(tp_size)
# mpu.set_pipeline_model_parallel_world_size(pp_size)
# mpu.set_expert_model_parallel_world_size(ep_size)
# # mpu.set_tensor_model_parallel_rank(0)
# # mpu.set_pipeline_model_parallel_rank(0)
# # mpu.set_expert_model_parallel_rank(0)

# for pp_rank in range(pp_size):
#     mpu.set_pipeline_model_parallel_rank(pp_rank)
#     for tp_rank in range(tp_size):
#         mpu.set_tensor_model_parallel_rank(tp_rank)
#         for ep_rank in range(ep_size):
#             mpu.set_expert_model_parallel_rank(ep_rank)
#             print(f" > pp rank: {pp_rank}, tp rank: {tp_rank}, ep rank: {ep_rank}")
#             state_dict, _, _ = _load_base_checkpoint(load_dir, rank0=False)
#             for name, val in state_dict["model"].items():
#                 if "_extra_state" in name:
#                     continue
#                 print("name:", name)
#                 print("shape:", val.shape)
#                 print("val:", torch.sum(torch.abs(val)), val)







# def check_mg_eg_forward(mgmodel, hgmodel, mgargs):
#     hg_hiddens = [{} for _ in range(mgargs.num_layers)]
#     mg_hiddens = [{} for _ in range(mgargs.num_layers)]

#     head_dim = mgargs.hidden_size // mgargs.num_attention_heads
#     hidden_size = mgargs.hidden_size
#     def print_input_hook(module, args, kwargs, layer_idx, mode):
#         frame, name = mode.split('-')
#         if frame == 'hg':
#             hg_hiddens[layer_idx][name] = args[0].transpose(0, 1)
#         elif frame == 'mg' and 'layer' in mode:
#             mg_hiddens[layer_idx][name] = kwargs.get('hidden_states')
#         elif frame == 'mg':
#             mg_hiddens[layer_idx][name] = args[0]
    
#     def print_output_hook(module, args, kwargs, output, layer_idx, mode):
#         frame, name = mode.split('-')
#         if mode in ['hg-q_proj_out', 'hg-k_proj_out', 'hg-v_proj_out']:
#             hg_hiddens[layer_idx][name] = output
#             hg_hiddens[layer_idx][name+'_weight'] = module.weight
#         elif mode in ['hg-lmhead']:
#             hg_hiddens[layer_idx][name] = output.transpose(0, 1)
#             hg_hiddens[layer_idx][name + '_token'] = output.transpose(0, 1).max(dim=-1)[1]
#             print(output.transpose(0, 1).max(dim=-1))
#         elif mode == 'hg-attn_out':
#             hg_hiddens[layer_idx][name] = output[0].transpose(0, 1)
#         elif mode in ['mg-lmhead']:
#             mg_hiddens[layer_idx][name] = output[0]
#             mg_hiddens[layer_idx][name + '_token'] = output[0].max(dim=-1)[1]
#             print(output[0].max(dim=-1))
#         elif mode ==  'mg-attn_out':
#             mg_hiddens[layer_idx][name] = output[0]
#         elif mode == 'mg-qkv':
#             mixed_qkv = output[0]
#             sq, b, _ = mixed_qkv.shape
#             mixed_qkv = mixed_qkv.view(sq, b, mgargs.num_query_groups, -1)
#             qh = mgargs.num_attention_heads // mgargs.num_query_groups
#             qo, ko, vo = torch.split(mixed_qkv, [qh * head_dim, head_dim, head_dim], dim=3)
#             qo = qo.reshape(b, -1, hidden_size)
#             ko = ko.reshape(b, -1, hidden_size // qh)
#             vo = vo.reshape(b, -1, hidden_size // qh)
#             mg_hiddens[layer_idx]['q_proj_out'] = qo
#             mg_hiddens[layer_idx]['k_proj_out'] = ko
#             mg_hiddens[layer_idx]['v_proj_out'] = vo

#             weight = module.weight.view(mgargs.num_query_groups, -1, head_dim, hidden_size)
#             qw, kw, vw= weight.split([qh, 1, 1], dim=1)
#             mg_hiddens[layer_idx]['q_proj_out_weight'] = qw.reshape(-1, hidden_size)
#             mg_hiddens[layer_idx]['k_proj_out_weight'] = kw.reshape(-1, hidden_size // qh)
#             mg_hiddens[layer_idx]['v_proj_out_weight'] = vw.reshape(-1, hidden_size // qh)

#     hgmodel.lm_head.register_forward_hook(partial(print_output_hook, layer_idx=31, mode='hg-lmhead'), with_kwargs=True)
#     mgmodel.output_layer.register_forward_hook(partial(print_output_hook, layer_idx=31, mode='mg-lmhead'), with_kwargs=True)

#     for idx, layer in enumerate(hgmodel.model.layers):
#         layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hg-layer_in'), with_kwargs=True)
#         layer.self_attn.o_proj.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='hg-o_proj_in'), with_kwargs=True)
#         layer.self_attn.q_proj.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hg-q_proj_out'), with_kwargs=True)
#         layer.self_attn.k_proj.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hg-k_proj_out'), with_kwargs=True)
#         layer.self_attn.v_proj.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hg-v_proj_out'), with_kwargs=True)
#         layer.self_attn.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='hg-attn_out'), with_kwargs=True)
    
#     for idx, layer in enumerate(mgmodel.decoder.layers):
#         layer.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='mg-layer_in'), with_kwargs=True)
#         layer.self_attention.linear_qkv.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='mg-qkv'), with_kwargs=True)
#         layer.self_attention.linear_proj.register_forward_pre_hook(partial(print_input_hook, layer_idx=idx, mode='mg-o_proj_in'), with_kwargs=True)
#         layer.self_attention.register_forward_hook(partial(print_output_hook, layer_idx=idx, mode='mg-attn_out'), with_kwargs=True)

#     input_ids = torch.tensor([[1,2,3]]).long().cuda()
#     attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(input_ids, -100, True, True, True)
    
#     with torch.inference_mode():
#         try:
#             hgmodel.cuda()
#             hgmodel(input_ids=input_ids)
#         except torch.cuda.OutOfMemoryError:
#             print('oom for huggingface model forward')
#         hgmodel.cpu()
#         del hgmodel   
 
#     with torch.inference_mode():
#         try:
#             mgmodel.cuda()
#             mgmodel(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
#         except torch.cuda.OutOfMemoryError:
#             print('oom for megatron model forward')
#         mgmodel.cpu()
#         del mgmodel

#     epsilon = 1e-5
#     for idx, (hgh, mgh) in enumerate(zip(hg_hiddens, mg_hiddens)):
#         if len(hgh) != len(mgh):
#             continue
#         for k, hgv in hgh.items():
#             mgv, hgv = mgh[k].cpu(), hgv.cpu()
#             same_num = (hgv != mgv).sum()
#             diff_num = ((hgv - mgv) > epsilon).sum()
#             diff_max = (hgv - mgv).abs().max()
#             print(f'layer:{idx}, {k}, diff: {same_num}, diff>{epsilon}:[{diff_num}/{hgv.numel()}] diff_max:{diff_max}')



# def check_tokenizer_is_same(hgtokenizer, mgtokenizer):
#     if transformers.__version__ <= '4.33.2':
#         print('please update transformers')
#         return
    
#     if mgtokenizer is None:
#         return
        
#     conversation = [
#         {"role": "user", "content": "what's your name"},
#         {"role": "bot", "content": "cold"},
#     ]
#     hgres = hgtokenizer.apply_chat_template(conversation)
#     mgres = mgtokenizer.apply_chat_template(conversation)
#     for x, y in zip(hgres, mgres):
#         assert x == y, 'tokenizer is different for huggingface and megatron'

