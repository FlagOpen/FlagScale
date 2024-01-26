# LLaMA2 MODEL

## Table of contents
- [1. Training Setup](#1-training-setup)
- [2. Configurations](#2-configurations)

## 1. Training setup
<a id="markdown-training-setup" name="training-setup"></a>

To run the model using a docker container run it as follows
```
PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:23.09-py3
CHECKPOINT_PATH="" #<Specify path>
TENSORBOARD_LOGS_PATH=""#<Specify path>
TOKENIZER_PATH="" #<Specify path to file>/tokenizer.model
DATA_PATH="" #<Specify path and file prefix>_text_document

docker run \
  --gpus=all \
  --ipc=host \
  --workdir /workspace/megatron-lm \
  -v /path/to/data:/path/to/data \
  -v /path/to/megatron-lm:/workspace/megatron-lm \
  megatron-lm nvcr.io/nvidia/pytorch:23.09-py3 \
  bash /examples/llama2/train_llama2_70b_distributed.sh $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH $TOKENIZER_PATH $DATA_PATH "

```
NOTE: Depending on the environment you are running it the above command might like slightly different.


## 2. Configurations
<a id="markdown-configurations" name="configurations"></a>
The example in this folder shows you how to run 70B model. There are other configs you could run as well

## 3.DataSet and Tokenizer
Dataset:
       https://bd.bcebos.com/v1/klx-pytorch-work-bd/training/wurui04/llama_00_text_document.bin
       https://bd.bcebos.com/v1/klx-pytorch-work-bd/training/wurui04/llama_00_text_document.idx

LLAMA 2 COMMUNITY LICENSE AGREEMENT	
Llama 2 Version Release Date: July 18, 2023

"Agreement" means the terms and conditions for use, reproduction, distribution and 
modification of the Llama Materials set forth herein.

"Documentation" means the specifications, manuals and documentation 
accompanying Llama 2 distributed by Meta at ai.meta.com/resources/models-and-
libraries/llama-downloads/.

"Licensee" or "you" means you, or your employer or any other person or entity (if 
you are entering into this Agreement on such person or entity's behalf), of the age 
required under applicable laws, rules or regulations to provide legal consent and that 
has legal authority to bind your employer or such other person or entity if you are 
entering in this Agreement on their behalf.

"Llama 2" means the foundational large language models and software and 
algorithms, including machine-learning model code, trained model weights, 
inference-enabling code, training-enabling code, fine-tuning enabling code and other 
elements of the foregoing distributed by Meta at ai.meta.com/resources/models-and-
libraries/llama-downloads/.

"Llama Materials" means, collectively, Meta's proprietary Llama 2 and 
Documentation (and any portion thereof) made available under this Agreement.

"Meta" or "we" means Meta Platforms Ireland Limited (if you are located in or, if you 
are an entity, your principal place of business is in the EEA or Switzerland) and Meta 
Platforms, Inc. (if you are located outside of the EEA or Switzerland). 

By clicking "I Accept" below or by using or distributing any portion or element of the 
Llama Materials, you agree to be bound by this Agreement.

1. License Rights and Redistribution. 

      a. Grant of Rights. You are granted a non-exclusive, worldwide, non-
transferable and royalty-free limited license under Meta's intellectual property or 
other rights owned by Meta embodied in the Llama Materials to use, reproduce, 
distribute, copy, create derivative works of, and make modifications to the Llama 
Materials.  
      
      b. Redistribution and Use.  

            i. If you distribute or make the Llama Materials, or any derivative works 
thereof, available to a third party, you shall provide a copy of this Agreement to such 
third party. 
            ii.  If you receive Llama Materials, or any derivative works thereof, from 
a Licensee as part of an integrated end user product, then Section 2 of this 
Agreement will not apply to you. 

            iii. You must retain in all copies of the Llama Materials that you 
distribute the following attribution notice within a "Notice" text file distributed as a 
part of such copies: "Llama 2 is licensed under the LLAMA 2 Community License, 
Copyright (c) Meta Platforms, Inc. All Rights Reserved."

            iv. Your use of the Llama Materials must comply with applicable laws 
and regulations (including trade compliance laws and regulations) and adhere to the 
Acceptable Use Policy for the Llama Materials (available at 
https://ai.meta.com/llama/use-policy), which is hereby incorporated by reference into 
this Agreement.

            v. You will not use the Llama Materials or any output or results of the 
Llama Materials to improve any other large language model (excluding Llama 2 or 
derivative works thereof).  

2. Additional Commercial Terms. If, on the Llama 2 version release date, the 
monthly active users of the products or services made available by or for Licensee, 
or Licensee's affiliates, is greater than 700 million monthly active users in the 
preceding calendar month, you must request a license from Meta, which Meta may 
grant to you in its sole discretion, and you are not authorized to exercise any of the 
rights under this Agreement unless or until Meta otherwise expressly grants you 
such rights.
            
3. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW, THE 
LLAMA MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM ARE 
PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, 
EITHER EXPRESS OR IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY 
WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR 
FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE 
FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING 
THE LLAMA MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR 
USE OF THE LLAMA MATERIALS AND ANY OUTPUT AND RESULTS.

4. Limitation of Liability. IN NO EVENT WILL META OR ITS AFFILIATES BE 
LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, 
NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS 
AGREEMENT, FOR ANY LOST PROFITS OR ANY INDIRECT, SPECIAL, 
CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN 
IF META OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF 
ANY OF THE FOREGOING.
 
5. Intellectual Property.

      a. No trademark licenses are granted under this Agreement, and in 
connection with the Llama Materials, neither Meta nor Licensee may use any name 
or mark owned by or associated with the other or any of its affiliates, except as 
required for reasonable and customary use in describing and redistributing the 
Llama Materials.

      b. Subject to Meta's ownership of Llama Materials and derivatives made by or 
for Meta, with respect to any derivative works and modifications of the Llama 
Materials that are made by you, as between you and Meta, you are and will be the 
owner of such derivative works and modifications.

      c. If you institute litigation or other proceedings against Meta or any entity 
(including a cross-claim or counterclaim in a lawsuit) alleging that the Llama 
Materials or Llama 2 outputs or results, or any portion of any of the foregoing, 
constitutes an infringement of intellectual property or other rights owned or licensable 
by you, then any licenses granted to you under this Agreement shall terminate as of 
the date such litigation or claim is filed or instituted. You will indemnify and hold 
harmless Meta from and against any claim by any third party arising out of or related 
to your use or distribution of the Llama Materials.

6. Term and Termination. The term of this Agreement will commence upon your 
acceptance of this Agreement or access to the Llama Materials and will continue in 
full force and effect until terminated in accordance with the terms and conditions 
herein. Meta may terminate this Agreement if you are in breach of any term or 
condition of this Agreement. Upon termination of this Agreement, you shall delete 
and cease use of the Llama Materials. Sections 3, 4 and 7 shall survive the 
termination of this Agreement. 

7. Governing Law and Jurisdiction. This Agreement will be governed and 
construed under the laws of the State of California without regard to choice of law 
principles, and the UN Convention on Contracts for the International Sale of Goods 
does not apply to this Agreement. The courts of California shall have exclusive 
jurisdiction of any dispute arising out of this Agreement. 

Tokenizer:
       https://bd.bcebos.com/v1/klx-pytorch-work-bd/training/wurui04/llama2/tokenizer.model

### 7B 
```
       --num-layers 32 \
       --hidden-size 4096 \
       --num-attention-heads 32 \
       --seq-length 4096 \
       --ffn-hidden-size 11008 \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
```

### 13B 
```
       --num-layers 40 \
       --hidden-size 5120 \
       --num-attention-heads 40 \
       --seq-length 4096 \
       --ffn-hidden-size 13824 \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 1 \
```
