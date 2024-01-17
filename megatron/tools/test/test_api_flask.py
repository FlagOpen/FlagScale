# -*- coding: utf-8 -*-
import json
import requests

prompt = "请介绍一下智源研究院"

raw_request = {
            "prompt": prompt,
            "temperature": 0.9,
            "num_return_sequences": 1,
            "max_new_tokens": 512,
            "top_p": 0.95,
            "top_k_per_token": 0,
            "stop_sequences": [],
            "seed": 123,
            "sft": True,
        }

# Please specify the following url of your server
url = "xxxx"

data_json = json.dumps(raw_request)
response = requests.post(url, json=data_json)
print(response)
result = response.json()
print(result)
