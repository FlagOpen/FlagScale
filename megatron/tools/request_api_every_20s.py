# -*- coding: utf-8 -*-
import json
import requests

prompt = "你好啊,"

raw_request = {
            "prompt": prompt,
            "temperature": 0.9,
            "num_return_sequences": 1,
            "max_new_tokens": 2,
            "top_p": 0.95,
            "top_k_per_token": 0,
            "stop_sequences": [],
            "seed": 123,
        }

url = 'http://127.0.0.1:5060/batch_func'

import time 

while True:
    data_json = json.dumps(raw_request)
    try:
        response = requests.post(url, json=data_json)
        print(response)
        result = response.json()
        print(result)
    except:
        pass 

    time.sleep(20)
