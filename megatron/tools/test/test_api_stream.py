from multiprocessing import Pool
import requests, json
def process(prompt):
    data = {
                "engine": '',
                "prompt": prompt,
                "temperature": 0.9,
                "num_return_sequences": 1,
                "max_new_tokens": 32,
                "top_p": 0.8,
                "echo_prompt": False,
                "top_k_per_token": 1,
                "stop_sequences": [],
                "seed": 0,
                "sft": True,
            }
    data = json.dumps(data, ensure_ascii=False)
    with requests.post(f'http://0.0.0.0:5060/stream_func',json=data, stream=True) as r:
        result = []
        for content in r.iter_content(chunk_size=None):
            result.append(content.decode("utf-8"))

        return "".join(result)


tasks = ["1.简单介绍智源研究院。"]
with Pool(1) as p, open("/tmp/result.txt", "w") as f:
    result = p.map_async(process, tasks)
    result_list = result.get()
print(result_list)
