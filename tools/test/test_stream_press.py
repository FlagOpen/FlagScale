from multiprocessing import Pool
import requests, json
import random

tasks_dic = [
  {
    "prompt": "什么是牛顿的第一定律？"
  },
  {
    "prompt": "怎样预防感冒？"
  },
  {
    "prompt": "如何进行有效的环境保护？"
  },
  {
    "prompt": "鲁迅的文学作品有哪些？"
  },
  {
    "prompt": "美国南北战争是哪一年开始的？"
  },
  {
    "prompt": "电商有哪些营销方式？"
  },
  {
    "prompt": "变速箱的种类有那些？"
  },
  {
    "prompt": "变速箱的种类有那些？变速箱的种类有那些？"
  },
  {
    "prompt": "防抱死制动系统的原来是什么？"
  },
  {
    "prompt": "什么是扭矩？"
  },
  {
    "prompt": "中国的传统节日有那些？"
  },
  {
    "prompt": "国庆节是哪一天？"
  },
  {
    "prompt": "人体细胞中有多少染色体？"
  },
  {
    "prompt":"有机化学与无机化学的区别？"
  },
  {
    "prompt":"世界四大宗教是什么？"
  },
  {
    "prompt":"《麦田里的守望者》的作者是谁？"
  },
  {
    "prompt":"期货投资的特点是什么？"
  },
  {
    "prompt":"什么是光合作用？"
  },
  {
    "prompt":"中国古代有哪些朝代？"
  },
  {
    "prompt":"什么是垄断资本主义？"
  },
  {
    "prompt": "什么是加速度？"
  },
  {
    "prompt": "怎样预防火灾？"
  },
  {
    "prompt": "如何进行有效的海洋环境保护？"
  },
  {
    "prompt": "余华的文学作品有哪些？"
  },
  {
    "prompt": "哥伦布发现新大陆是哪一年？"
  },
  {
    "prompt": "咖啡店怎么营销？"
  },
  {
    "prompt": "汽车的种类有那些？"
  },
  {
    "prompt": "汽车发动机的原理是什么？"
  },
  {
    "prompt": "驱动系统的原来是什么？"
  },
  {
    "prompt": "什么是力矩？"
  },
  {
    "prompt": "美国的传统节日有那些？"
  },
  {
    "prompt": "蒙古国庆节是哪一天？"
  },
  {
    "prompt": "人体中有多少块骨头？"
  },
  {
    "prompt":"红细胞与白细胞的区别？"
  },
  {
    "prompt":"基督教是什么？"
  },
  {
    "prompt":"《哈利波特》的作者是谁？"
  },
  {
    "prompt":"基金的特点是什么？"
  },
  {
    "prompt":"什么是自花授粉？"
  },
  {
    "prompt":"四大文明古国是哪些？"
  },
  {
    "prompt":"什么是对赌协议？"
  },
  {
    "prompt": "英国的传统节日有那些？"
  },
  {
    "prompt": "意大利国庆节是哪一天？"
  },
  {
    "prompt": "人体中有多少白细胞？"
  },
  {
    "prompt":"雄蕊与雌蕊的区别？"
  },
  {
    "prompt":"佛教是什么？"
  },
  {
    "prompt":"《平凡的世界》的作者是谁？"
  },
  {
    "prompt":"金融杠杆的特点是什么？"
  },
  {
    "prompt":"什么是扦插？"
  },
  {
    "prompt":"埃及在哪里？"
  },
  {
    "prompt":"什么是互联网带货直播？"
  }
]


def process(prompt):
    temperature = random.random()
    top_p = random.random()
    top_k_per_token = random.randint(0, 150)
    seed = random.randint(0, 429496729)
    max_new_tokens = random.randint(32, 512)
    print("temperature:", temperature)
    print("top_p:", top_p)
    print("top_k_per_token:", top_k_per_token)
    print("seed:", seed)
    print("max_new_tokens:", max_new_tokens)
    data = {
                "engine": '',
                "prompt": prompt,
                "temperature": temperature,
                "num_return_sequences": 1,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
                "echo_prompt": False,
                "top_k_per_token": top_k_per_token,
                "stop_sequences": [],
                "seed": seed,
                "sft": True,
            }
    data = json.dumps(data, ensure_ascii=False)
    with requests.post(f'http://0.0.0.0:5060/stream_func',json=data, stream=True) as r:
        result = []
        for content in r.iter_content(chunk_size=None):
            result.append(content.decode("utf-8"))

        return "".join(result)

tasks = []
for dic in tasks_dic:
    tasks.append([dic['prompt']])

while True:
    for i in range(len(tasks)):
        print(tasks[i])
        with Pool(1) as p, open("/tmp/result.txt", "w") as f:
            result = p.map_async(process, tasks[i])
            result_list = result.get()
        print(result_list)
