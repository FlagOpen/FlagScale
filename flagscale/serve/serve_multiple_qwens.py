from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import ray
from flagscale import serve


@serve.remote(name="model1")
class ModelWorker1:
    def __init__(self, args):
        model_config = args["serve"]["model_args"]["model1"]
        self.llm = LLM(**model_config)
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95)

    def generate(self, prompt):
        outputs = self.llm.generate([prompt], sampling_params=self.sampling_params)
        return [output.text for output in outputs]


@serve.remote(name="model2")
class ModelWorker2:
    def __init__(self, args):
        model_config = args["serve"]["model_args"]["model2"]
        self.llm = LLM(**model_config)
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95)

    def generate(self, prompt):
        outputs = self.llm.generate([prompt], sampling_params=self.sampling_params)
        return [output.text for output in outputs]


ray.init()
model_worker1 = ModelWorker1.remote(serve.task_config)
model_worker2 = ModelWorker2.remote(serve.task_config)


app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str


@app.post("/generate")
async def generate(request: GenerateRequest):
    prompt = request.prompt
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    outputs1 = await model_worker1.generate.remote(prompt)
    outputs2 = await model_worker2.generate.remote(prompt)

    return {"outputs": outputs2}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9010)
