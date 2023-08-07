# To Run:
# python -m dsp.modules.hf_server --port 4242 --model "google/flan-t5-base"

# To Query:
# curl -d '{"prompt":"What is the answer to life, the universe, and everything?"}' \
# -X POST "http://0.0.0.0:4242" \
# -H 'Content-Type: application/json' 


from functools import lru_cache
import argparse
import time
import random
import os
import sys
import uvicorn
import warnings

from fastapi import FastAPI
from pydantic import BaseModel
from argparse import ArgumentParser
from starlette.middleware.cors import CORSMiddleware

from dsp.modules.hf import HFModel


class Query(BaseModel):
    prompt: str


warnings.filterwarnings("ignore")

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

parser = argparse.ArgumentParser("Server for Hugging Face models")
parser.add_argument("--port", type=int, required=True, help="Server port")
parser.add_argument("--model", type=str, required=True, help="Hugging Face model")
args = parser.parse_args()
# TODO: Convert this to a log message
print(f"#> Loading the language model {args.model}")
lm = HFModel(args.model)


@lru_cache(maxsize=None)
def generate(prompt, **kwargs):
    global lm
    generateStart = time.time()
    # TODO: Convert this to a log message
    print(f'#> Prompt: "{prompt}"')
    response = lm._generate(prompt, **kwargs)
    # TODO: Convert this to a log message
    print(f'#> Response: "{response}"')
    latency = (time.time() - generateStart) * 1000.0
    response["latency"] = latency
    return response


@app.post("/")
async def generate_post(query: Query):
    # TODO: Handle kwargs

    return generate(query.prompt)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        reload=False,
        log_level="info",
    )  # can make reload=True later
