from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_core.messages.ai import AIMessage
from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uuid
import time
from typing import List

# Initialize the FastAPI app
app = FastAPI()

model_name = 'Bllossom/llama-3-Korean-Bllossom-70B-gguf-Q4_K_M'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Llama(
    model_path = '/workspace/llama3_4Q_model/llama-3-Korean-Bllossom-70B-gguf-Q4_K_M.gguf',
    n_ctx=8190,
    n_gpu_layers=-1
)

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.0

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class QueryResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: dict


def creat_AImessage(response_text, finish_reason):   
    ai_message = AIMessage(
        content= response_text,
        response_metadata={
            "model_name": model_name,
            "system_fingerprint": None,
            "finish_reason": finish_reason
        },
        id= f"chatcmpl-{str(uuid.uuid4())}"
    )
    
    return ai_message


@app.post("/v1/completions/chat/completions")
async def completions(request: QueryRequest):
    messages = request.messages
    inputs = " ".join([msg.content for msg in messages])
    
    PROMPT = \
'''당신은 Langchain에 사용되며, 요청에 알맞는 답변을 하거나, 사용자의 질문에 적절한 Tool과 그에 들어갈 인수를 System Prompt에서 제공한 형식에 맞게 출력합니다.
그리고, 사용할 Tool의 이름과 인수가 제시될 경우, 제공된 Tool과 인수를 통해 사용자의 질문에 답변하기 위한 정보를 생성하고,
최종적으로 답변이 생성되면 해당 내용을 마지막으로 출력 (Final Answera)합니다.
적절한 정답을 찾을때까지 Tool을 이용해 필요한 작업을 연속적으로 수행합니다.
'''
    instruction=messages
    messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {"role": "user", "content": f"{instruction}"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize = False,
        add_generation_prompt=True
    )
    
    generation_kwargs = {
        "max_tokens":512,
        "stop":["<|eot_id|>"],
        "echo":True, # Echo the prompt in the output
        "top_p":0.9,
        "temperature":0.6,
    }
    response = model(prompt, **generation_kwargs)
    response_ = response['choices'][0]['text'] if 'choices' in response and len(response['choices']) > 0 else ""
    
    response_text = response_.split('assistant<|end_header_id|>\n\n')[1]

    if isinstance(response_text, str):
        if 'Final Answer' in response_text:
            finish_reason = 'stop'
        else :
            finish_reason = 'tool_calls' #ref.https://platform.openai.com/docs/api-reference/chat/object
    else :
        finish_reason = 'tool_calls'
    response = creat_AImessage(response_text, finish_reason)
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
