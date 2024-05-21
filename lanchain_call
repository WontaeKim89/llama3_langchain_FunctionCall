import torch
import os
from langchain_core.tools import tool
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_core.callbacks import CallbackManager, BaseCallbackHandler
import logging
from langchain import hub
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import create_json_chat_agent
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import create_json_chat_agent
from operator import itemgetter
import langsmith
from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import render_text_description_and_args
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.pydantic_v1 import BaseModel, Field
from typing import Tuple, List


local_llm = ChatOpenAI(
    base_url="http://localhost:8002/v1/completions",
    api_key="llama3_api",
    model="Bllossom/llama-3-Korean-Bllossom-70B",
)

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int
@tool
def korea_city_weather_info(city_name: str) -> str:
    """
    Tool Description : The weather information of the city inputted by the user is outputted.
    Args : 
        "city_name" : Name of the city to look up the weather
    """
    return "오늘 서울의 온도는 최고 24.2도, 최저 13.5도를 보이며, 낮 한때 소나기가 있겠습니다."

@tool
def search_phonenumber(query: str) -> str:
    """장소에 대한 전화번호 검색 결과를 반환할 때 사용되는 도구입니다. 전화번호를 알고싶을때 사용하는 도구입니다."""
    return "판교 몽중헌 전화번호: 010-1234-5678\n\n서울 OOO 전화번호: 02-123-4567"


tools = [search_phonenumber]

chat_model_with_stop = local_llm.bind(
    stop=["Observation", "\nObservation", "\n관측"])

json_prompt = hub.pull("hwchase17/react-chat-json")
llama3_agent = create_json_chat_agent(local_llm, tools, json_prompt)

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# 로깅 콜백 추가
class LoggingCallbackHandler(BaseCallbackHandler):
    def on_agent_start(self, agent, **kwargs):
        logger.debug(f"Agent started: {agent}")
    
    def on_agent_step(self, agent, step, **kwargs):
        logger.debug(f"Agent step: {step}")
    
    def on_agent_finish(self, agent, **kwargs):
        logger.debug(f"Agent finished: {agent}")
    
    def on_tool_start(self, tool, **kwargs):
        logger.debug(f"Tool started: {tool}")
    
    def on_tool_finish(self, tool, result, **kwargs):
        logger.debug(f"Tool finished: {tool} with result: {result}")
# 로깅 콜백 핸들러 인스턴스 생성
logging_callback_handler = LoggingCallbackHandler()

# 로깅 콜백 매니저 인스턴스 생성
logging_callback_manager = CallbackManager(handlers=[logging_callback_handler])

llama3_agent_executor = AgentExecutor(
    agent=llama3_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    early_stopping_method='force',
    callback_manager=logging_callback_manager
)


response = llama3_agent_executor.invoke(
    {"input": "판교 몽중헌 전화번호를 검색하여 결과를 알려주세요."}
)
print(f'답변: {response["output"]}')
