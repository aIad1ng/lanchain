import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import (
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


# 获取环境变量中的API_KEY
API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 创建一个 FastAPI 应用
app = FastAPI()
ASGI_APPLICATION = 'test:app'

# 创建一个模板对象，指定模板文件的路径
templates = Jinja2Templates(directory="templates")

# 创建一个缓存字典，用来存储已经生成过的回答
cache = {}

# 定义一个 RAG 生成函数，接受一个用户输入的问题，返回一个生成器对象
def rag_generator(question):
    # 创建一个系统消息，表示你是一个 RAG 应用
    system_message = SystemMessage(content="You are a RAG application")
    # 创建一个人类消息，表示用户输入的问题
    human_message = HumanMessage(content=question)
    # 将两个消息组合成一个列表
    messages = [system_message, human_message]
    # 检查缓存字典中是否有对应的回答，如果有则直接返回
    key = tuple(message.content for message in messages) # 将消息列表转化为一个元组，作为缓存的键
    if key in cache:
        yield cache[key]
    else:
        # 如果没有缓存，则调用 ChatOpenAI 对象，传入消息列表，得到一个生成器对象
        chat = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, verbose=True, max_tokens=256,callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
        generator = chat.invoke(messages)
        # 遍历生成器对象，得到每一个回答，将其缓存起来，并 yield 出来
        for answer in generator:
            cache[key] = answer
            yield answer

# 定义一个路由函数，接受一个 query 参数，返回一个 HTML 响应对象
@app.get("/rag", response_class=HTMLResponse) # 指定响应类为 HTMLResponse
async def rag(query: str):
    # 调用 RAG 生成函数，传入 query 参数，得到一个生成器对象
    generator = rag_generator(query)
    # 使用 list 函数将生成器对象转换为一个列表，再拼接成一个字符串
    answer = "".join(list(generator))
    # 将问题和答案一起传递给模板，渲染为 HTML 内容
    return templates.TemplateResponse("index.html", {"request": query, "response": answer})



