import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import requests
from bs4 import BeautifulSoup
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI()

session_cache = {}

DEFAULT_MODEL_NAME = "google/gemma-2-2b-jpn-it"
DEFAULT_MODEL_TYPE = "lc_hf"
SUPPORT_MODEL_TYPE = ["hf", "lc_hf"]


class QuestionRequest(BaseModel):
    url: Optional[str] = None
    question: str
    session_id: Optional[str] = None
    model_name: Optional[str] = None
    model_type: Optional[str] = None

    def additional_validate(self):
        if self.model_type and self.model_type not in SUPPORT_MODEL_TYPE:
            return "unsupported model type"
        if self.url and self.session_id:
            return "url unnecessary"
        if self.session_id and (self.model_type or self.model_name):
            return "can set model name or type, when only first request"
        return None


class AnswerResponse(BaseModel):
    answer: str
    session_id: str
    referenced_page: str
    history: List[str]


# ドキュメントのページをすべて取得する関数
def fetch_all_pages(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400, detail="Invalid URL or unable to fetch document"
        )

    # BeautifulSoupでHTMLを解析
    soup = BeautifulSoup(response.text, "html.parser")

    # 全てのリンクを抽出して同じドメイン内のページを取得
    base_url = url.rstrip("/")
    pages = [base_url]
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("/") or href.startswith(base_url):
            full_url = base_url + href if href.startswith("/") else href
            pages.append(full_url)

    # 重複を排除してリストを返す
    return list(set(pages))


def load_llm(model_name: str, type="hf"):
    if type == "hf":
        from langchain_huggingface import HuggingFacePipeline

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, trust_remote_code=True
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        return tokenizer, llm
    elif type == "lc_hf":
        from langchain_huggingface import HuggingFaceEndpoint

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            task="text-generation",
            max_new_tokens=512,
            do_sample=True,
        )
        return tokenizer, llm
    elif type == "gguf":
        raise ValueError("gguf unsupport")
    else:
        raise ValueError("Unsupported model")


def get_prompt(tokenizer):
    question_prompt_template_format = tokenizer.apply_chat_template(
        conversation=[
            {
                "role": "user",
                "content": "Context: {context}\nHistory: {history}\nQuestion: {question}\nAnswer:",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt = PromptTemplate(
        template=question_prompt_template_format,
        input_variables=["context", "history", "question"],
    )
    return prompt


def generate_answer(
    llm,
    prompt_template,
    question: str,
    document_content: str,
    history: List[str],
):
    try:
        chain = LLMChain(llm=llm, prompt=prompt_template)
        answer = chain.invoke(
            {
                "context": document_content,
                "history": "\n".join(history),
                "question": question,
            }
        )
    except Exception as e:
        return "", e
    return answer, None


def generate_session_id():
    return uuid.uuid4()


@app.post("/ask_question", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    validation_err = request.additional_validate()
    if validation_err:
        raise HTTPException(status_code=400, detail=validation_err)

    # 初回request
    if request.url and not request.session_id:
        pages = fetch_all_pages(request.url)

        document_content = ""
        for page in pages:
            try:
                page_content = requests.get(page).text
                document_content += page_content
            except Exception as e:
                continue  # ページ取得に失敗した場合スキップ

        if not document_content:
            raise HTTPException(
                status_code=404, detail="No valid content found in document"
            )
        session_id = generate_session_id()
        model_name = request.model_name or DEFAULT_MODEL_NAME

        session_cache[session_id] = {
            "document_content": document_content,
            "question_history": [],  # 質問履歴を保存
            "model_name": model_name,
        }

    elif not request.url and request.session_id:  # 2回目以降のrequest
        session_id = request.session_id
        if not session_id or session_id not in session_cache:
            raise HTTPException(status_code=400, detail="Invalid or missing session ID")

        document_content = session_cache[session_id]["document_content"]
    else:  ## ここには到達しないはず
        raise HTTPException(status_code=400, detail="bad request")

    session_cache[session_id]["question_history"].append(request.question)

    model_name = session_cache[session_id]["model_name"]

    model_type = request.model_type or DEFAULT_MODEL_TYPE
    tokenizer, llm_pipeline = load_llm(model_name, model_type)
    prompt_template = get_prompt(tokenizer)

    answer, err = generate_answer(
        llm_pipeline,
        prompt_template,
        request.question,
        document_content,
        session_cache[session_id]["question_history"],
    )
    if err:
        print("generate_answer error: ", err)
        raise HTTPException(status_code=500)

    # 7. レスポンスの作成
    return AnswerResponse(
        answer=answer,
        session_id=session_id,
        referenced_page=(request.url or session_cache[session_id]["referenced_page"]),
        history=session_cache[session_id]["question_history"],
    )
