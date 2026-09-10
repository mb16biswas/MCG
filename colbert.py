import os
import re
import json
import time
import argparse
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from huggingface_hub import login

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import pandas as pd
import string
from langchain.docstore.document import Document
import argparse
import re
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig, pipeline
from transformers import AutoModel,AutoModelForMaskedLM
from typing import List
import torch
import torch.nn.functional as F
from FlagEmbedding import FlagReranker
from torch import Tensor
import os
from transformers import set_seed
from huggingface_hub import login
from ragatouille import RAGPretrainedModel
import time
import argparse


# -----------------------------
# Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Run sparse (BM25) retrieval + QA with GPT-4 API or HuggingFace models")
parser.add_argument('--mode', type=str, choices=['gpt4', 'hf'], default='hf',
                    help="Choose 'gpt4' for API or 'hf' for local/endpoint HuggingFace model")
parser.add_argument('--f1', type=int, default=2,
                    help="Model choice: 1=Saul-7B, 2=Mistral-7B-v0.2, 3=Mistral-7B-v0.3, 4=Llama-3-8B, 5=Qwen2.5-7B, else=Llama-2-7B")
parser.add_argument('--f2', type=int, default=2,
                    help="HF variant: 1=HuggingFaceEndpoint, 2=local pipeline")
parser.add_argument('--token', type=str, default="",
                    help="HuggingFace API token (required for HF endpoint or gated models)")
parser.add_argument('--k', type=int, default=5, help="Top-k ")
parser.add_argument('--chunk_size', type=int, default=1000, help="Chunk size for text splitting")
parser.add_argument('--chunk_overlap', type=int, default=200, help="Chunk overlap for text splitting")
args, unknown_args = parser.parse_known_args()

if unknown_args:
    print(f"Unrecognized arguments: {unknown_args}")

# -----------------------------
# Configurations
# -----------------------------
API_URL = "http://XXX/openai_service"

set_seed(42)

pdf_source = "/workspace/data/Contract-QA2/cuad-pdf/"
saved_path = "/workspace/data/Graph-KD/results/colbert-final/"

print()
print("*" * 100)
print()
print("colbert.py")
print()
print(f"Mode: {args.mode}")
print(f"f1: {args.f1}")
print(f"f2: {args.f2}")
print(f"Top-k: {args.k}")
print(f"Chunk size / overlap: {args.chunk_size} / {args.chunk_overlap}")
print()
print("*" * 100)
print()


# -----------------------------
# Utilities
# -----------------------------
def create_folder(folder_name: str) -> str:
    folder_path = os.path.join(saved_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Folder ready: {folder_path}")
    return folder_path


def simple_tokenize(text: str):
    """
    Lightweight regex tokenizer: splits on non-word chars, keeps alphanumerics.
    Lowercases to normalize.
    """
    return re.findall(r"\w+", text.lower())


def create_prompt(question, contexts):
    contexts = [f"#Context {i + 1}: {contexts[i]}" for i in range(len(contexts))]
    context_string = "\n".join(contexts)
    return (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context "
        "to answer the question. If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n"
        f"#Question: {question}\n"
        f"{context_string}\n"
        "#Answer:"
    )


# -----------------------------
# Load Model or API
# -----------------------------
if args.mode == 'gpt4':
    f_name = "cuad-vanila-gpt4-bm25"
    output_folder = create_folder(f_name)

    # GPT-4 via API
    def run_inference(prompt: str) -> str:
        payload = {
            "chat_message": [{"role": "system", "content": prompt}],
            "max_tokens": 256,
            "model": "gpt-4",
            "temperature": 0.1,
            "presence_penalty": 1
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            return f"Error: {response.status_code}"
else:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = args.token
    login(token=args.token)

    # Model selection
    if args.f1 == 1:
        repo_id = "Equall/Saul-7B-Instruct-v1"
        f_name = "cuad-colbert-saul-colbert"
        s_index, e_index = 53, 150
    elif args.f1 == 2:
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
        f_name = "cuad-colbert-mis-2-colbert"
        s_index, e_index = 0, 150
    elif args.f1 == 3:
        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        f_name = "cuad-colbert-mis-3-colbert"
        s_index, e_index = 0, 150
    elif args.f1 == 4:
        repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        f_name = "cuad-colbert-llama-3-colbert"
        s_index, e_index = 53, 150
    elif args.f1 == 5:
        repo_id = "Qwen/Qwen2.5-7B-Instruct"
        f_name = "cuad-colbert-qwen-colbert"
        s_index, e_index = 0, 150
    else:
        repo_id = "daryl149/llama-2-7b-chat-hf"
        f_name = "cuad-colbert-llama-2-colbert"
        s_index, e_index = 0, 150

    output_folder = create_folder(f_name)

    print()
    print("*" * 100)
    print()
    print(f"repo_id: {repo_id}")
    print()
    print("*" * 100)
    print()

    if args.f2 == 1:
        # HuggingFace Inference Endpoint
        llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=2048, token=args.token, task="text-generation")
        print()
        print(llm)
        print()

        def run_inference(prompt: str) -> str:
            return llm.invoke(prompt)
    else:
        # Local pipeline
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_pretrained(repo_id, device_map="auto", torch_dtype="auto")

        print()
        print(model)
        print()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        gen_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,
            return_full_text=False
        )
        llm = HuggingFacePipeline(pipeline=gen_pipe)

        print()
        print(llm)
        print()

        def run_inference(prompt: str) -> str:
            return llm.invoke(prompt)



RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
# -----------------------------
# Data / Task Configuration
# -----------------------------
categories = [
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Expiration Date-Answer",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Non-Compete",
    "Exclusivity",
    "License Grant",
    "Audit Rights",
    "Uncapped Liability",
    "Cap on Liability",
    "Insurance",
]

questions = [
    "Who are the parties involved in the agreement?",
    "What is the agreement date mentioned in the contract?",
    "What is the effective date of the agreement?",
    "What is the expiration date of the agreement?",
    "What is the exact expiration date provided in the contract?",
    "What is the renewal term specified in the agreement?",
    "What is the notice period required to terminate the renewal?",
    "What is the governing law stated in the agreement?",
    "What non-compete clause is included in the agreement?",
    "What exclusivity rights are granted in the agreement?",
    "What rights are granted under the license in the agreement?",
    "What audit rights are provided in the agreement?",
    "What uncapped liability provisions are mentioned in the agreement?",
    "What is the cap on liability mentioned in the agreement?",
    "What insurance requirements are specified in the agreement?",
]


# -----------------------------
# Main Loop
# -----------------------------
# If using GPT-4 mode, s_index/e_index were not defined above. Set defaults.
if args.mode == 'gpt4':
    s_index, e_index = 0, len(os.listdir(pdf_source))

for index_, name in enumerate(sorted(os.listdir(pdf_source))[s_index:e_index]):
    try:
        top_k_contexts, all_prompts, answers, time_inf = [], [], [], []

        print()
        print("*" * 100)
        print()
        print(f"Processing {index_}: {name}")
        print()
        print("*" * 100)
        print()

        # --------- Load and Split ---------
        t0 = time.time()

        loader = PyPDFLoader(os.path.join(pdf_source, name))
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            add_start_index=True
        )
        splits = text_splitter.split_documents(docs)

        full_document = ""

        for page in splits:

            full_document += page.page_content

        RAG.index(
            collection=[full_document],
            index_name= f"{name}-{index_}",
            max_document_length=512,
            split_documents=True,
            overwrite_index="force_silent_overwrite"
        )

        for q in questions:

            results = RAG.search(query=q, k=args.k)
            context = [r["content"] for r in results]
            top_k_contexts.append(context)


        # # --------- Retrieve per Question ---------
        # for q in questions:
        #     q_tokens = simple_tokenize(q)
        #     scores = bm25.get_scores(q_tokens)  # ndarray over corpus
        #     # Top-k indices by descending score
        #     top_idx = np.argsort(-scores)[:args.k].tolist()
        #     retrieved_contexts = [split_texts[i] for i in top_idx]
        #     top_k_contexts.append(retrieved_contexts)

        # --------- Generate Answers ---------
        for i, q in enumerate(questions):
            t_start = time.time()
            prompt = create_prompt(q, top_k_contexts[i])
            ans = run_inference(prompt)
            # Some HF endpoints echo prompt; if that happens, strip it
            if isinstance(ans, str):
                ans = ans.replace(prompt, "")
            elif isinstance(ans, list) and len(ans) > 0 and isinstance(ans[0], str):
                ans = ans[0].replace(prompt, "")

            t_end = time.time()

            print()
            print("*" * 100)
            print()
            print(f"Answer Time (sec): {t_end - t_start:.4f}")
            print()
            print("*" * 100)
            print()

            all_prompts.append(prompt)
            answers.append(ans)
            time_inf.append(t_end - t_start)

        # --------- Save Results ---------
        result_df = pd.DataFrame({
            "categories": categories,
            "question": questions,
            "top_k": top_k_contexts,
            "prompt": all_prompts,
            "answers": answers,
            "time_inf": time_inf
        })
        out_path = os.path.join(output_folder, f"{os.path.splitext(name)[0]}.csv")
        result_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

    except Exception as e:
        print(f"Error processing {name}: {e}")
        continue
