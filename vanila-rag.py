import os
import json
import argparse
import requests
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEndpoint
from tqdm import tqdm 
import time

# -----------------------------
# Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Run inference using GPT-4 API or HuggingFace model")
parser.add_argument('--mode', type=str, choices=['gpt4', 'hf'], default='hf', help="Choose 'gpt4' or 'hf' for HuggingFace model")
parser.add_argument('--f1', type=int, default=2, help="Model choice: 1=Saul-7B, 2=Mistral-2-7B, 3=Mistral-3-7B, 4=Meta-Llama-3-8B, 5=llama-2-7b")
parser.add_argument('--f2', type=int, default=2, help="Variant for HF Model")
parser.add_argument('--token', type=str, default= "", help="HuggingFace API token")
args, unknown_args = parser.parse_known_args()

if unknown_args:
    print(f"Unrecognized arguments: {unknown_args}")

# -----------------------------
# Configurations
# -----------------------------
API_URL = "http://XXX/openai_service"

set_seed(42)

pdf_source = "/workspace/data/Contract-QA2/cuad-pdf/"
saved_path = "/workspace/data/Graph-KD/results/vanila-v2-final/"


print()
print("*"*100)
print()
print()
print()
print()
print("vanila-rag.py")
print()
print()
print()
print(f"Mode: {args.mode}")
print(f"f1: {args.f1}")
print(f"f2: {args.f2}")
print()
print("*"*100)
print()


# -----------------------------
# Create Output Folder
# -----------------------------
def create_folder(folder_name):
    folder_path = f"{saved_path}{folder_name}"
    os.makedirs(folder_path, exist_ok=True)
    print(f"Folder ready: {folder_path}")
    return folder_path



# -----------------------------
# Load Model or API
# -----------------------------
if args.mode == 'gpt4':

    f_name = f"cuad-vanila-gpt4"

    output_folder = create_folder(f_name)
    # GPT-4 via API
    def run_inference(prompt):
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

    if args.f1 == 1:
        repo_id = "Equall/Saul-7B-Instruct-v1"
        f_name = f"cuad-vanila-saul"
        s_index = 80
        e_index = 200

    elif args.f1 == 2:
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
        f_name = f"cuad-vanila-mis-2"
        s_index = 0
        e_index = 200

    elif args.f1 == 3:
        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        f_name = f"cuad-vanila-mis-3"
        s_index = 0
        e_index = 200

    elif args.f1 == 4:
        repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        f_name = f"cuad-vanila-llama-3"
        s_index = 0
        e_index = 100

    else:
        repo_id = "daryl149/llama-2-7b-chat-hf"
        f_name = f"cuad-vanila-llama-2"
        s_index = 0
        e_index = 100
    
    output_folder = create_folder(f_name)

    print()
    print("*"*100)
    print()
    print(f"repo_id: {repo_id}")
    print()
    print("*"*100)
    print()
    # HuggingFace model
    if args.f2 == 1:


        # Endpoint mode
        
        llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=2048, token=args.token, task="text-generation")

        print()
        print(llm)
        print()

        def run_inference(prompt):
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
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048, return_full_text=False)
        llm = HuggingFacePipeline(pipeline=pipe)

        print()
        print(llm)
        print()

        def run_inference(prompt):
            return llm.invoke(prompt)

# -----------------------------
# Load Data
# -----------------------------
# df = pd.read_excel("/workspace/data/Contract-QA2/final-labelled-NAACL-Revised.xlsx")
# contract_names = list(df["Filename"])


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
    "Cap on Liability" , 
    "Insurance" , 
]


# questions = [
#     "Who are the parties involved in the agreement?",
#     "What is the agreement date mentioned in the contract?",
#     "What is the effective date of the agreement?",
#     "What is the expiration date of the agreement?",
#     "What is the exact expiration date provided in the contract?",
#     "What is the renewal term specified in the agreement?",
#     "What is the notice period required to terminate the renewal?",
#     "What is the governing law stated in the agreement?",
#     "Does the agreement include a non-compete clause?",
#     "Does the agreement grant exclusivity to any party?",
#     "What rights are granted under the license in the agreement?",
#     "Does the agreement provide audit rights?",
#     "Does the agreement mention uncapped liability?",
#     "What is the cap on liability mentioned in the agreement?",
#     "What insurance requirements are specified in the agreement?"
# ]


# questions = [
#     "Who are the parties involved in the agreement?",
#     "What is the agreement date mentioned in the contract?",
#     "What is the effective date of the agreement?",
#     "What is the expiration date of the agreement?",
#     "What is the exact expiration date provided in the contract?",
#     "What is the renewal term specified in the agreement?",
#     "What is the notice period required to terminate the renewal?",
#     "What is the governing law stated in the agreement?",
#     "What non-compete clause is included in the agreement?",
#     "What exclusivity rights are granted in the agreement?",
#     "What rights are granted under the license in the agreement?",
#     "What audit rights are provided in the agreement?",
#     "What uncapped liability provisions are mentioned in the agreement?",
#     "What is the cap on liability mentioned in the agreement?",
#     "What insurance requirements are specified in the agreement?"
# ]



questions = [
    # Parties
    "Who are the parties involved in the agreement? Identify all parties, companies, organizations, or entities involved in the agreement. Include the names of both contracting parties.", 

    # Agreement date
    "What is the agreement date mentioned in the contract?, locate the agreement date, execution date, or signature date mentioned anywhere in the contract.", 

    # Effective date
    "What is the effective date, commencement date, or the 'effective as of' date of the agreement?",

    # Expiration date
    "What is the expiration date, end date, or the date on which the agreement expires or terminates?",

    # Exact expiration date
    "Provide the exact expiration/termination date specified in the agreement. Look for phrases like 'expires on', 'valid until', or 'termination date'.",

    # Renewal term
    "What renewal term, auto-renewal period, or renewal clause duration is specified in the agreement?",

    # Notice period for renewal termination
    "What is the notice period required to terminate or prevent renewal of the agreement? Search for 'notice', 'days', or 'written notice'.",

    # Governing law
    "What governing law, jurisdiction, or venue is specified in the agreement?",

    # Non-compete clause
    "What non-compete clause is included in the agreement? Identify any non-compete, non-solicitation, or restriction-on-competition clause included in the agreement.",

    # Exclusivity
    "What exclusivity rights, exclusive obligations, or exclusive territory provisions are granted in the agreement?",

    # License rights
    "What license rights, permissions, or authorized uses are granted under the agreement?",

    # Audit rights
    "What audit rights, inspection rights, or verification rights are provided in the agreement?",

    # Uncapped liability
    "What uncapped liability provisions are mentioned in the agreement? Identify any uncapped liability provisions, unlimited liability terms, or liability without cap mentioned in the agreement.",

    # Liability cap
    "What is the cap on liability, liability limit amount, or maximum liability specified in the agreement?",

    # Insurance requirements
    "What insurance requirements, insurance obligations, or coverage types are specified in the agreement?",
]



# -----------------------------
# Prompt Creation
# -----------------------------
def create_prompt(question, contexts):
    contexts = [f"#Context {i+1}: {contexts[i]}" for i in range(len(contexts))]
    context_string = "\n".join(contexts)
    return f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
#Question: {question}
{context_string}
#Answer:"""

# -----------------------------
# Main Loop
# -----------------------------
for index_, name in enumerate(sorted(os.listdir(pdf_source))[s_index:e_index]):
    try:
        top_5, contexts, answers, time_inf = [], [], [], []

        print()
        print("*"*100)
        print()
        print(f"Processing {index_}: {name}")
        print()
        print("*"*100)
        print()

        time_start = time.time()

        loader = PyPDFLoader(f"{pdf_source}{name}")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        splits = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        time_end = time.time()

        print()
        print("*"*100)
        print()
        print()
        print("*"*100)
        print()
        print(f"V-DB Loading Time: {time_end - time_start}")
        print()
        print("*"*100)
        print()
        print()
        print("*"*100)
        print()


        for q in questions:
            r_doc = retriever.invoke(q)
            r_doc_ = [r.page_content for r in r_doc]
            top_5.append(r_doc_)

        for i, q in enumerate(questions):

            time_start = time.time()
            prompt = create_prompt(q, top_5[i])
            ans = run_inference(prompt)
            ans = ans.replace(prompt, '')

            time_end = time.time()

            print()
            print("*"*100)
            print()
            print()
            print("*"*100)
            print()
            print(f"V-DB Loading Time: {time_end - time_start}")
            print()
            print("*"*100)
            print()
            print()
            print("*"*100)
            print()

            contexts.append(prompt)
            answers.append(ans)
            time_inf.append(time_end - time_start)

        result_df = pd.DataFrame({"categories" : categories,"question": questions, "top_5": top_5, "prompt": contexts, "answers": answers, "time_inf" : time_inf})
        result_df.to_csv(f"{output_folder}/{name[:-4]}-dif-q.csv", index=False)

        vectorstore.delete_collection()

    except Exception as e:
        print(f"Error processing {name}: {e}")
        continue
