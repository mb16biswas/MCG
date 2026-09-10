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
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import time
import os
import torch.nn.functional as F
import re
from collections import deque
import gc



# -----------------------------
# Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Run inference using GPT-4 API or HuggingFace model")
parser.add_argument('--mode', type=str, choices=['gpt4', 'hf'], default='hf', help="Choose 'gpt4' or 'hf' for HuggingFace model")
parser.add_argument('--f1', type=int, default=2, help="Model choice: 1=Saul-7B, 2=Mistral-2-7B, 3=Mistral-3-7B, 4=Meta-Llama-3-8B, 5=llama-2-7b")
parser.add_argument('--f2', type=int, default=2, help="Variant for HF Model")
parser.add_argument('--f3', type=int, default=3, help="Graph Source")
parser.add_argument('--token', type=str, default= "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", help="HuggingFace API token")
args, unknown_args = parser.parse_known_args()

if unknown_args:
    print(f"Unrecognized arguments: {unknown_args}")

# -----------------------------
# Configurations
# -----------------------------
API_URL = "http://10.81.0.36:8000/openai_service"

set_seed(42)

if(args.f3 == 1): 

    graph_source =  "/workspace/data/Graph-KD/save-graph/"
    saved_path = "/workspace/data/Graph-KD/results/proposed-v1-final-v2/"

elif(args.f3 == 2): 
    
    graph_source =  "/workspace/data/Graph-KD/save-graph2/"
    saved_path = "/workspace/data/Graph-KD/results/proposed-v2-final-v2/"

else: 
    
    graph_source = "/workspace/data/Graph-KD/save-graph4/"
    saved_path = "/workspace/data/Graph-KD/results/proposed-ca-final-v2/"

# saved_path = "/workspace/data/Graph-KD/results/proposed-v2/"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)



print()
print("*"*100)
print()
print("proposed-updated.py")
print()
print("*"*100)
print()
print(f"Mode: {args.mode}")
print(f"f1: {args.f1}")
print(f"f2: {args.f2}")
print(f"f3: {args.f3}")
print(f"graph_source: {graph_source}")
print(f"saved_path: {saved_path}")



# -----------------------------
# Create Output Folder
# -----------------------------
def create_folder(folder_name):
    folder_path = f"{saved_path}{folder_name}"
    os.makedirs(folder_path, exist_ok=True)
    print(f"Folder ready: {folder_path}")
    return folder_path






# def prepare_graph_cache(G, device=None):
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     node_ids = list(G.nodes())
#     emb_list = []
#     ok_nodes = []
#     for n in node_ids:
#         emb = G.nodes[n].get('embedding', None)
#         if emb is None:
#             continue
#         t = torch.as_tensor(emb, dtype=torch.float32)
#         if t.ndim != 1:
#             continue
#         if torch.isnan(t).any() or torch.isinf(t).any():
#             continue
#         emb_list.append(t)
#         ok_nodes.append(n)

#     if not emb_list:
#         raise ValueError("No valid embeddings in graph nodes.")

#     node_embs = torch.stack(emb_list, dim=0).to(device)
#     node_embs = F.normalize(node_embs, p=2, dim=1)  # L2 normalize once

#     node_index_map = {nid: idx for idx, nid in enumerate(ok_nodes)}

#     # Cache lowercase keyword/entity sets once
#     for n in ok_nodes:
#         nd = G.nodes[n]
#         if "keywords_set" not in nd:
#             kw = nd.get("keywords", [])
#             nd["keywords_set"] = set(k.lower() for k, _ in kw) if kw else set()
#         if "entities_set" not in nd:
#             ner = nd.get("NER", {})
#             nd["entities_set"] = set(e.lower() for ents in ner.values() for e in ents) if ner else set()

#     G.graph["cache"] = {
#         "device": device,
#         "node_ids": ok_nodes,
#         "node_embs": node_embs,
#         "node_index_map": node_index_map,
#     }





def prepare_graph_cache(G, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    node_ids = list(G.nodes())
    emb_list = []
    ok_nodes = []
    for n in node_ids:
        emb = G.nodes[n].get('embedding', None)
        if emb is None:
            continue
        t = torch.as_tensor(emb, dtype=torch.float32)
        if t.ndim != 1:
            continue
        if torch.isnan(t).any() or torch.isinf(t).any():
            continue
        emb_list.append(t)
        ok_nodes.append(n)

    if not emb_list:
        raise ValueError("No valid embeddings in graph nodes.")

    node_embs = torch.stack(emb_list, dim=0).to(device)
    node_embs = F.normalize(node_embs, p=2, dim=1)  # L2 normalize once

    node_index_map = {nid: idx for idx, nid in enumerate(ok_nodes)}

    # --- Helpers to robustly normalize any 'entity-like' value to strings ---
    def _iter_norm_strings(val):
        """
        Yield 0..N normalized (lower/strip) string tokens from val.
        - Flattens lists/sets/tuples.
        - For 2-tuples (e.g., date ranges), yields BOTH individual endpoints AND
          an atomic joined token "start..end".
        - Stringifies numbers/dicts safely.
        - Skips None/empty strings.
        """
        if val is None:
            return

        # Containers: list/tuple/set
        if isinstance(val, (list, tuple, set)):
            # If it's exactly a 2-tuple, build an atomic range token as well
            if isinstance(val, tuple) and len(val) == 2:
                s1 = (str(val[0]).strip().lower() if val[0] is not None else "")
                s2 = (str(val[1]).strip().lower() if val[1] is not None else "")
                if s1:
                    yield s1
                if s2:
                    yield s2
                if s1 and s2:
                    # atomic representation for full-range equality
                    yield f"{s1}..{s2}"
                return  # we've handled the 2-tuple fully

            # Generic container: yield each element normalized
            for x in val:
                if x is None:
                    continue
                sx = str(x).strip().lower()
                if sx:
                    yield sx
            return

        # Dicts: stringify (optional alternative: ignore or extract specific fields)
        if isinstance(val, dict):
            s = str(val).strip().lower()
            if s:
                yield s
            return

        # Everything else: coerce to string
        s = str(val).strip().lower()
        if s:
            yield s

    # Cache lowercase keyword/entity sets once
    for n in ok_nodes:
        nd = G.nodes[n]

        # Keywords can be list of (kw, score)
        if "keywords_set" not in nd:
            kw = nd.get("keywords", []) or []
            kw_set = set()
            for item in kw:
                # Expecting (keyword, score); but be defensive
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    k = item[0]
                else:
                    k = item
                for token in _iter_norm_strings(k):
                    kw_set.add(token)
            nd["keywords_set"] = kw_set

        if "entities_set" not in nd:
            ner = nd.get("NER", {}) or {}
            ent_set = set()
            for ents in ner.values():         # ents is typically a list
                if not ents:
                    continue
                for e in ents:                 # e can be str, tuple, list, dict, None
                    for token in _iter_norm_strings(e):
                        ent_set.add(token)
            nd["entities_set"] = ent_set

    G.graph["cache"] = {
        "device": device,
        "node_ids": ok_nodes,
        "node_embs": node_embs,
        "node_index_map": node_index_map,
    }









def safe_jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0

def compute_query_signature_fast(query: str, device: str):
    toks = set(re.findall(r"\w+", query.lower()))
    q = embedder.encode(query, convert_to_tensor=True).to(device)
    q = F.normalize(q, p=2, dim=0)
    return q, toks, toks



def retrieve_nodes_fast(
    G,
    query: str,
    top_k: int = 5,
    num_anchors: int = 2,
    bfs_depth: int = 2,
    w_sim: float = 0.1,
    w_key: float = 0.1,
    w_ent: float = 0.1,
    w_edge: float = 0.05,
    use_overlap: bool = True,
    max_neighbors_per_node: int = None  # e.g., set 20 to cap fan-out
):
    cache = G.graph.get("cache", None)
    if cache is None:
        prepare_graph_cache(G)
        cache = G.graph["cache"]

    device = cache["device"]
    node_ids = cache["node_ids"]
    node_embs = cache["node_embs"]          # (N, d)
    node_index_map = cache["node_index_map"]

    # 1) signature
    t0 = time.time()
    q, qkw, qent = compute_query_signature_fast(query, device)
    t1 = time.time()

    # 2) batched cosine via dot (cosine because L2 normalized)
    with torch.no_grad():
        sim = node_embs @ q  # (N,)
    t2 = time.time()

    # 3) anchors by top-k similarity
    m = max(1, min(num_anchors, top_k, sim.numel()))
    a_scores, a_idx = torch.topk(sim, k=m)
    anchors = [(node_ids[i.item()], a_scores[j].item()) for j, i in enumerate(a_idx)]
    anchor_nodes = [a for a, _ in anchors]
    t3 = time.time()

    # 4) BFS with deque; score neighbors
    visited = set(anchor_nodes)
    cand = {}
    dq = deque()
    for n, _ in anchors:
        dq.append((n, 0))

    while dq:
        cur, depth = dq.popleft()
        if depth > bfs_depth:
            continue

        # Optional: cap neighborhood fan-out
        neighbors = list(G.neighbors(cur))
        if max_neighbors_per_node is not None and len(neighbors) > max_neighbors_per_node:
            # pick top by edge weight
            neighbors = sorted(
                neighbors,
                key=lambda nbr: float(G.edges[cur, nbr].get("weight", 0.0)),
                reverse=True
            )[:max_neighbors_per_node]

        for nbr in neighbors:
            if nbr in visited:
                continue
            visited.add(nbr)

            idx = node_index_map.get(nbr, None)
            if idx is None:
                continue
            sim_n = sim[idx].item()

            k_ov = e_ov = 0.0
            if use_overlap:
                nd = G.nodes[nbr]
                k_ov = safe_jaccard(nd.get("keywords_set", set()), qkw)
                e_ov = safe_jaccard(nd.get("entities_set", set()), qent)

            edge_w = float(G.edges[cur, nbr].get("weight", 0.0))
            score = (w_sim * sim_n) + (w_key * k_ov) + (w_ent * e_ov) + (w_edge * edge_w)
            cand[nbr] = max(cand.get(nbr, float("-inf")), score)

            dq.append((nbr, depth + 1))
    t4 = time.time()

    # 5) Final ranking
    remain = max(0, top_k - len(anchor_nodes))
    top_neighbors = []
    if remain > 0 and cand:
        top_neighbors = [n for n, _ in sorted(cand.items(), key=lambda x: x[1], reverse=True)[:remain]]

    result = anchor_nodes + top_neighbors
    t5 = time.time()

    # Debug timings (optional)
    print(f"[fast] sig:{t1-t0:.4f}s sim:{t2-t1:.4f}s anchors:{t3-t2:.4f}s bfs:{t4-t3:.4f}s rank:{t5-t4:.4f}s N={len(node_ids)}")
    return result


# def make_context_block(G, node_id, idx):
#     nd = G.nodes[node_id]
#     chunk = nd.get("chunk", "")
#     keywords = [k for k, _ in nd.get("keywords", [])]
#     entities = nd.get("NER", {})
    

#     return f"""
# #Context {idx+1}
# ## Chunk
# {chunk}

# ## Metadata

# - Keywords: {keywords}
# - Entities: {entities}

# """.strip()



def make_context_block(G, node_id, idx):
    nd = G.nodes[node_id]
    
    chunk = nd.get("chunk", "").strip()
    
    # Extract keywords safely
    kw_pairs = nd.get("keywords", [])
    keywords = [k for k, _ in kw_pairs] if kw_pairs else []
    
    # Extract entities safely & remove empty lists
    ent_dict = nd.get("NER", {}) or {}
    entities = {k: v for k, v in ent_dict.items() if v}  # keep only non-empty
    
    # Build metadata lines
    meta_lines = []
    if keywords:
        meta_lines.append(f"- Keywords: {keywords}")
    if entities:
        meta_lines.append(f"- Entities: {entities}")
    
    # If EVERYTHING was empty, show placeholder
    if not meta_lines:
        meta_lines.append("- (no metadata)")

    metadata_str = "\n".join(meta_lines)

    return f"""
##
##
{chunk}

## Metadata
{metadata_str}
""".strip()





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
        f_name = f"cuad-prop-saul"

        s_index = 0
        e_index = 100

    elif args.f1 == 2:
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
        f_name = f"cuad-prop-mis-2"

        s_index = 0
        e_index = 100

    elif args.f1 == 3:
        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        f_name = f"cuad-prop-mis-3"

        s_index = 38
        e_index = 138

    elif args.f1 == 4:
        repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        f_name = f"cuad-prop-llama-3"

        s_index = 0
        e_index = 100

    else:
        repo_id = "daryl149/llama-2-7b-chat-hf"
        f_name = f"cuad-prop-llama-2"
    
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
# Questions
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






#######################################################v3 prompt ##############################################

# f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# #Question: {question}
# {context_string}
# #Answer:"""        

# [f"#Context {i+1}: {contexts[i]}" for i in range(len(contexts))]

####################################################### v3 prompt ##############################################




#######################################################v4 prompt ##############################################

# You are a highly precise extraction assistant. 
# Your answer must be taken directly from the context. 
# Do not infer, transform, or paraphrase the information. 
# Only copy the exact text span from the context that best answers the question. 
# If the answer is not present in the context, reply: "I don’t know."

# #Question: {question}
# {context_string}
# #Answer:


# [f"#Context {i+1}: {contexts[i]}" for i in range(len(contexts))]

####################################################### v4 prompt ##############################################


#########################################################V5 Prompt ##############################################

# You are a contract-analysis assistant that extracts information only from the provided evidence.
# Your task is EXTRACTIVE, not interpretive.

# Follow these principles:
# - Your answer must be a short phrase copied exactly from the context.
# - Do NOT rephrase, infer, generalize, or guess.
# - Rely only on the text shown in the evidence blocks.
# - If none of the evidence contains the answer, reply exactly: "I don’t know."
# - Prefer text that appears closest to the meaning of the question (e.g., dates near “Agreement”, “Effective”, or signature blocks).

# # EVIDENCE
# The following are the most relevant extracted portions of the contract. 
# Each block contains:
# - Chunk text
# - Metadata (entities, dates, keywords)

# Review each block independently.




# -----------------------------
# Prompt Creation
# -----------------------------
def create_prompt(question, contexts):
    contexts = [f"# Evidence Block {i+1}: {contexts[i]}" for i in range(len(contexts))]
    context_string = "\n".join(contexts)
    return f"""You are a contract-analysis assistant that extracts information only from the provided evidence.
Your task is EXTRACTIVE, not interpretive.

Follow these principles:
- Your answer must be a short phrase copied exactly from the context.
- Do NOT rephrase, infer, generalize, or guess.
- Rely only on the text shown in the evidence blocks.
- If none of the evidence contains the answer, reply exactly: "The Answer is not mentioned in the context"
- Prefer text that appears closest to the meaning of the question (e.g., dates near “Agreement”, “Effective”, or signature blocks).

# EVIDENCE
The following are the most relevant extracted portions of the contract. 
Each block contains:
- Chunk text
- Metadata (entities, dates, keywords)

Review each block independently.

#Question: {question} \n
{context_string}
#Answer: (Copy the exact phrase from the evidence that answers the question.)"""






# === [ADDED] Single-evidence extractive prompt (one context at a time) ===
def create_single_evidence_prompt(question: str, evidence_block: str) -> str:
    return f"""You are a contract-analysis assistant that extracts information only from the provided evidence.
Your task is EXTRACTIVE, not interpretive.

Follow these principles:
- Your answer must be a short phrase copied exactly from the context.
- Do NOT rephrase, infer, generalize, or guess.
- Rely only on the text shown in the evidence block.

# EVIDENCE
The following are the most relevant extracted portions of the contract. 
Each block contains:
- Chunk text
- Metadata (entities, dates, keywords)

# Question
{question}

# Evidence
{evidence_block}

# Answer (one line only):
"""

# # === [ADDED] Answer cleaner (keep first non-empty line and strip quotes) ===
# def clean_answer(raw: str) -> str:
#     a = (raw or "").strip()
#     # Remove leading "Answer:" variants
#     a = re.sub(r'^\s*(#?\s*Answer\s*:?)\s*', '', a, flags=re.IGNORECASE)
#     # Take first non-empty line
#     for line in a.splitlines():
#         line = line.strip()
#         if line:
#             a = line
#             break
#     # Strip wrapping quotes
#     a = a.strip().strip('"').strip("'").strip()
#     # Normalize NOT FOUND variant
#     if re.fullmatch(r'(?is)\s*(the\s*answer\s*is\s*not\s*mentioned\s*in\s*the\s*context)\s*', a):
#         return "The Answer is not mentioned in the context"
#     return a if a else "The Answer is not mentioned in the context"

# === [ADDED] Judge prompt to select best candidate given (evidence, answer) pairs ===



################################################V1#############################################

# def build_judge_prompt(question: str, evidence_blocks: list, answers: list) -> str:
#     blocks = []
#     for i, (ev, ans) in enumerate(zip(evidence_blocks, answers)):
#         blocks.append(
# f"""### Candidate {i}
# # Evidence
# {ev}

# # ExtractedAnswer
# "{ans}" """
#         )
#     blocks_text = "\n\n".join(blocks)
#     return f"""You are a meticulous contract-analysis judge.
# Given a question and multiple candidates (each has Evidence + ExtractedAnswer), choose the SINGLE best candidate
# whose ExtractedAnswer is directly supported by its Evidence. If none is supported

# Rules:
# - STRICTLY extractive: the chosen text MUST appear verbatim in its Evidence.
# - Prefer precise, minimal spans that directly answer the question.
# - If multiple are valid, choose the one with the clearest, most local support.


# # Question
# {question}

# # Candidates
# {blocks_text}

# #Answer:
# """

################################################V2#############################################

def build_judge_prompt(question: str, evidence_blocks: list, answers: list) -> str:
    blocks = []
    for i, (ev, ans) in enumerate(zip(evidence_blocks, answers)):
        blocks.append(
f"""### Evidence Block
{ev}

### Proposed Answer
{ans}
"""
        )

    blocks_text = "\n\n".join(blocks)

    return f"""
You are an extraction judge.

Your task:
- Read the question.
- For each candidate (Evidence Block + Proposed Answer), check whether the proposed answer is **exactly supported** by its evidence.
- The answer MUST appear verbatim in the evidence and MUST directly answer the question.
- Choose the SINGLE best supported answer.
- If NONE are supported, output exactly: The Answer is not mentioned in the context

IMPORTANT:
- DO NOT mention candidate numbers.
- DO NOT reference “candidate”, “block”, “option”.
- DO NOT explain your choice.
- DO NOT output lists, analysis, or reasoning.
- DO NOT generate new text.
- ONLY output the final answer text exactly as it appears in the evidence.

# Question
{question}

# Candidates
{blocks_text}

# Final Answer:
"""



# -----------------------------
# Main Loop
# -----------------------------






for index_, name in enumerate(sorted(os.listdir(graph_source))[s_index:e_index]):

    # try:

        print()
        print("*"*100)
        print()
        print()
        print("*"*100)
        print()
        print(index_)
        print(name)
        print()
        print("*"*100)
        print()
        print()
        print("*"*100)
        print()

        t_start = time.time()

        with open(os.path.join(graph_source, name), "rb") as f:
            G = pickle.load(f)

        prepare_graph_cache(G)

        t_end = time.time()

        print()
        print("*"*100)
        print()
        print()
        print("*"*100)
        print()
        print(f"Graph Loading Time: {t_end - t_start}")
        print()
        print("*"*100)
        print()
        print()
        print("*"*100)
        print()

        top_5, contexts, answers, inf_time  = [], [], [], []
        print(f"Processing {index_}: {name}")

        graph_time = 0

        # -----------------------------
        # Retrieval stays the same
        # -----------------------------
        for q in questions:
            t_start = time.time()

            ranked_nodes = retrieve_nodes_fast(
                G, q, top_k=4, num_anchors=2, bfs_depth=2,
                max_neighbors_per_node=20  # optional guard for big graphs
            )

            r_doc_ = [make_context_block(G, node, i) for i, node in enumerate(ranked_nodes)]
            # r_doc_ = [G.nodes[node]['chunk'] for node in ranked_nodes]

            t_end = time.time()
            graph_time += t_end - t_start

            top_5.append(r_doc_) # (n,5)

        print()
        print("*"*100)
        print()
        print()
        print("*"*100)
        print()
        print(f"retrive time Avg: {graph_time/len(questions)}")
        print()
        print("*"*100)
        print()
        print()
        print("*"*100)
        print()

        # -----------------------------
        # CHANGED: per-evidence extraction + judge selection
        # -----------------------------
        for i, q in enumerate(questions):

            t_start = time.time()

            evidence_blocks = top_5[i]     # list of k contexts (k=5)
            cand_answers = []

            # (1) One context at a time -> one answer per context
            for ev in evidence_blocks:

                single_prompt = create_single_evidence_prompt(q, ev)

                raw_ans = run_inference(single_prompt)

                ans = raw_ans.replace(single_prompt, '')

                cand_answers.append(ans)

            # (2) Judge over (answers + contexts) to select final
            judge_prompt = build_judge_prompt(q, evidence_blocks, cand_answers)
            judge_raw = run_inference(judge_prompt)

            final_answer = judge_raw.replace(judge_prompt, '')
            
            t_end = time.time()

            # Store the judge prompt and the final selected answer
            contexts.append(judge_prompt)            # the prompt used in the final pass (judge)
            answers.append(final_answer)             # the final chosen answer
            inf_time.append(t_end - t_start)
            
            
            gc.collect()
            torch.cuda.empty_cache()

        print()
        print("*"*100)
        print()
        print()
        print("*"*100)
        print()
        print(f"Avg Time : {sum(inf_time)/len(inf_time)}")
        print()
        print("*"*100)
        print()
        print()
        print("*"*100)
        print()

        print()
        print("*"*100)
        print()
        print()
        print("*"*100)
        print()
        print(f"Avg Time : {sum(inf_time)/len(inf_time)}")
        print()
        print("*"*100)
        print()
        print()
        print("*"*100)
        print()

        result_df = pd.DataFrame({"categories": categories, "question": questions, "top_5": top_5, "prompt": contexts, "answers": answers, "inf_time" : inf_time})
        result_df.to_csv(f"{output_folder}/{name[:-4]}.csv", index=False)

    # except Exception as e:

    #     print()
    #     print("!"*100)
    #     print("!"*100)
    #     print()
    #     print(f"error: @ index: {index_} file name : {name}")
    #     print()
    #     print()
    #     print(e)
    #     print()
    #     print("!"*100)
    #     print("!"*100)
    #     print()
