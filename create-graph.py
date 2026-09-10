# import subprocess

# # "pip install torch==2.0.1",
# commands = [
# "pip install --upgrade pip",
# "pip install langchain==0.3.20",
# "pip install langchain-community==0.3.19",
# "pip install langchainhub",
# "pip install chromadb",
# "pip install bs4",
# "pip install pypdf",
# "pip install langchain-huggingface",
# "pip install langchain-huggingface==1.0.0", 
# "pip install ragatouille",
# "pip install openpyxl",
# "pip install lxml",
# "pip install pandas",
# "pip install --upgrade transformers==4.44.2",
# "pip install --upgrade peft==0.11.1", 
# "pip install --upgrade sentence-transformers", 
# "pip install accelerate",
# "pip uninstall -y apex",
# "pip install FlagEmbedding==1.3.2",
# "pip install networkx" , 
# "pip install flair", 
# "pip install IProgress", 
# "pip install keybert", 


# ]


# for cmd in commands:
#     subprocess.run(cmd, shell=True)


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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import pandas as pd
import string
from langchain.docstore.document import Document
import os
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint
import re
from tqdm import tqdm
# from sentence_transformers import SentenceTransformer, util
import torch
from langchain_community.vectorstores import Chroma
from transformers import set_seed
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig, pipeline
from huggingface_hub import login
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import classification_report
import json
import time
import ast

from pypdf import PdfReader
# import nltk
import re
import json
from typing import List
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import re
from tqdm import tqdm
import unicodedata

from flair.models import SequenceTagger
from flair.data import Sentence
from keybert import KeyBERT
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple
import math
import pickle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--f1', type=int, default = 400)
parser.add_argument('--f2', type=int, default = 500)
parser.add_argument('--f3', type=int, default = 1000)
parser.add_argument('--f4', type=int, default = 200)
parser.add_argument('--f5', type=float, default = 0.80)
parser.add_argument('--f6', type=bool, default = False)

args, unknown_args = parser.parse_known_args()

if unknown_args:
    print(f"Unrecognized arguments: {unknown_args}")




min_tokens = args.f1
max_tokens = args.f2
Chunk_Size = args.f3
Chunk_Overlap = args.f4
similarity_threshold = args.f5
is_sim_chunk = args.f6 
base_path = "/workspace/data/Momojit/Contract-QA2/cuad-pdf"
save_graph_path = "/workspace/data/Momojit/Graph-KD/save-graph3/"
PDFs = sorted(os.listdir(base_path))[20:]


print()
print("*"*100)
print("*"*100)
print()
print(f"min_tokens : {min_tokens}")
print(f"max_tokens : {max_tokens}")
print(f"Chunk_Size : {Chunk_Size}")
print(f"Chunk_Overlap : {Chunk_Overlap}")
print(f"similarity_threshold : {similarity_threshold}")
print(f"is_sim_chunk : {is_sim_chunk}")
print()
print("*"*100)
print("*"*100)
print()




model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
tagger = SequenceTagger.load("ner-ontonotes-fast")




# Define OntoNotes entity categories
ENTITY_CATEGORIES = [
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT",
    "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME",
    "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"
]

kw_model = KeyBERT()


# ------------------- Text Cleaning -------------------
def clean_text(text: str) -> str:
    """Normalize and clean PDF text."""
    text = unicodedata.normalize('NFC', text)                # Normalize Unicode
    text = re.sub(r'-\s+', '', text)                         # Fix hyphenation
    text = re.sub(r'Page\s*\d+', '', text)                   # Remove page numbers
    text = re.sub(r'\s+', ' ', text)                         # Collapse whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)               # Remove non-ASCII
    return text.strip()

# ------------------- Token Counting -------------------
def count_tokens(text: str) -> int:
    """Count tokens using MiniLM tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))

# ------------------- PDF Extraction -------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract and clean text from PDF."""
    reader = PdfReader(pdf_path)
    full_text = []
    for page in tqdm(reader.pages):
        text = page.extract_text()
        if text:
            full_text.append(clean_text(text))
    return " ".join(full_text)



def semantic_chunker(text: str,
                     similarity_threshold: float = 0.80,
                     min_tokens: int = min_tokens,
                     max_tokens: int = max_tokens,
                     overlap_sentences: int = 2) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    embeddings = model.encode(sentences, convert_to_tensor=True)

    chunks = []
    current_chunk = [sentences[0]]
    current_tokens = count_tokens(sentences[0])

    for i in tqdm(range(1, len(sentences))):
        similarity = util.pytorch_cos_sim(embeddings[i - 1], embeddings[i]).item()
        sentence_tokens = count_tokens(sentences[i])

        if (current_tokens + sentence_tokens > max_tokens) or \
           (similarity < similarity_threshold and current_tokens >= min_tokens):
            chunks.append(" ".join(current_chunk))
            # Add overlap from previous chunk
            current_chunk = current_chunk[-overlap_sentences:] + [sentences[i]]
            current_tokens = count_tokens(" ".join(current_chunk))
        else:
            current_chunk.append(sentences[i])
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks



# ------------------- Coverage Test -------------------
def verify_chunk_coverage(original_text: str, chunks: List[str]) -> float:
    """Check coverage percentage of original text in chunks."""
    def normalize(text):
        return re.sub(r'\s+', ' ', text).strip().lower()
    
    original_norm = normalize(original_text)
    combined_norm = normalize(" ".join(chunks))
    
    # Compute coverage ratio
    
    coverage_ratio = len(set(original_norm.split()) & set(combined_norm.split())) / len(original_norm.split())
    flag = original_norm in combined_norm
    
    print({
        "is_covered" : flag, 
        "coverage_ratio" : coverage_ratio
    })
    

# if(is_sim_chunk ):


#     text = extract_text_from_pdf(PDF_FILE)
#     chunks = semantic_chunker(text, similarity_threshold, min_tokens, max_tokens)

# else: 
#     loader = PyPDFLoader(PDF_FILE)
#     docs = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=Chunk_Size, chunk_overlap=Chunk_Overlap, add_start_index=True)
#     chunks = text_splitter.split_documents(docs)




def extract_ner_grouped(chunk: str) -> dict:

    # Initialize dictionary for all categories
    grouped_entities = {cat: [] for cat in ENTITY_CATEGORIES}

    # Create Flair sentence and predict NER
    sentence = Sentence(chunk)
    tagger.predict(sentence)

    # Iterate over detected entities
    for entity in sentence.get_spans('ner'):
        entity_type = entity.tag.upper()
        if entity_type in grouped_entities and entity.text not in grouped_entities[entity_type]:
            grouped_entities[entity_type].append(entity.text)

    return grouped_entities



def extract_top_keywords(chunk: str, n : float = 0.10) -> list:
    """
    Extract top 10% keywords from a text chunk using KeyBERT.

    Args:
        chunk (str): Text chunk to process.

    Returns:
        list: List of top keywords (strings).
    """
    # Extract keywords with KeyBERT (default uses BERT embeddings)
    n = int(len(chunk.split())*n)
    keywords = kw_model.extract_keywords(chunk, keyphrase_ngram_range=(1, 1), stop_words=None,top_n = n)
    


    return keywords



# if(is_sim_chunk ):


#     text = extract_text_from_pdf(PDF_FILE)
#     chunks = semantic_chunker(text, similarity_threshold, min_tokens, max_tokens)

# else: 
#     loader = PyPDFLoader(PDF_FILE)
#     docs = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=Chunk_Size, chunk_overlap=Chunk_Overlap, add_start_index=True)
#     chunks = text_splitter.split_documents(docs)




# ex_ner_chunks = []
# top_keyword_chunks = []

# for c in tqdm(chunks): 
    
#     top_keywords = extract_top_keywords(c)
#     ex_ner = extract_ner_grouped(c)
    
#     top_keyword_chunks.append(top_keywords)
#     ex_ner_chunks.append(ex_ner)



def entity_overlap_score(chunk_i, chunk_j):

    # Extract all entities from both chunks
    entities_i = set([e.lower().strip() for ents in chunk_i.values() for e in ents])
    entities_j = set([e.lower().strip() for ents in chunk_j.values() for e in ents])
    
    # Handle empty sets
    if not entities_i and not entities_j:
        return 0.0
    
    # Compute Jaccard similarity
    intersection = entities_i.intersection(entities_j)
    union = entities_i.union(entities_j)
    
    return len(intersection) / len(union)



def keyword_overlap_score(keywords_i, keywords_j):

    # Extract keywords and normalize
    set_i = set([kw.lower().strip() for kw, _ in keywords_i])
    set_j = set([kw.lower().strip() for kw, _ in keywords_j])
    
    # Handle empty sets
    if not set_i and not set_j:
        return 0.0
    
    # Compute Jaccard similarity
    intersection = set_i.intersection(set_j)
    union = set_i.union(set_j)
    
    return len(intersection) / len(union)




def embedding_cosine_similarity(chunk_i: str, chunk_j: str) -> float:

    # Encode both chunks
    emb_i = model.encode(chunk_i, convert_to_tensor=True)
    emb_j = model.encode(chunk_j, convert_to_tensor=True)
    
    # Compute cosine similarity
    similarity = util.cos_sim(emb_i, emb_j)
    
    return float(similarity)




for pdf_name in tqdm(PDFs): 
    
    try: 
 
        print()
        print("*"*100)
        print()
        print(pdf_name)
        print()
        print("*"*100)
        print()

        PDF_FILE = os.path.join(base_path,pdf_name)

        print()
        print(f"PDF_FILE : {PDF_FILE}")
        print()


        if(is_sim_chunk ):

            print()
            print(f"is_sim_chunk : {is_sim_chunk}")
            print()


            text = extract_text_from_pdf(PDF_FILE)
            chunks = semantic_chunker(text, similarity_threshold, min_tokens, max_tokens)

        else: 

            print()
            print(f"is_sim_chunk : {is_sim_chunk}")
            print()


            loader = PyPDFLoader(PDF_FILE)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=Chunk_Size, chunk_overlap=Chunk_Overlap, add_start_index=True)
            chunks = text_splitter.split_documents(docs)
            chunks = [c.page_content for c in chunks]
        
        print()
        print()
        print()
        
        ex_ner_chunks = []
        top_keyword_chunks = []

        for c in tqdm(chunks): 

            top_keywords = extract_top_keywords(c)
            ex_ner = extract_ner_grouped(c)

            top_keyword_chunks.append(top_keywords)
            ex_ner_chunks.append(ex_ner)


        # Initialize graph
        G = nx.Graph()

        # Step 1: Add nodes with full properties
        for idx, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()  # Convert tensor to list for storage
            G.add_node(idx,
                    chunk=chunk,
                    embedding=embedding,
                    NER=ex_ner_chunks[idx],
                    keywords=top_keyword_chunks[idx])

        # Step 2: Define weights for edge scoring
        alpha, beta, gamma, delta = 0.1, 0.1, 0.1, 0.1  # Adjust weights as needed



        # Step 3: Add edges with combined score and metadata
        for i in range(len(chunks) - 1):          # Outer loop
            for j in range(i + 1, len(chunks)):   # Inner loop
                ner_score = entity_overlap_score(ex_ner_chunks[i], ex_ner_chunks[j])
                key_score = keyword_overlap_score(top_keyword_chunks[i], top_keyword_chunks[j])
                sim_score = embedding_cosine_similarity(chunks[i], chunks[j])
                rel_dist = 1 / abs(i - j)

                edge_weight = (alpha * sim_score) + (beta * ner_score) + (gamma * key_score) + (delta * rel_dist)

                G.add_edge(i, j,
                        weight=edge_weight,
                        sim_score=sim_score,
                        ner_score=ner_score,
                        key_score=key_score,
                        rel_dist=rel_dist)


    #     for edge_id, (u, v, data) in enumerate(G.edges(data=True), start=1):
    #         print(f"\nEdge {edge_id}: {u} ↔ {v}")
    #         print("-" * 80)
    #         print(f"Chunk {u}: {G.nodes[u]['chunk'][:80]}...")
    #         print(f"NER {u}: {G.nodes[u]['NER']}")
    #         print(f"Keywords {u}: {G.nodes[u]['keywords']}")
    #         print()
    #         print(f"Chunk {v}: {G.nodes[v]['chunk'][:80]}...")
    #         print(f"NER {v}: {G.nodes[v]['NER']}")
    #         print(f"Keywords {v}: {G.nodes[v]['keywords']}")
    #         print()
    #         print(f"Edge Properties → weight: {data['weight']:.4f}, sim_score: {data['sim_score']:.4f}, "
    #             f"ner_score: {data['ner_score']:.4f}, key_score: {data['key_score']:.4f}, rel_dist: {data['rel_dist']:.4f}")


        if(is_sim_chunk):

            Graph_name = f"{pdf_name[:-4]}-{is_sim_chunk}-min-{min_tokens}-max-{max_tokens}.pkl" 

        else: 

            Graph_name = f"{pdf_name[:-4]}-{is_sim_chunk}-Chunk_Size-{Chunk_Size}-Chunk_Overlap-{Chunk_Overlap}.pkl" 



        with open(os.path.join(save_graph_path,Graph_name), "wb") as f:
            pickle.dump(G, f)
        
        print()
        print(f"saved: {Graph_name}")
        print()
    
    
    
    except Exception as e: 
        
        print(e)



# def compute_query_signature(query: str):
#     # Encode query
#     query_emb = model.encode(query, convert_to_tensor=True)
#     # Simple keyword extraction (replace with KeyBERT for better results)
#     query_keywords = set(query.lower().split())
#     # Simple entity extraction placeholder (replace with spaCy for real NER)
#     query_entities = set(query.lower().split())
#     return query_emb, query_keywords, query_entities

# def compute_lms(node_data, query_emb, query_keywords, query_entities,
#                 w1=0.5, w2=0.25, w3=0.25):
#     # Cosine similarity
#     node_emb = node_data['embedding']
#     sim_score = float(util.cos_sim(query_emb, node_emb))
#     # Keyword overlap
#     node_keywords = set([kw.lower() for kw, _ in node_data['keywords']])
#     key_overlap = len(node_keywords & query_keywords) / len(node_keywords | query_keywords) if node_keywords else 0
#     # Entity overlap
#     node_entities = set([e.lower() for ents in node_data['NER'].values() for e in ents])
#     ent_overlap = len(node_entities & query_entities) / len(node_entities | query_entities) if node_entities else 0
#     return (w1 * sim_score) + (w2 * ent_overlap) + (w3 * key_overlap)

# def retrieve_nodes(G, query, top_k=3, bfs_depth=2):
#     query_emb, query_keywords, query_entities = compute_query_signature(query)
    
#     print(query_emb, query_keywords, query_entities)
    
   
#     scores = {node: compute_lms(G.nodes[node], query_emb, query_keywords, query_entities) for node in G.nodes()}
    
#     print(scores)
    
#     # Select top-K anchors
#     anchors = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
#     print(anchors)
    
#     # BFS expansion
#     visited = set()
#     candidates = {}
#     for anchor, anchor_score in anchors:
#         queue = [(anchor, 0)]
#         while queue:
#             current, depth = queue.pop(0)
#             if current in visited or depth > bfs_depth:
#                 continue
#             visited.add(current)
#             # Aggregate score
#             candidates[current] = candidates.get(current, 0) + anchor_score
#             for neighbor in G.neighbors(current):
#                 edge_weight = G.edges[current, neighbor]['weight']
#                 candidates[neighbor] = candidates.get(neighbor, 0) + edge_weight
#                 queue.append((neighbor, depth + 1))
    
#     # Rank candidates
#     ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
#     return ranked

# # Example usage
# query = "When did the Restated Net Investment Income Maintenance Agreement become effective?"
# results = retrieve_nodes(G, query)

# # Print top results
# for node_id, score in results[:5]:
#     print(f"Node {node_id} (Score: {score:.4f}) ...")


