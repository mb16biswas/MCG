import subprocess

# List of commands
commands = [
    "pip install datasets==2.16.1", 
    "pip install scikit-learn numpy pandas", 
    "pip install transformers==4.30", 
    "pip install -q -U trl accelerate", 
    "pip uninstall -y apex", 
    "pip install -U sentence-transformers",
    "pip install evaluate",
    "pip install nltk rouge_score",
    "pip install sacrebleu",
    "pip install sacremoses",
    "pip install bert_score",
    "pip install openpyxl",
    "pip install lxml",
    "pip install pandas", 
    "pip install IProgress", 

]

# Execute each command
for cmd in commands:
    subprocess.run(cmd, shell=True)



from datasets import load_metric
import evaluate
from evaluate import load
import pandas as pd
import os
import nltk
import numpy as np
import json
import re
import numpy as np
from collections import Counter
import os 
import pandas as pd
import re
import string
from collections import Counter
from difflib import SequenceMatcher
import ast


nltk.download('punkt_tab')

meteor = evaluate.load("meteor")
sari = load("sari")
rouge = load_metric("rouge",trust_remote_code=True)
bleu = load_metric("bleu",trust_remote_code=True)
sacrebleu = load_metric("sacrebleu",trust_remote_code=True)
bertscore = load("bertscore")



source = "/workspace/data/Momojit/Graph-KD/results/final-latest/"
target = "/workspace/data/Momojit/Graph-KD/results/final-scores-updated-metrics-latest/"


ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
WS = re.compile(r"\s+")



def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = ARTICLES.sub(" ", s)
    s = PUNCT.sub(" ", s)
    s = WS.sub(" ", s)
    return s.strip()

def tokenize(s: str):
    return normalize_text(s).split()

def exact_match(p: str, r: str) -> int:
    return int(normalize_text(p) == normalize_text(r))

def prf1_for_pair(p: str, r: str):
    ptoks, rtoks = tokenize(p), tokenize(r)
    if len(ptoks) == 0 and len(rtoks) == 0:
        return 1.0, 1.0, 1.0
    if len(ptoks) == 0 or len(rtoks) == 0:
        return 0.0, 0.0, 0.0

    pc, rc = Counter(ptoks), Counter(rtoks)
    overlap = sum((pc & rc).values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(ptoks)
    recall = overlap / len(rtoks)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def qa_metrics(preds, refs):
    ems, ps, rs, f1s = [], [], [], []
    for p, r in zip(preds, refs):
        ems.append(exact_match(p, r))
        p_i, r_i, f1_i = prf1_for_pair(p, r)
        ps.append(p_i); rs.append(r_i); f1s.append(f1_i)
    return {
        "EM": float(np.mean(ems)),
        "Precision": float(np.mean(ps)),
        "Recall": float(np.mean(rs)),
        "F1": float(np.mean(f1s)),
    
    }



def _coerce_to_list_of_str(gt_input):
    """
    Coerce gt_input into List[str]:
      - If it's None -> []
      - If it's already a list/tuple -> [str(x) ...]
      - If it's a string:
          * Try to parse JSON/Python list via ast.literal_eval
          * Else, treat as a single item string
    """
    if gt_input is None:
        return []
    
    # Already list/tuple
    if isinstance(gt_input, (list, tuple)):
        return [str(x) for x in gt_input if str(x).strip()]

    # String input
    if isinstance(gt_input, str):
        s = gt_input.strip()
        if not s:
            return []
        # Try to interpret as a Python/JSON list (e.g., '["a","b"]' or "['a','b']")
        if (s.startswith('[') and s.endswith(']')):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x) for x in parsed if str(x).strip()]
            except Exception:
                # Fall through to single-item handling
                pass
        # Fallback: treat as a single item string
        return s.split()
    
    # Any other type -> cast to str and wrap
    return [str(gt_input).split()]

def fuzzy_score(gt_list_or_str, pred_text, threshold=0.6):
    """
    Fuzzy partial match scoring.

    gt_list_or_str: Union[str, List[str]]  -> Ground truth party names
    pred_text: str                         -> Predicted text
    threshold: float (0..1)                -> Fuzzy match threshold

    Returns: (score, matches, total)
    """
    gt_list = _coerce_to_list_of_str(gt_list_or_str)
    pred = (pred_text or "").lower()

    matches = 0
    total = len(gt_list)
    if total == 0:
        return 0.0, 0, 0

    for gt in gt_list:
        gt_clean = (gt or "").lower()
        if not gt_clean:
            continue
        ratio = SequenceMatcher(None, gt_clean, pred).ratio()
        if ratio >= threshold:
            matches += 1

    score = matches / total if total > 0 else 0.0
    return score

# files = os.listdir(source)

# files = ["f-proposed-v5-final-cuad-prop-mis-2--results.csv",
#         "f-proposed-v5-final-cuad-prop-mis-3--results.csv",
#         "f-proposed-v5-final-cuad-prop-saul--results.csv",]

files = ['f-sparse-final-cuad-vanila-saul-bm25--results.csv',
 'f-sparse-final-cuad-vanila-mis-2-bm25--results.csv',
 'f-sparse-final-cuad-vanila-mis-3-bm25--results.csv',
 'f-sparse-final-cuad-vanila-llama-3-bm25--results.csv',
 'f-sparse-final-cuad-vanila-qwen-bm25--results.csv',
 'f-colbert-final-cuad-vanila-saul-colbert--results.csv',
 'f-colbert-final-cuad-vanila-mis-2-colbert--results.csv',
 'f-colbert-final-cuad-vanila-mis-3-colbert--results.csv',
 'f-colbert-final-cuad-vanila-llama-3-colbert--results.csv',
 'f-colbert-final-cuad-vanila-qwen-colbert--results.csv']



def rouge_score(pred,truth):
    
    print()
    print("rouge_score")
    print()


    FmeasureL = []
    FmeasureLs = []


    for i,j in zip(pred,truth):

        res = rouge.compute(predictions=[i], references=[j])


        FmeasureL.append(res["rougeL"].mid.fmeasure)
        FmeasureLs.append(res["rougeLsum"].mid.fmeasure)


    return np.mean(FmeasureL), np.mean(FmeasureLs)


def bleu_score(pred,truth):
    
    print()
    print("bleu_score")
    print()


    Blue = []

    for i,j in zip(pred,truth):


        i = [i.split(" ")]
        j = [[j.split(" ")]]

        res = bleu.compute(predictions=i, references=j)['bleu']

        Blue.append(res)

    return np.mean(Blue)


def sacrebleu_score(pred,truth):

    print()
    print("sacrebleu_score")
    print()
    

    Blue = []

    for i,j in zip(pred,truth):

        i = [i]
        j = [[j]]

        res = sacrebleu.compute(predictions=i, references=j)['score']
        Blue.append(res)

    return np.mean(Blue)


def meteor_score(pred,truth):
    
    
    print()
    print("meteor_score")
    print()

    Meteor = []

    for i,j in zip(pred,truth):

        i = [i]
        j = [j]

        res = meteor.compute(predictions=i, references=j)['meteor']
        Meteor.append(res)

    return np.mean(Meteor)


def sari_score(pred,truth):
    
    print()
    print("sari_score")
    print()

    Sari = []

    for i,j in zip(pred,truth):

        i = [i]
        j = [j]

        res = sari.compute(sources = i , predictions=j, references=[j])['sari']
        Sari.append(res)

    return np.mean(Sari)


def bert_score(pred,truth):

    print()
    print("bert_score")
    print()
    
    Bert_f1 = []
    Bert_recall = []
    Bert_pre = []

    for i,j in zip(pred,truth):

        i = [i]
        j = [j]

        res = bertscore.compute(predictions = i , references=j, model_type = "distilbert-base-uncased")


        Bert_f1.append(res['f1'][0])
        Bert_recall.append(res['recall'][0])
        Bert_pre.append(res['precision'][0])
        

    return np.mean(Bert_f1), np.mean(Bert_recall), np.mean(Bert_pre)

print()
print("*"*100)
print("*"*100)
print()

print()
print("*"*100)
print("*"*100)
print()

for f in files:
    
    print()
    print("*"*100)
    print("*"*100)
    print()
    print(f)
    print()
    print("*"*100)
    print("*"*100)
    print()

    d = {}
    
    d_path = os.path.join(source,f)
    
    df = pd.read_csv(d_path).head(150)

    cols = df.columns

    for i_ in range(0,len(cols)//2):

        name_ = cols[2*i_][3:]

        gt = list(df[cols[2*i_ ]])
        pred = list(df[cols[2* i_ + 1]])

        pred = [i if type(i) == str else "The Answer is not mentioned in the context" for i in pred] 
        gt = [i if type(i) == str else "The Answer is not mentioned in the context" for i in gt] 
        
        score = 0 

        for g,p in zip(gt, pred):

            score  += fuzzy_score(g, p, threshold=0.15)
      

        Bert_f1,Bert_recall,Bert_pre = bert_score(pred,gt)
        
        

        d1 = {

            "Bert_f1" : Bert_f1, 
            "Bert_recall" : Bert_recall, 
            "Bert_pre" : Bert_pre, 
            "fuzzy_match" : score / len(gt)
            
        }



        d[name_] = d1 
    

    
    with open(os.path.join(target,f[:-4] + ".json"), "w") as f_:
        json.dump(d, f_, indent=4)  
    



all_rows = []
for file in os.listdir(target):
    if file.endswith(".json"):

        json_path = os.path.join(target, file)

        with open(json_path, "r") as f:
            data = json.load(f)

        row = {"file": file.replace(".json", "")}

        # Flatten metrics horizontally
        for col_name, metrics in data.items():
            for metric_name, value in metrics.items():
                col_key = f"{col_name}_{metric_name}"
                row[col_key] = value

        all_rows.append(row)

# Convert into DataFrame
df = pd.DataFrame(all_rows)

# Sort columns: file first, then metrics alphabetically
df = df.reindex(columns=["file"] + sorted([c for c in df.columns if c != "file"]))




output_csv = os.path.join(target, "ALL_METRICS_FINAL-Compile.csv")
df.to_csv(output_csv, index=False)

print("Final CSV saved at:", output_csv)


cols_to_drop = df.columns[df.columns.str.contains(r'Blue|Sac_Blue|Sari', case=False)]
df = df.drop(columns=cols_to_drop)

output_csv = os.path.join(target, "ALL_METRICS_FINAL-Compile-short.csv")
df.to_csv(output_csv, index=False)


print("Final CSV saved at:", output_csv)


final_rows = []

for file in os.listdir(target):
    if file.endswith(".json"):
        json_path = os.path.join(target, file)
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        for col_name, metrics in data.items():
            row = {
                "file": file.replace(".json", ""),
                "column": col_name
            }
            row.update(metrics)
            final_rows.append(row)

# Convert to dataframe
df = pd.DataFrame(final_rows)

output_csv = os.path.join(target, "ALL_METRICS_FINAL-Compile-v2.csv")
df.to_csv(output_csv, index=False)

print("Final CSV saved at:", output_csv)


cols_to_drop = df.columns[df.columns.str.contains(r'Blue|Sac_Blue|Sari', case=False)]
df = df.drop(columns=cols_to_drop)

output_csv = os.path.join(target, "ALL_METRICS_FINAL-Compile-short-v2.csv")
df.to_csv(output_csv, index=False)

print("Final CSV saved at:", output_csv)
    
