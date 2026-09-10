import subprocess

# "pip install torch==2.0.1",
commands = [
"pip install --upgrade pip",
"pip install langchain==0.3.20",
"pip install langchain-community==0.3.19",
"pip install langchainhub",
"pip install chromadb",
"pip install bs4",
"pip install pypdf",
"pip install langchain-huggingface",
"pip install langchain-huggingface==1.0.0", 
"pip install ragatouille",
"pip install openpyxl",
"pip install lxml",
"pip install pandas",
"pip install --upgrade transformers==4.44.2",
"pip install --upgrade peft==0.11.1", 
"pip install --upgrade sentence-transformers", 
"pip install accelerate",
"pip uninstall -y apex",
"pip install FlagEmbedding==1.3.2",
"pip install networkx" , 
"pip install flair", 
"pip install IProgress", 
"pip install keybert", 


]


for cmd in commands:
    subprocess.run(cmd, shell=True)

