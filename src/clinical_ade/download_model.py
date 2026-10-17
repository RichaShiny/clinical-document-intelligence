"""Download a pinned biomedical encoder from Hugging Face; no remote code."""
import argparse
from pathlib import Path
from .biomed import MODEL,REVISION
from .data import download


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());root=p.parse_args().root
    directory=root/'runtime/biomedbert';directory.mkdir(parents=True,exist_ok=True)
    for name in ['config.json','tokenizer_config.json','vocab.txt','pytorch_model.bin']:
        print(name,download(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}',directory/name),flush=True)

if __name__=='__main__':main()
