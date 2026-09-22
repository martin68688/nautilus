import os,shlex,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
def configure():
    for line in Path('/etc/autoresearch/apizh-terra-low.env').read_text().splitlines():
        line=line.strip().removeprefix('export ');k,sep,v=line.partition('=')
        if sep: os.environ[k]=shlex.split(v)[0] if shlex.split(v) else ''
    assert os.environ['AUTORESEARCH_API_BASE'].rstrip('/')=='https://apizh-ai.com/v1';assert os.environ['OPENAI_MODEL']=='gpt-5.6-terra';assert os.environ['OPENAI_REASONING_EFFORT']=='low'
    os.environ.setdefault('AUTORESEARCH_BUDGET_SECONDS','7200')
