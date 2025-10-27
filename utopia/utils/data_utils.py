
from io import BytesIO
import json
import os
from typing import Tuple, Union, Dict, Any, List, Optional
import base64
import re
import numpy as np

from datasets import load_dataset, DatasetDict
from PIL import Image

def read_jsonl(f):
    return [json.loads(x) for x in open(f).readlines()]

def read_json(f):
    return json.load(open(f, 'r'))

def write_jsonl(data: list, f):
    with open(f, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

def write_json(data: dict, f):
    with open(f, 'w') as file:
        json.dump(data, file, indent=2)

def load_data(debug: bool = False):
    from langchain_core.documents import Document
    
    
def top_percentile(value, data):
    data = np.array(data)
    percentile = np.sum(data >= value) / len(data) * 100
    return percentile