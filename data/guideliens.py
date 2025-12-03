# ---- 5) 读取 HuggingFace 数据集（线上仓库或本地路径） ----
from datasets import load_dataset, Dataset, DatasetDict
from typing import Optional, Union

def load_hf_dataset(dataset_path_or_name: str,
                    split: Optional[str] = None,
                    name: Optional[str] = None,
                    cache_dir: Optional[str] = None,
                    streaming: bool = False,
                    **kwargs) -> Dataset | DatasetDict:
    """
    统一入口：
      - dataset_path_or_name: 例如 'allenai/riddlesense' 或 本地路径 './my_ds'
      - split: 'train' / 'validation' / 'test' / None（返回 Dataset 或 DatasetDict）
      - name: 可选的子配置名（有的仓库需要）
      - cache_dir: 指定缓存目录
      - streaming: 是否以流式方式读取（大数据时可为 True）
      - kwargs: 其余参数传给 datasets.load_dataset

    返回:
      - 指定 split 时返回 Dataset
      - 否则返回 DatasetDict（包含各 split）
    """
    ds = load_dataset(path=dataset_path_or_name,
                      name=name,
                      split=split,
                      cache_dir=cache_dir,
                      streaming=streaming,
                      **kwargs)
    return ds

ds = load_hf_dataset("epfl-llm/guidelines")
import pdb;pdb.set_trace()
print(ds.keys())
from collections import defaultdict
guidelines, _guidelines = [], []
source_dict = defaultdict(dict)
for entry in ds['train']:
    temp = {
        'id': entry['id'],
        'text': entry['clean_text'],
        'title': entry['title'],
        'overview': entry['overview'],
    }

    if entry['title'] != 'None' or entry['overview'] != 'None':
        guidelines.append(temp)
    else:
        _guidelines.append(temp)

    source = entry['source']
    if entry['title'] != 'None':
        source_dict[entry['source']][entry['title']] = {         
        'id': entry['id'],
        'text': entry['clean_text'],
        # 'title': entry['title'],
        'overview': entry['overview']}
    elif entry['title'] == 'None' and entry['overview'] != 'None':
                source_dict[entry['source']][entry['overview']] = {         
        'id': entry['id'],
        'text': entry['clean_text'],
        # 'title': entry['title'],
        'overview': entry['overview']}
    # else:
         
    
import json
import pdb;pdb.set_trace()
with open('guideline_ready.json', 'w') as f:
    json.dump(guidelines, f, indent=4)
with open('guideline_incomp.json', 'w') as f:
    json.dump(_guidelines, f, indent=4)
with open('guideline_dict.json', 'w') as f:
    json.dump(source_dict, f, indent=4)