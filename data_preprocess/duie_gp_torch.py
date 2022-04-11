# -*- encoding: utf-8 -*-
#
# Author: LL
#
# cython: language_level=3
#


import torch

from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
# from transformers.utils import PaddingStrategy
# from transformers import PaddingStrategy
# from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.tokenization_utils import PreTrainedTokenizer
# from my_py_toolkit.torch.transformers_pkg import bert_tokenize
from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer


ignore_list = ["offset_mapping", "text"]

@dataclass
class DataCollatorForGPLinker:
    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    num_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        print(f'collate')
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        new_features = [
            {k: v for k, v in f.items() if k not in ["labels"] + ignore_list}
            for f in features
        ]

        # batch = self.tokenizer.pad(
        #     new_features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors="pt",
        # )
        # todo: padding
        batch = {'input_ids': [], 'attention_mask':[]}
        max_len = max([len(v['input_ids']) for v in new_features])
        for v in new_features:
            batch['input_ids'].append(v['input_ids'] + [0] * (max_len - len(v['input_ids'])))
            batch['attention_mask'].append(v['attention_mask'] + [0] * (max_len - len(v['attention_mask'])))
            # v['input_ids'].extend([0] * (max_len - len(v['input_ids'])))
            # v['attention_mask'].extend([0] * (max_len - len(v['attention_mask'])))
        for k,v in batch.items():
            batch[k] = torch.tensor(v, dtype=torch.long)
        if labels is None:  # for test
            if "text" in features[0].keys():
                batch["text"] = [feature["text"] for feature in features]
            if "offset_mapping" in features[0].keys():
                batch["offset_mapping"] = [
                    feature["offset_mapping"] for feature in features
                ]
            return batch

        bs = batch["input_ids"].size(0)
        max_spo_num = max([len(lb) for lb in labels])
        batch_entity_labels = torch.zeros(bs, 2, max_spo_num, 2, dtype=torch.long)
        batch_head_labels = torch.zeros(
            bs, self.num_labels, max_spo_num, 2, dtype=torch.long
        )
        batch_tail_labels = torch.zeros(
            bs, self.num_labels, max_spo_num, 2, dtype=torch.long
        )
        for i, lb in enumerate(labels):
            for spidx, (sh, st, p, oh, ot) in enumerate(lb):
                batch_entity_labels[i, 0, spidx, :] = torch.tensor([sh, st])
                batch_entity_labels[i, 1, spidx, :] = torch.tensor([oh, ot])
                batch_head_labels[i, p, spidx, :] = torch.tensor([sh, oh])
                batch_tail_labels[i, p, spidx, :] = torch.tensor([st, ot])

        batch["labels"] = [batch_entity_labels, batch_head_labels, batch_tail_labels]
        return batch

def get_dataload(data_path, bert_cfg, num_label, batch_size, num_workers=1, shuffle=True):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    collate_fn = DataCollatorForGPLinker(tokenizer, num_labels=num_label)
    dataset = Dataset.from_file(data_path)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return dataloader
        
if __name__ == '__main__':
    p = 'F:/Study/Github/GPLinker_pytorch/data_caches/spo/spo/1.0.0/4f608ff3259ef2cd7e69d8dd58a3aa33a4f4e1fef81d944e417267137f59e237/cache-train-bert-128-resources_bert_model_bert.arrow'
    bert_cfg = './resources/bert_model/bert'
    # tokenizer = bert_tokenize(bert_cfg)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    ds = Dataset.from_file(p)
    data_collate = DataCollatorForGPLinker(tokenizer, num_labels=49)
    dl = DataLoader(ds, shuffle=True, collate_fn=data_collate, batch_size=5, num_workers=1)
    for v in dl:
        print(v)
        break