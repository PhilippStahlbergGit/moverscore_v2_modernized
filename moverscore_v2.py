"""
Vendored dependency: moverscore_v2.py
Original source: https://github.com/AIPHES/emnlp19-moverscore?tab=readme-ov-file

Vendoring rationale:
- The upstream implementation is unmaintained and assumes CUDA + older library APIs.
- This project requires a CPU-compatible, reproducible implementation under a modern Python stack.

Local patches in this repo (HyTE_Stahlberg_):
   - Removed hardcoded 'cuda:0' device selection.
   - Ensured all tensors/models are moved to torch.device("cpu").
   - Replaced deprecated tokenizer.max_len usage with tokenizer.model_max_length (with a safe fallback).
   - Replaced model/tokenizer loading to avoid import-time CUDA failures (kept same base model: distilbert-base-uncased).
   - Updated get_bert_embedding to use outputs.last_hidden_state directly (no torch.stack on non-tensor outputs).
   - Replaced deprecated aliases (np.float/np.int/np.bool/np.object) with supported dtypes (e.g., np.float64).

Notes:
- These patches are intended to preserve the metric definition while making the code runnable and reproducible.
- For exact dependency versions used in experiments, see this repo’s environment/requirements files.

Vendored/edited by: Philipp Stahlberg
Date: 08.03.2026
"""
from __future__ import absolute_import, division, print_function
# 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import string
from pyemd import emd, emd_with_flow
from torch import nn
from math import log
from itertools import chain

from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial


from transformers import *


model_name = 'distilbert-base-uncased'

device = torch.device("cpu")

config = DistilBertConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
tokenizer = DistilBertTokenizer.from_pretrained(model_name, do_lower_case=True)
model = DistilBertModel.from_pretrained(model_name, config=config)
model.eval()
model.to(device) 
                
def truncate(tokens):
    max_len = getattr(tokenizer, "model_max_length", 512)
    # Set fallback
    if max_len is None or max_len > 100000:
        max_len = 512

    if len(tokens) > max_len - 2:
        tokens = tokens[: (max_len - 2)]
    return tokens

def process(a):
    a = ["[CLS]"]+truncate(tokenizer.tokenize(a))+["[SEP]"]
    a = tokenizer.convert_tokens_to_ids(a)
    return set(a)


def get_idf_dict(arr, nthreads=4):
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda : log((num_docs+1)/(1)))
    idf_dict.update({idx:log((num_docs+1)/(c+1)) for (idx, c) in idf_count.items()})
    return idf_dict

def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, :lens[i]] = 1
    return padded, lens, mask

def bert_encode(model, x, attention_mask):
    model.eval()
    with torch.no_grad():
        output, x_encoded_layers, _ = model(input_ids = x, attention_mask = attention_mask)
    return x_encoded_layers

#with open('stopwords.txt', 'r', encoding='utf-8') as f:
#    stop_words = set(f.read().strip().split(' '))
         
def collate_idf(arr, tokenize, numericalize, idf_dict,
                pad="[PAD]", device=torch.device("cpu")):
    
    tokens = [["[CLS]"]+truncate(tokenize(a))+["[SEP]"] for a in arr]  
    arr = [numericalize(a) for a in tokens]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]
    
    pad_token = numericalize([pad])[0]

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, pad_token, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask, tokens

def get_bert_embedding(all_sens, model, tokenizer, idf_dict,
                       batch_size=-1, device=torch.device("cpu")):

    padded_sens, padded_idf, lens, mask, tokens = collate_idf(
        all_sens,
        tokenizer.tokenize,
        tokenizer.convert_tokens_to_ids,
        idf_dict,
        device=device,
    )

    if batch_size == -1:
        batch_size = len(all_sens)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(all_sens), batch_size):
            outputs = model(padded_sens[i:i + batch_size], attention_mask=mask[i:i + batch_size])
            batch_embedding = outputs.last_hidden_state  # (batch, seq_len, hidden)

            # Enforce 3D shape always
            if batch_embedding.dim() == 2:
                batch_embedding = batch_embedding.unsqueeze(0)

            embeddings.append(batch_embedding)

    total_embedding = torch.cat(embeddings, dim=0)  # concat batches on batch dimension
    return total_embedding, lens, mask, padded_idf, tokens

def _safe_divide(numerator, denominator):
    return numerator / (denominator + 1e-30)

def batched_cdist_l2(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-30).sqrt_()
    return res

def word_mover_score(refs, hyps, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords = True, batch_size=256, device=torch.device("cpu")):
    preds = []
    for batch_start in range(0, len(refs), batch_size):
        batch_refs = refs[batch_start:batch_start+batch_size]
        batch_hyps = hyps[batch_start:batch_start+batch_size]
        
        ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = get_bert_embedding(batch_refs, model, tokenizer, idf_dict_ref,
                                       device=device)
        hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = get_bert_embedding(batch_hyps, model, tokenizer, idf_dict_hyp,
                                       device=device)
        if ref_embedding.dim() == 2:
            ref_embedding = ref_embedding.unsqueeze(0)
        if hyp_embedding.dim() == 2:
            hyp_embedding = hyp_embedding.unsqueeze(0)


        batch_size = len(ref_tokens)
        for i in range(batch_size):  
            ref_ids = [k for k, w in enumerate(ref_tokens[i]) 
                                if w in stop_words or '##' in w 
                                or w in set(string.punctuation)]
            hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) 
                                if w in stop_words or '##' in w
                                or w in set(string.punctuation)]
          
            ref_embedding[i, ref_ids,:] = 0                        
            hyp_embedding[i, hyp_ids,:] = 0
            
            ref_idf[i, ref_ids] = 0
            hyp_idf[i, hyp_ids] = 0
            
        raw = torch.cat([ref_embedding, hyp_embedding], 1)
                             
        raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-30) 
        
        distance_matrix = batched_cdist_l2(raw, raw).double().cpu().numpy()
                
        for i in range(batch_size):  
            c1 = np.zeros(raw.shape[1], dtype=np.float64)
            c2 = np.zeros(raw.shape[1], dtype=np.float64)
            c1[:len(ref_idf[i])] = ref_idf[i]
            c2[len(ref_idf[i]):] = hyp_idf[i]
            
            c1 = _safe_divide(c1, np.sum(c1))
            c2 = _safe_divide(c2, np.sum(c2))
            
            dst = distance_matrix[i]
            _, flow = emd_with_flow(c1, c2, dst)
            flow = np.array(flow, dtype=np.float32)
            score = 1 - np.sum(flow * dst)
            preds.append(score)

    return preds

import matplotlib.pyplot as plt

def plot_example(is_flow, reference, translation, device=torch.device("cpu")):
    
    idf_dict_ref = defaultdict(lambda: 1.) 
    idf_dict_hyp = defaultdict(lambda: 1.)
    
    ref_embedding, ref_lens, ref_masks, ref_idf, ref_tokens = get_bert_embedding([reference], model, tokenizer, idf_dict_ref,
                                       device=device)
    hyp_embedding, hyp_lens, hyp_masks, hyp_idf, hyp_tokens = get_bert_embedding([translation], model, tokenizer, idf_dict_hyp,
                                       device=device)
   
    ref_embedding = ref_embedding[-1]
    hyp_embedding = hyp_embedding[-1]
               
    raw = torch.cat([ref_embedding, hyp_embedding], 1)            
    raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 1e-30) 
    
    distance_matrix = batched_cdist_l2(raw, raw)
    masks = torch.cat([ref_masks, hyp_masks], 1)        
    masks = torch.einsum('bi,bj->bij', (masks, masks))
    distance_matrix = masks * distance_matrix              

    
    i = 0
    c1 = np.zeros(raw.shape[1], dtype=np.float64)
    c2 = np.zeros(raw.shape[1], dtype=np.float64)
    c1[:len(ref_idf[i])] = ref_idf[i]
    c2[len(ref_idf[i]):] = hyp_idf[i]
    
    c1 = _safe_divide(c1, np.sum(c1))
    c2 = _safe_divide(c2, np.sum(c2))
    
    dst = distance_matrix[i].double().cpu().numpy()

    if is_flow:        
        _, flow = emd_with_flow(c1, c2, dst)
        new_flow = np.array(flow, dtype=np.float32)    
        res = new_flow[:len(ref_tokens[i]), len(ref_idf[i]): (len(ref_idf[i])+len(hyp_tokens[i]))]
    else:    
        res = 1 - dst[:len(ref_tokens[i]), len(ref_idf[i]): (len(ref_idf[i])+len(hyp_tokens[i]))]

    r_tokens = ref_tokens[i]
    h_tokens = hyp_tokens[i]
    
    fig, ax = plt.subplots(figsize=(len(r_tokens)*0.8, len(h_tokens)*0.8))
    im = ax.imshow(res, cmap='Blues')
    
    ax.set_xticks(np.arange(len(h_tokens)))
    ax.set_yticks(np.arange(len(r_tokens)))
  
    ax.set_xticklabels(h_tokens, fontsize=10)
    ax.set_yticklabels(r_tokens, fontsize=10)
    plt.xlabel("System Translation", fontsize=14)
    plt.ylabel("Human Reference", fontsize=14)
    plt.title("Flow Matrix", fontsize=14)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
#    for i in range(len(r_tokens)):
#        for j in range(len(h_tokens)):
#            text = ax.text(j, i, '{:.2f}'.format(res[i, j].item()),
#                           ha="center", va="center", color="k" if res[i, j].item() < 0.6 else "w")    
    fig.tight_layout()
    plt.show()

