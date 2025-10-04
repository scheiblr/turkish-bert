#!/usr/bin/env python

from transformers import AutoModel
import pandas as pd


models = {
    # SindBERT
    "sindbert_base": "SindBERT/sindbert_base",
    "sindbert_large": "SindBERT/sindbert_large",

    # Turkish BERT zoo (cased only)
    "BERTurk_32k": "dbmdz/bert-base-turkish-cased",
    "BERTurk_128k": "dbmdz/bert-base-turkish-128k-cased",
    "DistilBERTurk": "dbmdz/distilbert-base-turkish-cased",
    "ConvBERTurk": "dbmdz/convbert-base-turkish-cased",
    "ConvBERTurk_mC4": "dbmdz/convbert-base-turkish-mc4-cased",
    "ELECTRA_small": "dbmdz/electra-small-turkish-cased-discriminator",
    "ELECTRA_base": "dbmdz/electra-base-turkish-cased-discriminator",
    "ELECTRA_base_mC4": "dbmdz/electra-base-turkish-mc4-cased-discriminator",
    "RoBERTurk": "Nuri-Tas/roberturk-base",

    # Vergleichsmodelle (multilingual)
    "XLM_RoBERTa_base": "FacebookAI/xlm-roberta-base",
    "XLM_RoBERTa_large": "FacebookAI/xlm-roberta-large",
    "mmBERT_small": "jhu-clsp/mmBERT-small",
    "mmBERT_base": "jhu-clsp/mmBERT-base",
    "EuroBERT_210M": "EuroBERT/EuroBERT-210m",
    "EuroBERT 610M": "EuroBERT/EuroBERT-610m",
}


data = []

for name, path in models.items():
    try:
        model = AutoModel.from_pretrained(path, trust_remote_code=True)
        params = model.num_parameters()
        vocab = model.config.vocab_size
        
        data.append({
            "model": name,
            "source": path,
            "vocab_size": vocab,
            "#params": params
        })

        print(f"✅ Success: {name}")
        print(f"   • Parameters: {params:,}")
        print(f"   • Vocabulary size: {vocab}\n")
    except Exception as e:
        print(f"Error loading {name} from {path}: {e}")
        data.append({
            "model": name,
            "source": path,
            "vocab_size": None,
            "#params": None,
            "error": str(e)
        })

df_results = pd.DataFrame.from_dict(data)
df_results.to_csv('model_props.csv', index=False)
