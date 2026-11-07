# Dados

Esta pasta não é versionada (veja `.gitignore`); só este README é mantido.

```
data/
├── raw/         # dados originais, sem alteração
│   ├── aes_enem/      -> python -m src.data.dowload_data  (CSVs do Hugging Face: kamel-usp/aes_enem_dataset)
│   └── essay_br/      -> extended_essay-br.csv  (colunas: prompt, title, essay, c1..c5, score)
└── processed/   # dados prontos para o modelo
    ├── aes_enem/      -> python -m src.data.preprocessing --dataset aes_enem   (train/val/test.parquet)
    └── essay_br/      -> clean_essay_br.parquet (versão limpa do EssayBR)
                          python -m src.data.preprocessing --dataset essay_br   (train/val/test.parquet)
```

Os parquets finais (`train/val/test`) têm as colunas `essay_text`, `c1_class`…`c5_class` (0–5), `c1_score`…`c5_score` e `total_score`. Detalhes no README principal, seção "Formato dos dados".
