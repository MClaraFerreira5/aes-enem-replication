# aes-enem-replication

Replicação do baseline do artigo **"A New Benchmark for Automatic Essay Scoring in Portuguese"** (Silveira et al., PROPOR 2024) para correção automática de redações do ENEM, e extensão do experimento para o dataset **EssayBR**.

O modelo é o **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) com cabeça de **regressão ordinal (CORN)**. Um modelo é treinado para cada uma das 5 competências do ENEM (C1–C5); cada uma é prevista em 6 níveis (0, 40, 80, 120, 160, 200) e a **nota total (0–1000)** é a soma das cinco.

## Sumário

1. [Estrutura do projeto](#estrutura-do-projeto)
2. [Instalação](#instalação)
3. [Início rápido](#início-rápido)
4. [Rodando com o EssayBR](#rodando-com-o-essaybr)
5. [Opções do `src.train`](#opções-do-srctrain)
6. [Formato dos dados](#formato-dos-dados)
7. [Resultados](#resultados)
8. [Problemas comuns](#problemas-comuns)

## Estrutura do projeto

```
aes-enem-replication/
├── data/
│   ├── raw/                    # dados brutos (CSV baixados)
│   │   ├── aes_enem/
│   │   └── essay_br/extended_essay-br.csv
│   └── processed/              # dados prontos para o modelo (parquet)
│       ├── aes_enem/           # train / val / test (+ test_easy, test_hard, sourceB)
│       └── essay_br/           # clean_essay_br.parquet -> train / val / test
├── src/
│   ├── data/
│   │   ├── dowload_data.py     # baixa o AES-ENEM do Hugging Face
│   │   ├── preprocessing.py    # gera os parquets de AES-ENEM e EssayBR
│   │   └── dataset.py          # Dataset / DataLoaders (tokenização)
│   ├── models/bertimbau_module.py  # modelo (PyTorch Lightning)
│   ├── utils/metrics.py        # QWK, RMSE e discrepância > 80 pontos
│   └── train.py                # treino + avaliação (ponto de entrada)
├── experiments/                # resultados (baseline_results.md e results_*.md/json)
├── checkpoints/                # melhores modelos treinados (criado ao treinar; ignorado pelo git)
├── notebooks/01_baseline_replication.ipynb  # execução do baseline no Google Colab
└── docs/                       # artigo em LaTeX (template SBC)
```

## Instalação

Treinar o BERTimbau em CPU é impraticável; use uma **GPU** (o baseline foi rodado no Google Colab).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Todos os comandos abaixo devem ser executados **a partir da raiz do projeto** (a pasta que contém este README).

## Início rápido

Reproduz o baseline (treina e avalia no AES-ENEM):

```bash
# 1. (só uma vez) baixar o AES-ENEM -> data/raw/aes_enem/
python -m src.data.dowload_data

# 2. converter os CSVs em parquets com as colunas que o modelo espera
python -m src.data.preprocessing --dataset aes_enem

# 3. treinar as 5 competências e avaliar no conjunto de teste
python -m src.train
```

Ao final são impressas as métricas por competência e da nota total, e salvas em
`experiments/results_aes_enem_on_aes_enem.md` (e `.json`). Os melhores checkpoints ficam em
`checkpoints/aes_enem/c1/best.ckpt`, … , `c5/best.ckpt`.

No Google Colab (como em `notebooks/01_baseline_replication.ipynb`):

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/aes-enem-replication
!pip install -r requirements.txt
!PYTHONPATH=. python -m src.data.preprocessing --dataset aes_enem
!PYTHONPATH=. python -m src.train
```

## Rodando com o EssayBR

O arquivo `data/processed/essay_br/clean_essay_br.parquet` **não pode ser passado direto** ao modelo: ele tem as notas em `c1..c5` (0–200) e não tem divisão treino/validação/teste. O passo de pré-processamento abaixo converte para o formato esperado (veja [Formato dos dados](#formato-dos-dados)) e divide em 80% treino / 10% validação / 10% teste (aleatório, semente 42):

```bash
python -m src.data.preprocessing --dataset essay_br
```

Isso cria `data/processed/essay_br/{train,val,test}.parquet`. Depois escolha o experimento:

### A) Treinar e avaliar no EssayBR

```bash
python -m src.train --train-dataset essay_br
```

Resultado em `experiments/results_essay_br_on_essay_br.md`.

### B) Usar o modelo treinado no AES-ENEM e avaliar no EssayBR (sem retreinar)

Requer os checkpoints de `checkpoints/aes_enem/` (gerados por `python -m src.train`). Se você treinou no Colab, copie os `.ckpt` de `lightning_logs/.../checkpoints/` do Drive para `checkpoints/aes_enem/c1/best.ckpt`, … , `c5/best.ckpt`.

```bash
python -m src.train --train-dataset aes_enem --test-dataset essay_br --test-split all --skip-training
```

`--test-split all` avalia nas ~6,5 mil redações do EssayBR (já que o modelo nunca viu esses dados); use `--test-split test` para avaliar só no 10% de teste. Resultado em `experiments/results_aes_enem_on_essay_br.md`.

### C) Treinar no AES-ENEM e avaliar no EssayBR em uma única execução

```bash
python -m src.train --train-dataset aes_enem --test-dataset essay_br --test-split all
```

### Aviso: redações repetidas entre AES-ENEM e EssayBR

Os dois datasets compartilham redações: o AES-ENEM (sourceA) tem 1.165 linhas, mas só ~395 redações únicas (cada uma aparece até 3 vezes: a nota oficial, `crawled_from_web`, e as de dois avaliadores, `grader_a` e `grader_b`), e quase todas também estão no EssayBR. Ou seja, ~6% do EssayBR já foi visto no treino do modelo do AES-ENEM. Para avaliar só em dados realmente novos:

```bash
python -m src.data.preprocessing --dataset essay_br   # cria a coluna in_aes_enem
python -m src.train --train-dataset aes_enem --test-dataset essay_br --test-split all --skip-training --exclude-overlap
```

O resultado vai para `experiments/results_aes_enem_on_essay_br_sem_sobreposicao.md`. O pré-processamento precisa de `data/raw/aes_enem/` para detectar as repetições.

## Opções do `src.train`

| Opção | Padrão | O que faz |
|---|---|---|
| `--train-dataset` | `aes_enem` | Dataset de treino (`aes_enem` ou `essay_br`). Com `--skip-training`, de onde vêm os checkpoints. |
| `--test-dataset` | igual ao treino | Dataset de avaliação. |
| `--test-split` | `test` | `test` = só `test.parquet`; `all` = dataset inteiro. |
| `--exclude-overlap` | desligado | Remove do teste (EssayBR) as redações que também estão no AES-ENEM. Ver [aviso](#aviso-redações-repetidas-entre-aes-enem-e-essaybr). |
| `--skip-training` | desligado | Só carrega `checkpoints/<train-dataset>/cN/best.ckpt` e avalia. |
| `--competencies` | `c1 c2 c3 c4 c5` | Subconjunto de competências (a nota total só é calculada com as 5). |
| `--batch-size` | `8` | Tamanho do lote. Reduza se faltar memória da GPU. |
| `--max-epochs` | `15` | Máximo de épocas (há early stopping com `--patience`). |
| `--lr` | `3e-5` | Taxa de aprendizado. |
| `--patience` | `2` | Épocas sem melhora da `val_loss` antes de parar. |

Teste rápido de que tudo funciona (uma competência, poucas épocas):

```bash
python -m src.train --train-dataset essay_br --competencies c5 --max-epochs 1
```

## Formato dos dados

O código lê, em cada `train/val/test.parquet`, estas colunas:

| Coluna | Tipo | Descrição |
|---|---|---|
| `essay_text` | texto | Texto da redação (truncado em 512 tokens). |
| `c1_class` … `c5_class` | int 0–5 | Classe ordinal de cada competência = nota ÷ 40. Alvo do treino. |
| `c1_score` … `c5_score`, `total_score` | número | Notas originais (0–200 por competência; total 0–1000). |
| `in_aes_enem` | bool | (só EssayBR) a redação também existe no AES-ENEM. Usada por `--exclude-overlap`. |

Como o `preprocessing.py` converte o EssayBR:

| EssayBR (`clean_essay_br.parquet`) | Formato do modelo |
|---|---|
| `essay` | `essay_text` (se vier como lista de parágrafos, é unido por quebras de linha) |
| `c1` … `c5` (0–200) | `c1_score` … e `c1_class` = `c1 // 40` |
| `score` | `total_score` (soma das 5 competências; confere em 100% das linhas) |
| `prompt`, `title` | mantidas, sem uso no treino |

Para usar **outro dataset** no futuro: gere `train/val/test.parquet` com as colunas acima em `data/processed/<nome>/` e registre `<nome>` no dicionário `DATASETS` em `src/train.py`.

## Resultados

Baseline (AES-ENEM) comparado ao artigo original, em `experiments/baseline_results.md`:

| Experimento | C5 QWK | C5 RMSE | Total QWK | Total RMSE | Total Discrepância (>80p) |
|---|---|---|---|---|---|
| Resultado atual | 0.4796 | 52.71 | 0.4925 | 191.87 | 54.07% |
| Artigo original (Silveira et al.) | 0.4820 | 51.10 | 0.5640 | 172.50 | 48.30% |
| Diferença | −0.0024 | +1.61 | −0.0715 | +19.37 | +5.77 p.p. |

Métricas: **QWK** (Quadratic Weighted Kappa, maior é melhor), **RMSE** (menor é melhor) e **Discrepância** (% de redações em que a nota prevista difere da humana em mais de 80 pontos; menor é melhor).

Cada execução de `src.train` grava seus próprios resultados em `experiments/results_<treino>_on_<teste>.md`.

O passo a passo dos experimentos (decisões, erros encontrados e resultados) está em [`experiments/experiment_log.md`](experiments/experiment_log.md).

## Problemas comuns

- **`ModuleNotFoundError: No module named 'src'`** — rode da raiz do projeto e use `python -m src.train` (não `python src/train.py`). No Colab, prefixe com `PYTHONPATH=.`.
- **`FileNotFoundError: ... train.parquet`** — falta o pré-processamento (`python -m src.data.preprocessing --dataset ...`).
- **`FileNotFoundError: Checkpoint não encontrado`** — com `--skip-training` os checkpoints precisam existir em `checkpoints/<train-dataset>/cN/best.ckpt`; rode sem `--skip-training` ou copie-os.
- **`CUDA out of memory`** — use `--batch-size 4` (ou menor).
- **`ImportError` ao ler parquet** — instale o `pyarrow` (`pip install -r requirements.txt`).

## Referência

Silveira, I. C.; Barbosa, A.; Mauá, D. D. *A New Benchmark for Automatic Essay Scoring in Portuguese.* PROPOR 2024. https://aclanthology.org/2024.propor-1.23
