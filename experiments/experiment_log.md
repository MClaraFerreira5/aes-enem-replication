# Diário de experimentos

Registro passo a passo de como o modelo foi aplicado ao EssayBR. Cada passo traz o que foi feito, o comando e o que se concluiu. **Novos passos entram no final.**

Legenda: ✅ concluído · ⏳ pendente

---

## Passo 1 ✅ Diagnóstico: por que o modelo não rodava com `clean_essay_br.parquet`

**Objetivo:** obter, para o EssayBR, as mesmas métricas (QWK, RMSE, discrepância > 80 p) do baseline do AES-ENEM.

**Problema:** o código só lê o formato do AES-ENEM.

| O código espera | O `clean_essay_br.parquet` tem |
|---|---|
| `essay_text` | `essay` |
| `c1_class` … `c5_class` (0–5) | `c1` … `c5` (notas 0–200, múltiplos de 40) |
| arquivos `train/val/test.parquet` | um único arquivo, sem divisão |

Também verificado: `score` = soma de `c1..c5` em 100% das 6.577 linhas.

## Passo 2 ✅ Mudanças no código

| Arquivo | Mudança |
|---|---|
| `src/data/preprocessing.py` | Novo `--dataset essay_br`: converte o formato e divide 80/10/10 (aleatório, semente 42) em `data/processed/essay_br/{train,val,test}.parquet`. |
| `src/data/dataset.py` | Novo `get_test_dataloader` (avaliação só no teste ou no dataset inteiro). |
| `src/train.py` | Linha de comando: `--train-dataset`, `--test-dataset`, `--test-split`, `--skip-training`, `--competencies` etc. Salva o melhor modelo em `checkpoints/<dataset>/cN/best.ckpt` e grava `experiments/results_<treino>_on_<teste>.md/.json`. Sem argumentos, reproduz o baseline como antes. |
| `README.md`, `data/README.md` | Reescritos com passo a passo, formato dos dados e erros comuns. |
| `.gitignore` | Ignora `checkpoints/`. |

## Passo 3 ✅ Checkpoints do AES-ENEM

Os checkpoints do baseline estavam em `lightning_logs/` (vieram do Drive). As versões 0–23 não têm `.ckpt` (só logs). As **versões 24–28** (lr = 3e-5, criadas em ordem crescente) correspondem à última execução, **C1 → C5**:

| Competência | Origem | Destino |
|---|---|---|
| C1 | `lightning_logs/version_24` | `checkpoints/aes_enem/c1/best.ckpt` |
| C2 | `version_25` | `checkpoints/aes_enem/c2/best.ckpt` |
| C3 | `version_26` | `checkpoints/aes_enem/c3/best.ckpt` |
| C4 | `version_27` | `checkpoints/aes_enem/c4/best.ckpt` |
| C5 | `version_28` | `checkpoints/aes_enem/c5/best.ckpt` |

Cada arquivo tem 1,3 GB. Foram **movidos** (`mv`), não copiados (cópia de 1,3 GB é lenta). O mapeamento por ordem de criação foi confirmado no Passo 5.

## Passo 4 ✅ Execução no Google Colab (Drive compartilhado) e dois erros

Caminho no Colab para pasta de Drive compartilhado: `%cd "/content/drive/Shareddrives/NOME_DO_DRIVE/aes-enem-replication"` (ou criar atalho em "Meu Drive" se for "Compartilhados comigo").

| Erro | Causa | Solução |
|---|---|---|
| Os dois resultados vieram do AES-ENEM, ignorando `--test-dataset essay_br`. | O Drive ainda tinha o `src/train.py` antigo (sem argumentos), que retreinou no AES-ENEM. Prova: as porcentagens de discrepância (4,78%, 52,15%, 52,63%) só são possíveis com n = 209 (10, 109 e 110 redações), nunca com as 6.577 do EssayBR. | Reenviar os arquivos novos ao Drive e reiniciar o ambiente de execução. |
| `FileNotFoundError: checkpoints/aes_enem/c1/best.ckpt` | O `mv` foi feito no computador local, não no Drive. | Mover no Colab `lightning_logs/version_24..28` → `checkpoints/aes_enem/cN/best.ckpt`. |

## Passo 5 ✅ Primeiros resultados

### 5a. Verificação: checkpoints reproduzem o baseline (AES-ENEM, 209 linhas)

```bash
python -m src.train --skip-training
```

C5 QWK 0.4796 / RMSE 52.72; total QWK 0.4925 / RMSE 191.87 / discrepância 54,07%. **Idêntico ao `baseline_results.md`** → modelo e mapeamento C1–C5 corretos.

### 5b. Modelo treinado no AES-ENEM avaliado no EssayBR inteiro (6.577 redações), **com sobreposição**

```bash
python -m src.train --train-dataset aes_enem --test-dataset essay_br --test-split all --skip-training
```

| | QWK AES-ENEM | QWK EssayBR | RMSE AES-ENEM | RMSE EssayBR | Disc. AES-ENEM | Disc. EssayBR |
|---|---|---|---|---|---|---|
| C1 | 0.3439 | 0.3400 | 42.05 | 31.64 | 6.70% | 1.49% |
| C2 | 0.2906 | 0.4170 | 53.65 | 37.51 | 4.78% | 1.35% |
| C3 | 0.3178 | 0.3887 | 51.69 | 38.52 | 5.26% | 1.89% |
| C4 | 0.2922 | 0.4046 | 44.61 | 38.94 | 6.70% | 2.68% |
| C5 | 0.4796 | 0.4801 | 52.72 | 48.11 | 5.74% | 4.64% |
| **Total** | **0.4925** | **0.5704** | **191.87** | **140.13** | **54.07%** | **42.13%** |

⚠️ Estes números do EssayBR **incluem redações que o modelo viu no treino** (Passo 6). Não usar como resultado final.

## Passo 6 ✅ Descoberta: AES-ENEM e EssayBR compartilham redações

Verificação feita comparando janelas de 8 palavras consecutivas (texto bruto do EssayBR):

- AES-ENEM sourceA: 1.165 linhas, mas só **395 redações únicas** (train 258, validação 66, teste 71). Cada redação aparece até 3 vezes, na coluna `reference`: nota oficial (`crawled_from_web`, 395 linhas) e notas de dois avaliadores (`grader_a` e `grader_b`, 385 linhas cada); em 362 redações as três notas são diferentes. Não há redação repetida entre train/val/test do AES-ENEM.
- ~93% das linhas do AES-ENEM têm mais de 50% do texto no EssayBR.
- No sentido inverso, **364 a 389 redações do EssayBR** (≈ 6%) estão no AES-ENEM; a distribuição é bem separada (≈ 6.150 com < 5% de coincidência e ≈ 370 com > 50%), então o limiar não é crítico (0,2 → 389; 0,3 → 383; 0,5 → 364).
- Consequência: o teste do baseline do AES-ENEM tem só **71 redações únicas** (resultados ruidosos), e a avaliação no EssayBR inteiro mistura redações já vistas no treino.

**Mudança no código:**
- `preprocessing.py` cria a coluna booleana `in_aes_enem` (limiar de 30% das janelas de 8 palavras presentes no AES-ENEM sourceA).
- `train.py --exclude-overlap` remove essas redações do teste e grava `experiments/results_aes_enem_on_essay_br_sem_sobreposicao.md`.

## Passo 7 ✅ Resultado limpo: EssayBR sem as redações do AES-ENEM

Reenviar ao Drive `src/train.py`, `src/data/preprocessing.py`, `src/data/dataset.py`, `README.md`; reiniciar o Colab; rodar:

```bash
python -m src.data.preprocessing --dataset essay_br
python -m src.train --train-dataset aes_enem --test-dataset essay_br --test-split all --skip-training --exclude-overlap
```

Saída obtida: **383 de 6577** redações do EssayBR também estão no AES-ENEM (com o texto limpo do parquet, igual ao previsto pelo CSV bruto) e **"Avaliando em 6194 redações (all)"**. Resultado em `experiments/results_aes_enem_on_essay_br_sem_sobreposicao.md`.

| | QWK | RMSE | Disc. > 80 p |
|---|---|---|---|
| C1 | 0.3357 | 29.69 | 0.89% |
| C2 | 0.3948 | 36.83 | 1.13% |
| C3 | 0.3578 | 37.75 | 1.70% |
| C4 | 0.4092 | 37.42 | 2.10% |
| C5 | 0.4680 | 47.72 | 4.42% |
| **Total** | **0.5655** | **133.68** | **40.98%** |

## Passo 8 ✅ Interpretação

### 8a. A sobreposição quase não mudou o resultado

| | Total QWK | Total RMSE | Total Disc. | C5 QWK |
|---|---|---|---|---|
| AES-ENEM (teste, baseline) | 0.4925 | 191.87 | 54.07% | 0.4796 |
| EssayBR inteiro (com 383 repetidas) | 0.5704 | 140.13 | 42.13% | 0.4801 |
| **EssayBR só redações novas (6.194)** | **0.5655** | **133.68** | **40.98%** | **0.4680** |

Retirar as 383 redações já vistas mexeu pouco (total QWK −0.005). A vantagem sobre o teste do AES-ENEM **não vem da sobreposição**. O RMSE caiu porque as redações retiradas tinham nota média menor (506 contra 640) e eram mais difíceis.

### 8b. O EssayBR tem notas menos dispersas que o teste do AES-ENEM

Desvio-padrão das notas reais = RMSE de um "modelo" que sempre prevê a média (referência para o RMSE):

| | C1 | C2 | C3 | C4 | C5 | Total |
|---|---|---|---|---|---|---|
| Desvio-padrão, EssayBR sem sobreposição | 30.4 | 37.2 | 36.3 | 43.5 | 51.0 | 168.7 |
| RMSE do modelo no EssayBR | 29.7 | 36.8 | 37.7 | 37.4 | 47.7 | 133.7 |
| Desvio-padrão, teste do AES-ENEM | 47.4 | 56.9 | 52.2 | 46.7 | 57.9 | 223.5 |
| RMSE do modelo no AES-ENEM | 42.1 | 53.7 | 51.7 | 44.6 | 52.7 | 191.9 |

(Média da nota total: 640 no EssayBR, 530 no teste do AES-ENEM.)

Conclusões:
- O RMSE menor no EssayBR é, em boa parte, **efeito de a escala de notas ser mais concentrada**, não de o modelo ser melhor lá. Em C1, C2 e C3 o RMSE do modelo fica praticamente igual ao desvio-padrão (em C3 é pior), ou seja, **não supera "prever sempre a média"** nessas competências. Na nota total o modelo reduz o erro em ~21% sobre essa referência no EssayBR e ~14% no AES-ENEM.
- O **QWK** (0.34–0.47 por competência; 0.57 no total) mostra que o modelo captura alguma ordenação das redações, e é a métrica mais comparável entre os dois datasets.
- O teste do AES-ENEM tem só 71 redações únicas; a diferença de QWK total entre os datasets (0.49 contra 0.57) é pequena demais para afirmar que o modelo generaliza melhor no EssayBR. Não foi calculado intervalo de confiança.
- Não verificado: se o modelo tem viés (prevê notas mais baixas, já que treinou em dados com média 530 e o EssayBR tem média 640).

---

## Limitações conhecidas

- A detecção de redações repetidas foi testada no **CSV bruto**; com o texto do `clean_essay_br.parquet` o número exato pode mudar um pouco (o próprio script imprime a contagem).
- A divisão treino/val/teste do EssayBR é aleatória (não separa por tema), caso se treine no EssayBR (`--train-dataset essay_br`).
- Não se sabe como o `clean_essay_br.parquet` foi gerado a partir de `extended_essay-br.csv` (o script de limpeza não está no repositório).
- O ambiente onde o código foi editado não tinha GPU nem `torch`: todas as execuções do modelo foram feitas pela usuária no Colab.

## Próximos passos possíveis

- Calcular intervalos de confiança (bootstrap por redação única) e o viés médio das previsões.
- Treinar no EssayBR (`--train-dataset essay_br`) e comparar com o modelo do AES-ENEM.
- Avaliar também o modelo treinado no EssayBR no AES-ENEM (`--train-dataset essay_br --test-dataset aes_enem`).
- Escrever os resultados na seção de experimentos do artigo (`docs/`).
