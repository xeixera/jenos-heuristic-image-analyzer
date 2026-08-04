# jenos-heuristic-image-analyzer

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-an%C3%A1lise%20heur%C3%ADstica-red)
![scikit--learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)
![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

Ferramenta de perícia digital que estima a probabilidade de uma imagem (`.jpg`, `.png`) ter sido **gerada ou manipulada por IA**, usando métodos heurísticos clássicos de análise forense (em vez de uma rede neural "caixa-preta"). O resultado vem sempre acompanhado das características que mais pesaram na decisão (explicabilidade).

## Sumário

1. [Como funciona](#1-como-funciona)
2. [Estrutura do código](#2-estrutura-do-código)
3. [Instalação](#3-instalação)
4. [Estrutura de dataset](#4-estrutura-de-dataset)
5. [Uso — ordem de execução](#5-uso--ordem-de-execução)
6. [Limitações conhecidas](#6-limitações-conhecidas)

---

## 1. Como funciona

Cada imagem passa por **27 análises heurísticas**: ELA, análise de ruído/PRNU, FFT, entropia, wavelet, CFA, ruído cromático, saturação/LAB e densidade de bordas. Esses valores alimentam um classificador supervisionado (RandomForest), treinado previamente com imagens reais e geradas por IA, que devolve:

- a probabilidade estimada da imagem ser gerada/manipulada por IA;
- as características que mais pesaram nessa decisão (relatório de evidências).

## 2. Estrutura do código

```
jenos/
├── common.py               #pré-processamento padrão (resize 1024x1024, ELA seguro)
├── extractor.py            #extrai as características heurísticas de uma imagem
├── calibrator.py           #monta baseline + dataset rotulado, por classe
├── trainer.py              #treina o classificador (real vs. IA), por classe
├── detector.py             #CLI de análise (imagem única ou lote)
├── requirements.txt
└── output/                 #guarda arquivos .joblib e .csv
```

O projeto separa duas frentes, que não se misturam:

| Frente | Arquivos | Quem usa |
|---|---|---|
| Calibração/treino | `calibrator.py`, `trainer.py` | Só quem desenvolve/valida o modelo gerando os artefatos (`.json`, `.joblib`) |
| Uso | `detector.py` | Análise do dia a dia — só carrega os artefatos prontos, não treina nada |

**`common.py`** garante que toda imagem ,no treino e no uso, passe pela mesma padronização (resize para 1024×1024 e isolamento seguro do arquivo temporário do ELA) evitando que resolução do arquivo vire um atalho de decisão no lugar de autenticidade.

**`trainer.py`** treina um classificador binário (RandomForest por padrão, `--modelo logistic` como alternativa) a partir de imagens reais e IA já rotuladas, reportando acurácia, precisão, recall, F1, AUC-ROC e matriz de confusão.

## 3. Instalação

```bash
pip install -r requirements.txt
```

## 4. Estrutura de dataset

```
dataset/
    real/
        faces/
        <custom>/
    ia/
        faces/
        <custom>/
```

Cada subpasta de `dataset/real/` define uma classe, detectada automaticamente pelo `calibrator.py`.

> **Atenção:** garanta variação de resolução nativa dentro de cada classe, sobreposta entre `real` e `ia`. Se uma classe vier toda de uma única resolução e a outra de outra, o modelo pode aprender a distinguir origem do arquivo em vez de autenticidade (desconfie de acurácia/AUC perfeitos).

## 5. Uso — ordem de execução

```
calibrator.py;  trainer.py --class <classe>;   detector.py --class <classe> --image/--dir            
```

**Passo 1 — Calibrar** (todas as classes de uma vez):
```bash
python3 calibrator.py
```
Gera `output/baseline_<classe>.json` e `output/features_<classe>.csv`.

**Passo 2 — Treinar** (uma classe por vez):
```bash
python3 trainer.py --class faces
```

Opcional: `--model {random-forest, logistic}`, `--holdout 0.25`.

Gera `output/modelo_<classe>.joblib` e `output/metricas_<classe>.json`.

**Passo 3 — Analisar**:
```bash
# imagem única
python3 detector.py --class faces --image caminho/imagem.jpg

# lote (diretório inteiro -> CSV)
python3 detector.py --class faces --dir dataset/teste/faces --output output/resultados.csv
```

## 6. Limitações conhecidas

- **ELA em PNG**: o sinal ainda é calculado, mas seu significado forense é mais fraco (a técnica pressupõe dupla compressão JPEG). O metadado `_metadadoFormatoOriginalJpeg` permite segmentar essa análise depois.
- **Diversidade de geradores de IA**: treinar com um único gerador tende a ensinar o modelo a reconhecer aquele gerador específico, não "IA em geral".
- **Seleção de classe manual**: hoje o `--classe` é informado à mão; um classificador de cena automático é uma extensão futura natural. (Sendo implementado)
- **Interface gráfica**: planejada como app desktop (uso por outros peritos) onde o terminal continua funcionando de forma independente. (Sendo implementado)
