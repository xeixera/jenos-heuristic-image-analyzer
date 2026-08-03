#train_classifier.py

#To passando uma fome grande aqui por que a primeira vez mexendo com isso

#Treina um classificador binário (real=0 / ia=1) para cada classe a partir do dataset
#rotulado gerado pelo calibrator.py, (output/features_<classe>.csv).

#No protótipo que eu fiz apenas para testar o funcionamento, cada feature da imagem de teste era comparada à
#média das imagens REAIS por meio do Zscore. As distâncias obtidas eram ponderadas por pesos definidos manualmente
#e somadas para produzir uma pontuação final.

#Isso gerava um problema de detecção de anomalias de uma única classe. Com qualquer imagem estatisticamente
#distante da distribuição das imagens reais tendendo a receber uma alta pontuação de "provável IA", incluindo imagens
#reais que passaram por edições intensas, HDR, digitalização, denoising agressivo ou outros processos que alterem
#significativamente suas características.

#Então optei por um classificador supervisionado aprendendo automaticamente a importância de cada feature, lidando
#melhor com características correlacionadas e produzindo classificações mais robustas.
#O Random Forest é utilizado como modelo padrão, mantendo a regressão logística como alternativa.

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

DIRETORIO_SAIDA = "output"


def carregarDataset(classe: str) -> pd.DataFrame:
    caminho = os.path.join(DIRETORIO_SAIDA, f"features_{classe}.csv")
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Dataset não encontrado: {caminho}. Rode calibrator.py primeiro."
        )
    return pd.read_csv(caminho)


def construirModelo(tipoModelo: str):
    if tipoModelo == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
    # default: random forest
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def treinarClasse(classe: str, tipoModelo: str = "random_forest", testeProporcao: float = 0.25):
    print(f"\nTreinando classificador para: {classe} ")

    dataFrame = carregarDataset(classe)

    if "rotulo" not in dataFrame.columns:
        raise Exception(f"Dataset de '{classe}' não tem coluna 'rotulo'.")

    contagem = dataFrame["rotulo"].value_counts().to_dict()
    print(f"Amostras -> real: {contagem.get(0, 0)} | ia: {contagem.get(1, 0)}")

    if contagem.get(0, 0) < 10 or contagem.get(1, 0) < 10:
        print(
            "AVISO: menos de 10 amostras em alguma das classes. O resultado "
            "do treino/avaliação abaixo NÃO é estatisticamente confiável - "
            "trate como teste de fumaça do pipeline, não como validação real."
        )

    colunasFeature = [nomeColuna for nomeColuna in dataFrame.columns if nomeColuna not in ("rotulo", "_arquivo")]
    X = dataFrame[colunasFeature].values
    y = dataFrame["rotulo"].values

    XTreino, XTeste, yTreino, yTeste = train_test_split(
        X, y, test_size=testeProporcao, stratify=y, random_state=42
    )

    modelo = construirModelo(tipoModelo)
    modelo.fit(XTreino, yTreino)

    yPred = modelo.predict(XTeste)
    yProba = modelo.predict_proba(XTeste)[:, 1]

    metricas = {
        "acuracia": float(accuracy_score(yTeste, yPred)),
        "precisao": float(precision_score(yTeste, yPred, zero_division=0)),
        "recall": float(recall_score(yTeste, yPred, zero_division=0)),
        "f1": float(f1_score(yTeste, yPred, zero_division=0)),
        "matrizConfusao": confusion_matrix(yTeste, yPred).tolist(),
    }
    #AUC-ROC exige as duas classes presentes no teste (em testes)
    if len(set(yTeste)) == 2:
        metricas["aucRoc"] = float(roc_auc_score(yTeste, yProba))

    print(json.dumps(metricas, indent=4))
    print(classification_report(yTeste, yPred, target_names=["real", "ia"], zero_division=0))

    #validação cruzada (mais robusta que um único split, pra testes com dataset pequeno)
    try:
        scoresCV = cross_val_score(modelo, X, y, cv=5, scoring="roc_auc")
        print(f"AUC-ROC (5-fold CV): {scoresCV.mean():.3f} +/- {scoresCV.std():.3f}")
    except Exception as erro:
        print(f"Validação cruzada não pôde ser executada: {erro}")

    #feature-importance (RandomForest) para manter logica forense
    importancias = None
    if hasattr(modelo, "feature_importances_"):
        importancias = sorted(
            zip(colunasFeature, modelo.feature_importances_.tolist()),
            key=lambda parImportancia: parImportancia[1], reverse=True,
        )
        print("\nFeatures mais relevantes para esta classe:")
        for nome, valor in importancias[:10]:
            print(f"  {nome}: {valor:.4f}")

    #salvar modelo + metadados
    os.makedirs(DIRETORIO_SAIDA, exist_ok=True)
    caminhoModelo = os.path.join(DIRETORIO_SAIDA, f"modelo_{classe}.joblib")
    joblib.dump({
        "modelo": modelo,
        "colunasFeature": colunasFeature,
        "tipoModelo": tipoModelo,
    }, caminhoModelo)
    print(f"\nModelo salvo em: {caminhoModelo}")

    caminhoMetricas = os.path.join(DIRETORIO_SAIDA, f"metricas_{classe}.json")
    with open(caminhoMetricas, "w") as arquivo:
        json.dump({
            "metricas": metricas,
            "importanciaFeatures": importancias,
        }, arquivo, indent=4)
    print(f"Métricas salvas em: {caminhoMetricas}")


def main():
    parser = argparse.ArgumentParser(description="Treina o classificador Jenos por classe.")
    parser.add_argument("--classe", required=True, help="Nome da classe (ex: faces, paisagens)")
    parser.add_argument("--modelo", default="random_forest", choices=["random_forest", "logistic"])
    parser.add_argument("--teste-proporcao", type=float, default=0.25)
    args = parser.parse_args()

    treinarClasse(args.classe, tipoModelo=args.modelo, testeProporcao=args.teste_proporcao)


if __name__ == "__main__":
    main()