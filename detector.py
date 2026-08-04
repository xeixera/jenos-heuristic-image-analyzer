#detector.py

#Esse código recebe uma imagem (ou um diretório de imagens, no modo batch) e estima a probabilidade de ela ter
#sido gerada ou editada por IA, utilizando o classificador treinado por trainer.py para a classe indicada.

#Mudanças em relação ao protótipo

#O detector passou a usar cli baseada em argparse para processamento em lote e a avaliação do modelo
#em conjuntos de teste independentes.

#A decisão não é mais baseada em um somatório de z-scores com pesos manuais. Agora é utilizado o modelo
#supervisionado treinado para cada classe, mantendo um relatório de evidências fundamentado tanto na
#importância das features quanto no desvio em relação ao baseline.

#Também foi adicionado suporte ao modo batch com exportação dos resultados em CSV, facilitando a validação
#e análise de desempenho em conjuntos de teste separados.

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from extractor import extrairCaracteristicas
from common import ehArquivoDeImagemValido

DIRETORIO_SAIDA = "output"


def carregarModelo(classe: str):
    caminhoModelo = os.path.join(DIRETORIO_SAIDA, f"modelo_{classe}.joblib")
    if not os.path.exists(caminhoModelo):
        raise FileNotFoundError(
            f"Modelo não encontrado: {caminhoModelo}. Rode trainer.py --classe {classe} primeiro."
        )
    return joblib.load(caminhoModelo)


def carregarBaseline(classe: str) -> dict:
    caminhoBaseline = os.path.join(DIRETORIO_SAIDA, f"baseline_{classe}.json")
    if not os.path.exists(caminhoBaseline):
        return {}
    with open(caminhoBaseline, "r") as arquivo:
        return json.load(arquivo)


def gerarEvidencias(caracteristicas: dict, baseline: dict, modelo, colunasFeature: list, top_n: int = 5) -> list:
#Gera uma lista de evidências legíveis, combinando o quanto a feature se desvia do baseline de imagens reais (z-score)
#e a importância dessa feature para o modelo treinado (quando disponível)

    importancias = {}
    if hasattr(modelo, "feature_importances_"):
        importancias = dict(zip(colunasFeature, modelo.feature_importances_))

    evidencias = []
    for nomeFeature in colunasFeature:
        if nomeFeature not in baseline or nomeFeature not in caracteristicas:
            continue
        media = baseline[nomeFeature]["media"]
        desvio = baseline[nomeFeature]["desvioPadrao"]
        if desvio == 0:
            continue

        zScore = abs((caracteristicas[nomeFeature] - media) / desvio)
        pesoImportancia = importancias.get(nomeFeature, 0.0)

        #score combinado: desvio estatístico ponderado pela relevância da feature no modelo
        scoreEvidencia = zScore * (pesoImportancia if importancias else 1.0)

        if zScore > 2:
            evidencias.append({
                "feature": nomeFeature,
                "zScore": round(float(zScore), 2),
                "importanciaModelo": round(float(pesoImportancia), 4) if importancias else None,
                "scoreEvidencia": round(float(scoreEvidencia), 4),
            })

    evidencias.sort(key=lambda evidencia: evidencia["scoreEvidencia"], reverse=True)
    return evidencias[:top_n]


def analisarImagem(caminhoImagem: str, classe: str, modeloCarregado=None, baseline=None) -> dict:
    if modeloCarregado is None:
        modeloCarregado = carregarModelo(classe)
    if baseline is None:
        baseline = carregarBaseline(classe)

    modelo = modeloCarregado["modelo"]
    colunasFeature = modeloCarregado["colunasFeature"]

    caracteristicas = extrairCaracteristicas(caminhoImagem)
    caracteristicasNumericas = {chave: valor for chave, valor in caracteristicas.items() if not chave.startswith("_")}

    vetor = np.array([[caracteristicasNumericas.get(nomeFeature, 0.0) for nomeFeature in colunasFeature]])
    probabilidadeIA = float(modelo.predict_proba(vetor)[0, 1])

    evidencias = gerarEvidencias(caracteristicasNumericas, baseline, modelo, colunasFeature)

    return {
        "arquivo": os.path.basename(caminhoImagem),
        "classe": classe,
        "probabilidadeIA": round(probabilidadeIA, 4),
        "formatoOriginalJpeg": caracteristicas.get("_metadadoFormatoOriginalJpeg"),
        "evidencias": evidencias,
    }


def modoUnico(caminhoImagem: str, classe: str):
    resultado = analisarImagem(caminhoImagem, classe)
    print(json.dumps(resultado, indent=4, ensure_ascii=False))
    print("-" * 40)
    print(f"Probabilidade estimada de IA: {resultado['probabilidadeIA'] * 100:.2f}%")
    if not resultado["formatoOriginalJpeg"]:
        print(
            "Nota: imagem original não é JPEG - a análise de ELA tem "
            "confiabilidade reduzida para este arquivo (ver documentação do método)."
        )
    if resultado["evidencias"]:
        print("\nPrincipais evidências:")
        for evidencia in resultado["evidencias"]:
            print(f"  {evidencia['feature']}: z={evidencia['zScore']} (importância no modelo: {evidencia['importanciaModelo']})")
    else:
        print("\nNenhuma evidência estatística relevante (z > 2) encontrada.")


def modoBatch(diretorio: str, classe: str, caminhoSaida: str):
    modeloCarregado = carregarModelo(classe)
    baseline = carregarBaseline(classe)

    linhas = []
    for nomeArquivo in sorted(os.listdir(diretorio)):
        if not ehArquivoDeImagemValido(nomeArquivo):
            continue
        caminhoCompleto = os.path.join(diretorio, nomeArquivo)
        try:
            resultado = analisarImagem(caminhoCompleto, classe, modeloCarregado, baseline)
            linhas.append({
                "arquivo": resultado["arquivo"],
                "probabilidadeIA": resultado["probabilidadeIA"],
                "formatoOriginalJpeg": resultado["formatoOriginalJpeg"],
            })
            print(f"OK: {nomeArquivo} -> {resultado['probabilidadeIA'] * 100:.2f}%")
        except Exception as erro:
            print(f"ERRO: {nomeArquivo} -> {erro}")

    dataFrame = pd.DataFrame(linhas)
    dataFrame.to_csv(caminhoSaida, index=False)
    print(f"\nResultados salvos em: {caminhoSaida}")


def main():
    parser = argparse.ArgumentParser(description="Jenos - detecção heurística de imagens geradas/editadas por IA.")
    parser.add_argument("--classe", required=True, help="Classe do modelo a usar (ex: faces, paisagens)")

    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--image", help="Caminho de uma única imagem")
    grupo.add_argument("--dir", help="Diretório com várias imagens (modo batch)")

    parser.add_argument("--output", default="output/resultados_batch.csv", help="CSV de saída no modo batch")
    args = parser.parse_args()

    if args.imagem:
        modoUnico(args.imagem, args.classe)
    else:
        modoBatch(args.dir, args.classe, args.saida)


if __name__ == "__main__":
    main()