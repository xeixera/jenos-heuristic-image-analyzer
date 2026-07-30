# calibrator.py

#Gera, para cada classe, um baseline estatístico a partir das imagens reais e um dataset de features
#rotulado (real=0 / ia=1), utilizado posteriormente pelo train_classifier.py para treinar o modelo.

#Estrutura esperada:
#dataset/
#    real/
#        <classe>/*.jpg
#    ia/
#        <classe>/*.jpg

#O calibrador agora da suporte a várias classes, em vez de trabalhar apenas com um único conjunto de imagens reais.
#Agora ele filtra arquivos que não são imagens antes do processamento e faz a calibração ser mais robusta.
#Além do baseline estatístico, agora é gerado um dataset rotulado de features, para o treinamento de um classificador
#supervisionado.

import os
import json
import numpy as np
import pandas as pd

from extractor import extrairCaracteristicas
from common import ehArquivoDeImagemValido

DIRETORIO_DATASET = "dataset"
DIRETORIO_SAIDA = "output"


def listarClasses(diretorioReal: str) -> list:
    if not os.path.isdir(diretorioReal):
        return []
    return sorted([
        nome for nome in os.listdir(diretorioReal)
        if os.path.isdir(os.path.join(diretorioReal, nome))
    ])


def extrairFeaturesDeDiretorio(diretorio: str) -> list:
#Extrai as features de todas as imagens válidas presentes em um diretório, retornando uma lista de dicionários
#contendo as características extraídas e o nome de cada arquivo.
    resultados = []
    if not os.path.isdir(diretorio):
        return resultados

    for nomeArquivo in sorted(os.listdir(diretorio)):
        if not ehArquivoDeImagemValido(nomeArquivo):
            continue

        caminhoCompleto = os.path.join(diretorio, nomeArquivo)
        try:
            caracteristicas = extrairCaracteristicas(caminhoCompleto)
            caracteristicas["_arquivo"] = nomeArquivo
            resultados.append(caracteristicas)
            print(f"  OK: {nomeArquivo}")
        except Exception as erro:
            print(f"  ERRO: {nomeArquivo} -> {erro}")

    return resultados


def calcularBaseline(listaCaracteristicas: list) -> dict:
    """Baseline estatístico (media/desvio/percentis) - só para imagens reais."""
    if not listaCaracteristicas:
        return {}

    nomesNumericos = [
        chave for chave in listaCaracteristicas[0].keys()
        if not chave.startswith("_")
    ]

    baseline = {}
    for nome in nomesNumericos:
        valores = [item[nome] for item in listaCaracteristicas]
        baseline[nome] = {
            "media": float(np.mean(valores)),
            "desvioPadrao": float(np.std(valores) + 1e-8),
            "percentil95": float(np.percentile(valores, 95)),
            "percentil99": float(np.percentile(valores, 99)),
        }
    return baseline


def processarClasse(classe: str):
    print(f"\nClasse: {classe}")

    diretorioReal = os.path.join(DIRETORIO_DATASET, "real", classe)
    diretorioIA = os.path.join(DIRETORIO_DATASET, "ia", classe)

    print(f"Extraindo features de imagens REAIS ({diretorioReal})...")
    featuresReais = extrairFeaturesDeDiretorio(diretorioReal)

    print(f"Extraindo features de imagens IA ({diretorioIA})...")
    featuresIA = extrairFeaturesDeDiretorio(diretorioIA)

    if len(featuresReais) == 0:
        print(f"AVISO: nenhuma imagem real válida encontrada para a classe '{classe}'. Pulando.")
        return
    if len(featuresIA) == 0:
        print(
            f"AVISO: nenhuma imagem de IA encontrada para a classe '{classe}'. "
            f"O baseline será gerado, mas o dataset rotulado ficará incompleto "
            f"(train_classifier.py exige as duas classes)."
        )

    os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

    #baseline descritivo (só imagens reais, usando para testes)
    baseline = calcularBaseline(featuresReais)
    caminhoBaseline = os.path.join(DIRETORIO_SAIDA, f"baseline_{classe}.json")
    with open(caminhoBaseline, "w") as arquivo:
        json.dump(baseline, arquivo, indent=4)
    print(f"Baseline salvo em: {caminhoBaseline}")

    #dataset rotulado (reais+IA) para treino do classificador, ideia de versão final
    linhas = []
    for item in featuresReais:
        linha = {k: v for k, v in item.items() if not k.startswith("_")}
        linha["rotulo"] = 0  # 0 = real
        linhas.append(linha)
    for item in featuresIA:
        linha = {k: v for k, v in item.items() if not k.startswith("_")}
        linha["rotulo"] = 1  # 1 = gerado/editado por IA
        linhas.append(linha)

    dataframe = pd.DataFrame(linhas)
    caminhoDataset = os.path.join(DIRETORIO_SAIDA, f"features_{classe}.csv")
    dataframe.to_csv(caminhoDataset, index=False)
    print(f"Dataset rotulado salvo em: {caminhoDataset} ({len(dataframe)} amostras)")


def main():
    diretorioReal = os.path.join(DIRETORIO_DATASET, "real")
    classes = listarClasses(diretorioReal)

    if not classes:
        raise Exception(
            f"Nenhuma classe encontrada em '{diretorioReal}'. "
            f"Organize o dataset como dataset/real/<classe>/ e dataset/ia/<classe>/."
        )

    print(f"Classes encontradas: {classes}")
    for classe in classes:
        processarClasse(classe)

    print("\nCalibração concluída para todas as classes.")


if __name__ == "__main__":
    main()
