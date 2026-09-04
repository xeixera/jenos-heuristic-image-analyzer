#common.py

#Tem várias defs compartilhadas entre extractor.py, calibrator.py e detector.py.

#Serve para que todas as imagens passem pelo mesmo pré-processamento antes da análise, padronizando
#a resolução e outras etapas comuns, garantindo que as features extraídas permaneçam comparáveis entre diferentes
#imagens e que o baseline seja consistente.

import os
import cv2
import numpy as np
import tempfile

#Tamanho padrão (lado) para o qual toda imagem é normalizada antes da extração de características.
#1024 é o teste inicial por ser grande o suficiente para preservar textura/ruído de sensor, e pequeno o suficiente
#para manter o custo computacional baixo em lote.

TAMANHO_PADRAO = 1024

EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png"}


def ehArquivoDeImagemValido(nomeArquivo: str) -> bool:
    #filtra formatos que não são imagens (.wav, .pdf)
    _, extensao = os.path.splitext(nomeArquivo)
    return extensao.lower() in EXTENSOES_VALIDAS


def carregarImagemPadronizada(caminhoImagem: str) -> np.ndarray:
#Carrega uma imagem padronizando ela para TAMANHO_PADRAOxTAMANHO_PADRAO.

#O redimensionamento preserva a proporção da imagem e, em seguida, realiza center crop para evitar distorções e a
#introdução de bordas artificiais que poderiam afetar a extração das features.

#A leitura é feita com cv2.imread(), que já garante uma imagem BGR com três canais, dispensando tratamento
#adicional para imagens RGBA ou em escala de cinza.
    imagem = cv2.imread(caminhoImagem, cv2.IMREAD_COLOR)
    if imagem is None:
        raise Exception(f"Erro ao abrir imagem: {caminhoImagem}")

    altura, largura = imagem.shape[:2]
    ladoMenor = min(altura, largura)

    escala = TAMANHO_PADRAO / ladoMenor
    novaLargura = max(TAMANHO_PADRAO, int(round(largura * escala)))
    novaAltura = max(TAMANHO_PADRAO, int(round(altura * escala)))

    interpolacao = cv2.INTER_AREA if escala < 1.0 else cv2.INTER_CUBIC
    imagemRedimensionada = cv2.resize(
        imagem, (novaLargura, novaAltura), interpolation=interpolacao
    )

    # center-crop para TAMANHO_PADRAOxTAMANHO_PADRAO
    inicioY = (novaAltura - TAMANHO_PADRAO) // 2
    inicioX = (novaLargura - TAMANHO_PADRAO) // 2
    imagemFinal = imagemRedimensionada[
        inicioY:inicioY + TAMANHO_PADRAO,
        inicioX:inicioX + TAMANHO_PADRAO
    ]

    return imagemFinal


def formatoOriginalEhJpeg(caminhoImagem: str) -> bool:
#Indica se o arquivo original está no formato JPEG.

#Importante para interpretar o resultado do ELA, já que a técnica explora artefatos de compressão JPEG.
#Em imagens originalmente PNG ou de outros formatos, o ELA ainda pode fornecer informações úteis, mas sua interpretação
#é diferente, e por esse motivo o formato original é armazenado para preservar o contexto da análise forense.
    extensao = os.path.splitext(caminhoImagem)[1].lower()
    return extensao in {".jpg", ".jpeg"}


class ArquivoTemporarioELA:
#Context-manager responsável por criar um arquivo JPEG temporário com nome único para o cálculo do ELA.
#Esse arquivo é removido automaticamente ao final da execução, mesmo em caso de erro, evitando acúmulo de arquivos
#temporários e problemas de concorrência em execuções paralelas.

    def __init__(self, imagem: np.ndarray, qualidade: int = 90):
        self.imagem = imagem
        self.qualidade = qualidade
        self.caminho = None

    def __enter__(self) -> str:
        descritor, self.caminho = tempfile.mkstemp(suffix=".jpg", prefix="jenos_ela_")
        os.close(descritor)
        cv2.imwrite(self.caminho, self.imagem, [cv2.IMWRITE_JPEG_QUALITY, self.qualidade])
        return self.caminho

    def __exit__(self, tipoExcecao, valorExcecao, traceback):
        if self.caminho and os.path.exists(self.caminho):
            os.remove(self.caminho)
        return False
