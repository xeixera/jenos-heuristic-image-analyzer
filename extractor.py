#extractor.py

#Esse código é o responsável por extrair as características heurísticas de uma imagem
#usadas tanto para calibrar o baseline (calibrator.py) quanto para classificar uma imagem nova (detector.py).

#Melhorias baseadas no primeiro protótipo que não faziam sentido ou tinhama alguns bugs: 

#Toda imagem passa por carregarImagemPadronizada() (def contida em common.py) antes da extração,
#garantindo que features sensíveis a resolução (perfil radial a FFT, densidade de bordas, energia wavelet)
#sejam comparáveis entre imagens de tamanhos/origens diferentes.

#Arquivo temporário do ELA agora é único por chamada e sempre removido (ArquivoTemporarioELA em common.py),
#eliminando o risco de colisão em execuções paralelas ou lixo acumulado em disco não tratado nos testes iniciais.

#O Kernel do ruído cromático passou a ser proporcional ao tamanho da imagem, como o kernel principal
#(antes eu tinha colocado um valor fixo (3,3) para testes iniciais que era inconsistente com o restante do pipeline).

#Agora o código guarda o formato original do arquivo (JPEG, PNG e etc) junto ao resultado, por que isso termina
#afetando a interpretação forense do ELA dependendo do formato.

import json
import numpy as np
import cv2
import pywt
from scipy.stats import kurtosis, skew
from scipy.signal import find_peaks
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.measure import shannon_entropy

from common import (
    carregarImagemPadronizada,
    formatoOriginalEhJpeg,
    ArquivoTemporarioELA,
)


def calcularPerfilRadial(matrizDados):
    centro = np.array(matrizDados.shape) // 2
    y, x = np.indices(matrizDados.shape)

    raio = np.sqrt((x - centro[1]) ** 2 + (y - centro[0]) ** 2)
    raio = raio.astype(np.int32)

    somaPorRaio = np.bincount(raio.ravel(), matrizDados.ravel())
    contagemPorRaio = np.bincount(raio.ravel())
    perfilRadial = somaPorRaio / (contagemPorRaio + 1e-8)
    return perfilRadial


def extrairCaracteristicas(caminhoImagem: str) -> dict:
    imagemOriginal = carregarImagemPadronizada(caminhoImagem)
    ehJpeg = formatoOriginalEhJpeg(caminhoImagem)

    #gray
    imagemCinza = cv2.cvtColor(imagemOriginal, cv2.COLOR_BGR2GRAY)
    alturaImagem, larguraImagem = imagemCinza.shape

    #kernelsize (proporcional - agora consistente em todo o pipeline)
    tamanhoKernel = max(3, int(min(alturaImagem, larguraImagem) * 0.005))
    if tamanhoKernel % 2 == 0:
        tamanhoKernel += 1

    #noise
    imagemDesfocada = cv2.GaussianBlur(imagemCinza, (tamanhoKernel, tamanhoKernel), 0)
    residuoRuido = imagemCinza.astype(np.float32) - imagemDesfocada.astype(np.float32)
    listaValorRuido = residuoRuido.flatten()

    #PRNU (aproximado)
    imagemDenoised = cv2.fastNlMeansDenoising(imagemCinza, None, 7, 7, 21)
    residuoPRNU = imagemCinza.astype(np.float32) - imagemDenoised.astype(np.float32)
    listaValorPRNU = residuoPRNU.flatten()

    #FFT
    fftImagem = np.fft.fft2(imagemCinza)
    fftCentralizada = np.fft.fftshift(fftImagem)
    fftMagnitude = np.log(np.abs(fftCentralizada) + 1)
    listaValorFFT = fftMagnitude.flatten()
    perfilRadial = calcularPerfilRadial(fftMagnitude)
    picosRadiais, _ = find_peaks(perfilRadial)
    diferencaRadial = np.diff(perfilRadial)

    #entropy
    mapaEntropia = entropy(imagemCinza, disk(5))
    listaValorEntropia = mapaEntropia.flatten()

    #ELA (agora com arquivo temporário seguro e único)
    with ArquivoTemporarioELA(imagemOriginal, qualidade=90) as caminhoTemporarioELA:
        imagemRecomprimida = cv2.imread(caminhoTemporarioELA)
        imagemELA = cv2.absdiff(imagemOriginal, imagemRecomprimida)
        imagemELACinza = cv2.cvtColor(imagemELA, cv2.COLOR_BGR2GRAY)
        listaValorELA = imagemELACinza.flatten()

    #wavelet
    coeficientesWavelet = pywt.wavedec2(imagemCinza, 'haar', level=2)
    energiaWavelet = []
    for nivel in coeficientesWavelet[1:]:
        for subbanda in nivel:
            energiaWavelet.append(np.mean(np.abs(subbanda)))

    #CFA
    linhasPares = imagemCinza[::2, :]
    linhasImpares = imagemCinza[1::2, :]
    minimoLinhas = min(linhasPares.shape[0], linhasImpares.shape[0])
    linhasPares = linhasPares[:minimoLinhas]
    linhasImpares = linhasImpares[:minimoLinhas]

    diferencaCFA = np.mean(np.abs(
        linhasPares.astype(np.float32) -
        linhasImpares.astype(np.float32)
    ))

    #ruído cromático (kernel agora proporcional, igual ao restante do pipeline)
    canalAzul, canalVerde, canalVermelho = cv2.split(imagemOriginal)

    ruidoVermelho = np.std(
        canalVermelho.astype(np.float32) - cv2.GaussianBlur(canalVermelho, (tamanhoKernel, tamanhoKernel), 0)
    )
    ruidoVerde = np.std(
        canalVerde.astype(np.float32) - cv2.GaussianBlur(canalVerde, (tamanhoKernel, tamanhoKernel), 0)
    )
    ruidoAzul = np.std(
        canalAzul.astype(np.float32) - cv2.GaussianBlur(canalAzul, (tamanhoKernel, tamanhoKernel), 0)
    )

    #saturação
    imagemHSV = cv2.cvtColor(imagemOriginal, cv2.COLOR_BGR2HSV)
    canalSaturacao = imagemHSV[:, :, 1]

    #LAB (nomes com sufixo Lab para não colidir com os canais RGB acima)
    imagemLAB = cv2.cvtColor(imagemOriginal, cv2.COLOR_BGR2LAB)
    _, canalLabA, canalLabB = cv2.split(imagemLAB)

    #canny-edge
    mapaBordas = cv2.Canny(imagemCinza, 100, 200)
    densidadeBordas = np.mean(mapaBordas > 0)

    caracteristicas = {
        #RUÍDO
        "valorRuidoDesvio": float(np.std(listaValorRuido)),
        "valorRuidoAssimetria": float(skew(listaValorRuido)),
        "valorRuidoCurtose": float(kurtosis(listaValorRuido)),
        #PRNU
        "valorPRNUDesvio": float(np.std(listaValorPRNU)),
        "valorPRNUAssimetria": float(skew(listaValorPRNU)),
        "valorPRNUCurtose": float(kurtosis(listaValorPRNU)),
        #FFT
        "valorFFTDesvio": float(np.std(listaValorFFT)),
        "valorFFTAssimetria": float(skew(listaValorFFT)),
        "valorFFTCurtose": float(kurtosis(listaValorFFT)),
        #RADIAL FFT
        "valorRadialDesvio": float(np.std(perfilRadial)),
        "valorRadialPicos": int(len(picosRadiais)),
        "valorRadialEnergia": float(np.mean(np.abs(diferencaRadial))),
        "valorRadialCurtose": float(kurtosis(perfilRadial)),
        #ENTROPIA
        "valorEntropiaMedia": float(np.mean(listaValorEntropia)),
        "valorEntropiaDesvio": float(np.std(listaValorEntropia)),
        "valorEntropiaAssimetria": float(skew(listaValorEntropia)),
        "valorEntropiaCurtose": float(kurtosis(listaValorEntropia)),
        #ELA
        "valorELADesvio": float(np.std(listaValorELA)),
        "valorELAEntropia": float(shannon_entropy(imagemELACinza)),
        "valorELACurtose": float(kurtosis(listaValorELA)),
        #WAVELET
        "valorWaveletMedia": float(np.mean(energiaWavelet)),
        "valorWaveletDesvio": float(np.std(energiaWavelet)),
        #CFA
        "valorCFADiferenca": float(diferencaCFA),
        #RUÍDO CROMÁTICO
        "valorRuidoVermelho": float(ruidoVermelho),
        "valorRuidoVerde": float(ruidoVerde),
        "valorRuidoAzul": float(ruidoAzul),
        #SATURAÇÃO
        "valorSaturacaoMedia": float(np.mean(canalSaturacao)),
        "valorSaturacaoDesvio": float(np.std(canalSaturacao)),
        #LAB
        "valorLabADesvio": float(np.std(canalLabA)),
        "valorLabBDesvio": float(np.std(canalLabB)),
        #BORDAS
        "valorDensidadeBordas": float(densidadeBordas),
    }

    #Metadado (não usado como feature numérica direta no classificador, mas registrado para permitir análise/filtro
    #posterior, ex: separar o desempenho do modelo para imagens de origem JPEG vs PNG).
    caracteristicas["_metadadoFormatoOriginalJpeg"] = bool(ehJpeg)

    return caracteristicas


if __name__ == "__main__":
    caminhoImagem = input("path: ")
    resultado = extrairCaracteristicas(caminhoImagem)

    with open("caracteristicas.json", "w") as arquivo:
        json.dump(resultado, arquivo, indent=4)

    print(json.dumps(resultado, indent=4))