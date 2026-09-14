#extractor.py

#This code is responsible for extracting the heuristic features of an image,
#used both to calibrate the baseline (calibrator.py) and to classify a new image (detector.py).

#Improvements based on the first prototype that didn't make sense or had some bugs:

#Every image now goes through loadStandardizedImage() (def in common.py) before extraction,
#ensuring that resolution-sensitive features (radial FFT profile, edge density, wavelet energy)
#stay comparable across images of different sizes/origins.

#The ELA temporary file is now unique per call and always removed (TemporaryELAFile in common.py),
#eliminating the risk of collisions in parallel runs or leftover disk garbage that wasn't handled in initial tests.

#The chromatic noise kernel is now proportional to the image size, like the main kernel
#(before I had a fixed value (3,3) for initial tests, which was inconsistent with the rest of the pipeline).

#The code now also stores the file's original format (JPEG, PNG etc.) alongside the result, since this ends up
#affecting the forensic interpretation of ELA depending on the format.

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
    loadStandardizedImage,
    originalFormatIsJpeg,
    TemporaryELAFile,
)


def calculateRadialProfile(dataMatrix):
    center = np.array(dataMatrix.shape) // 2
    y, x = np.indices(dataMatrix.shape)

    radius = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    radius = radius.astype(np.int32)

    sumByRadius = np.bincount(radius.ravel(), dataMatrix.ravel())
    countByRadius = np.bincount(radius.ravel())
    radialProfile = sumByRadius / (countByRadius + 1e-8)
    return radialProfile


def extractFeatures(imagePath: str) -> dict:
    originalImage = loadStandardizedImage(imagePath)
    isJpeg = originalFormatIsJpeg(imagePath)

    #gray
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    imageHeight, imageWidth = grayImage.shape

    #kernel size (proportional)
    kernelSize = max(3, int(min(imageHeight, imageWidth) * 0.005))
    if kernelSize % 2 == 0:
        kernelSize += 1

    #noise
    blurredImage = cv2.GaussianBlur(grayImage, (kernelSize, kernelSize), 0)
    noiseResidual = grayImage.astype(np.float32) - blurredImage.astype(np.float32)
    noiseValueList = noiseResidual.flatten()

    #PRNU (approximate)
    denoisedImage = cv2.fastNlMeansDenoising(grayImage, None, 7, 7, 21)
    prnuResidual = grayImage.astype(np.float32) - denoisedImage.astype(np.float32)
    prnuValueList = prnuResidual.flatten()

    #FFT
    fftImage = np.fft.fft2(grayImage)
    fftCentered = np.fft.fftshift(fftImage)
    fftMagnitude = np.log(np.abs(fftCentered) + 1)
    fftValueList = fftMagnitude.flatten()
    radialProfile = calculateRadialProfile(fftMagnitude)
    radialPeaks, _ = find_peaks(radialProfile)
    radialDifference = np.diff(radialProfile)

    #entropy
    entropyMap = entropy(grayImage, disk(5))
    entropyValueList = entropyMap.flatten()

    #ELA (with temporary file)
    with TemporaryELAFile(originalImage, quality=90) as temporaryELAPath:
        recompressedImage = cv2.imread(temporaryELAPath)
        elaImage = cv2.absdiff(originalImage, recompressedImage)
        elaGrayImage = cv2.cvtColor(elaImage, cv2.COLOR_BGR2GRAY)
        elaValueList = elaGrayImage.flatten()

    #wavelet
    waveletCoefficients = pywt.wavedec2(grayImage, 'haar', level=2)
    waveletEnergy = []
    for level in waveletCoefficients[1:]:
        for subband in level:
            waveletEnergy.append(np.mean(np.abs(subband)))

    #CFA
    evenRows = grayImage[::2, :]
    oddRows = grayImage[1::2, :]
    minRows = min(evenRows.shape[0], oddRows.shape[0])
    evenRows = evenRows[:minRows]
    oddRows = oddRows[:minRows]

    cfaDifference = np.mean(np.abs(
        evenRows.astype(np.float32) -
        oddRows.astype(np.float32)
    ))

    #chromatic noise (kernel now proportional, same as the rest of the pipeline)
    blueChannel, greenChannel, redChannel = cv2.split(originalImage)

    redNoise = np.std(
        redChannel.astype(np.float32) - cv2.GaussianBlur(redChannel, (kernelSize, kernelSize), 0)
    )
    greenNoise = np.std(
        greenChannel.astype(np.float32) - cv2.GaussianBlur(greenChannel, (kernelSize, kernelSize), 0)
    )
    blueNoise = np.std(
        blueChannel.astype(np.float32) - cv2.GaussianBlur(blueChannel, (kernelSize, kernelSize), 0)
    )

    #saturation
    hsvImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)
    saturationChannel = hsvImage[:, :, 1]

    #LAB (names with a "Lab" suffix so they don't collide with the RGB channels above)
    labImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2LAB)
    _, labAChannel, labBChannel = cv2.split(labImage)

    #canny-edge
    edgeMap = cv2.Canny(grayImage, 100, 200)
    edgeDensity = np.mean(edgeMap > 0)

    features = {
        #NOISE
        "noiseStdDev": float(np.std(noiseValueList)),
        "noiseSkew": float(skew(noiseValueList)),
        "noiseKurtosis": float(kurtosis(noiseValueList)),
        #PRNU
        "prnuStdDev": float(np.std(prnuValueList)),
        "prnuSkew": float(skew(prnuValueList)),
        "prnuKurtosis": float(kurtosis(prnuValueList)),
        #FFT
        "fftStdDev": float(np.std(fftValueList)),
        "fftSkew": float(skew(fftValueList)),
        "fftKurtosis": float(kurtosis(fftValueList)),
        #RADIAL FFT
        "radialStdDev": float(np.std(radialProfile)),
        "radialPeaksCount": int(len(radialPeaks)),
        "radialEnergy": float(np.mean(np.abs(radialDifference))),
        "radialKurtosis": float(kurtosis(radialProfile)),
        #ENTROPY
        "entropyMean": float(np.mean(entropyValueList)),
        "entropyStdDev": float(np.std(entropyValueList)),
        "entropySkew": float(skew(entropyValueList)),
        "entropyKurtosis": float(kurtosis(entropyValueList)),
        #ELA
        "elaStdDev": float(np.std(elaValueList)),
        "elaEntropy": float(shannon_entropy(elaGrayImage)),
        "elaKurtosis": float(kurtosis(elaValueList)),
        #WAVELET
        "waveletMean": float(np.mean(waveletEnergy)),
        "waveletStdDev": float(np.std(waveletEnergy)),
        #CFA
        "cfaDifferenceValue": float(cfaDifference),
        #CHROMATIC NOISE
        "redNoiseValue": float(redNoise),
        "greenNoiseValue": float(greenNoise),
        "blueNoiseValue": float(blueNoise),
        #SATURATION
        "saturationMean": float(np.mean(saturationChannel)),
        "saturationStdDev": float(np.std(saturationChannel)),
        #LAB
        "labAStdDev": float(np.std(labAChannel)),
        "labBStdDev": float(np.std(labBChannel)),
        #EDGES
        "edgeDensityValue": float(edgeDensity),
    }

    #Metadata (not used as a direct numeric feature by the classifier, but recorded to allow later
    #analysis/filtering, e.g. splitting model performance for images originally JPEG vs PNG).
    features["_metadataOriginalFormatJpeg"] = bool(isJpeg)

    return features


if __name__ == "__main__":
    imagePath = input("path: ")
    result = extractFeatures(imagePath)

    with open("features.json", "w") as file:
        json.dump(result, file, indent=4)

    print(json.dumps(result, indent=4))
