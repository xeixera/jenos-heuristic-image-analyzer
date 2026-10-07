#common.py

#Has several defs shared between extractor.py, builder.py and detector.py.

#Ensures every image goes through the same preprocessing before analysis, standardizing
#resolution and other common steps, guaranteeing that extracted features stay comparable across
#different images and that the baseline stays consistent.

import os
import cv2
import numpy as np
import tempfile

#Standard size (side) to which every image is normalized before feature extraction.
#1024 was the initial choice for being large enough to preserve sensor texture/noise, and small enough
#to keep computational cost low in batch.

STANDARD_SIZE = 1024

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def isValidImageFile(fileName: str) -> bool:
    #filters out formats that aren't images (.wav, .pdf)
    _, extension = os.path.splitext(fileName)
    return extension.lower() in VALID_EXTENSIONS


def loadStandardizedImage(imagePath: str) -> np.ndarray:
#Loads an image, standardizing it to STANDARD_SIZExSTANDARD_SIZE.

#The resizing preserves the image's aspect ratio and then does a center crop to avoid distortion and the
#introduction of artificial edges that could affect feature extraction.

#Reading is done with cv2.imread(), which already guarantees a 3-channel BGR image, so no extra
#handling is needed for RGBA or grayscale images.
    image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    if image is None:
        raise Exception(f"Error opening image: {imagePath}")

    height, width = image.shape[:2]
    shorterSide = min(height, width)

    scale = STANDARD_SIZE / shorterSide
    newWidth = max(STANDARD_SIZE, int(round(width * scale)))
    newHeight = max(STANDARD_SIZE, int(round(height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resizedImage = cv2.resize(
        image, (newWidth, newHeight), interpolation=interpolation
    )

    # center-crop to STANDARD_SIZExSTANDARD_SIZE
    startY = (newHeight - STANDARD_SIZE) // 2
    startX = (newWidth - STANDARD_SIZE) // 2
    finalImage = resizedImage[
        startY:startY + STANDARD_SIZE,
        startX:startX + STANDARD_SIZE
    ]

    return finalImage


def originalFormatIsJpeg(imagePath: str) -> bool:
#Indicates whether the original file is in JPEG format.

#Important for interpreting the ELA result, since the technique exploits JPEG compression artifacts.
#For images originally in PNG or other formats, ELA can still provide useful information, but its interpretation
#is different, which is why the original format is stored to preserve the forensic analysis context.
    extension = os.path.splitext(imagePath)[1].lower()
    return extension in {".jpg", ".jpeg"}


class TemporaryELAFile:
#Context manager responsible for creating a temporary JPEG file with a unique name for the ELA calculation.
#This file is automatically removed at the end of execution, even on error, avoiding a buildup of
#temporary files and concurrency issues in parallel runs.

    def __init__(self, image: np.ndarray, quality: int = 90):
        self.image = image
        self.quality = quality
        self.path = None

    def __enter__(self) -> str:
        descriptor, self.path = tempfile.mkstemp(suffix=".jpg", prefix="jenos_ela_")
        os.close(descriptor)
        cv2.imwrite(self.path, self.image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return self.path

    def __exit__(self, excType, excValue, traceback):
        if self.path and os.path.exists(self.path):
            os.remove(self.path)
        return False
