#builder.py

#Generates, for each class, a statistical baseline from the real images and a labeled feature
#dataset (real=0 / ai=1), later used by trainer.py to train the model.

#Expected structure:
#dataset/
#    real/
#        <class>/*.jpg
#    ia/
#        <class>/*.jpg

#The builder now supports multiple classes, instead of working with just a single set of real images.
#It now filters out files that aren't images before processing, making the build more robust.
#Besides the statistical baseline, it now also generates a labeled feature dataset for training a
#supervised classifier.

import os
import json
import numpy as np
import pandas as pd

from extractor import extractFeatures
from common import isValidImageFile

DATASET_DIRECTORY = "dataset"
OUTPUT_DIRECTORY = "output"

def listClasses(realDirectory: str) -> list:
    if not os.path.isdir(realDirectory):
        return []
    return sorted([
        className for className in os.listdir(realDirectory)
        if os.path.isdir(os.path.join(realDirectory, className))
    ])


def extractFeaturesFromDirectory(directory: str) -> list:
#Extracts features from every valid image in a directory, returning a list of dictionaries
#containing the extracted features and each file's name.
    featureList = []
    if not os.path.isdir(directory):
        return featureList

    for fileName in sorted(os.listdir(directory)):
        if not isValidImageFile(fileName):
            continue

        fullPath = os.path.join(directory, fileName)
        try:
            features = extractFeatures(fullPath)
            features["_file"] = fileName
            featureList.append(features)
            print(f"  OK: {fileName}")
        except Exception as error:
            print(f"  ERROR: {fileName} -> {error}")

    return featureList


def calculateBaseline(featureRecordList: list) -> dict:
    #Statistical baseline (mean/stdDev/percentiles) for real images.
    if not featureRecordList:
        return {}

    numericNames = [
        key for key in featureRecordList[0].keys()
        if not key.startswith("_")
    ]

    baseline = {}
    for featureName in numericNames:
        values = [record[featureName] for record in featureRecordList]
        baseline[featureName] = {
            "mean": float(np.mean(values)),
            "stdDev": float(np.std(values) + 1e-8),
            "percentile95": float(np.percentile(values, 95)),
            "percentile99": float(np.percentile(values, 99)),
        }
    return baseline


def processClass(classLabel: str):
    print(f"\nClass: {classLabel}")

    realDirectory = os.path.join(DATASET_DIRECTORY, "real", classLabel)
    aiDirectory = os.path.join(DATASET_DIRECTORY, "ia", classLabel)

    print(f"Extracting features from REAL images ({realDirectory})...")
    realFeatures = extractFeaturesFromDirectory(realDirectory)

    print(f"Extracting features from AI images ({aiDirectory})...")
    aiFeatures = extractFeaturesFromDirectory(aiDirectory)

    if len(realFeatures) == 0:
        print(f"WARNING: no valid real image found for class '{classLabel}'. Skipping.")
        return
    if len(aiFeatures) == 0:
        print(
            f"WARNING: no AI image found for class '{classLabel}'. "
            "The baseline will be generated, but the labeled dataset will be incomplete "
            "(trainer.py requires both classes)."
        )

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    #descriptive baseline (real images only, used for tests)
    baseline = calculateBaseline(realFeatures)
    baselinePath = os.path.join(OUTPUT_DIRECTORY, f"baseline_{classLabel}.json")
    with open(baselinePath, "w") as file:
        json.dump(baseline, file, indent=4)
    print(f"Baseline saved to: {baselinePath}")

    #labeled dataset (real+AI) for training the classifier, final-version idea
    #0 = real / #1 = AI-generated/edited
    rows = []
    for record in realFeatures:
        row = {key: value for key, value in record.items() if not key.startswith("_")}
        row["label"] = 0
        rows.append(row)
    for record in aiFeatures:
        row = {key: value for key, value in record.items() if not key.startswith("_")}
        row["label"] = 1
        rows.append(row)

    dataFrame = pd.DataFrame(rows)
    datasetPath = os.path.join(OUTPUT_DIRECTORY, f"features_{classLabel}.csv")
    dataFrame.to_csv(datasetPath, index=False)
    print(f"Labeled dataset saved to: {datasetPath} ({len(dataFrame)} samples)")


def main():
    realDirectory = os.path.join(DATASET_DIRECTORY, "real")
    classes = listClasses(realDirectory)

    if not classes:
        raise Exception(
            f"No class found in '{realDirectory}'. "
            f"Organize the dataset as dataset/real/<class>/ and dataset/ia/<class>/."
        )

    print(f"Classes found: {classes}")
    for classLabel in classes:
        processClass(classLabel)

    print("\nBuild completed for all classes.")


if __name__ == "__main__":
    main()
