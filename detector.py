#detector.py

#This code takes an image (or a directory of images, in batch mode) and estimates the probability that it was
#generated or edited by AI, using the classifier trained by trainer.py for the given class.

#Changes compared to the prototype

#The detector now uses an argparse-based CLI for batch processing and model evaluation
#on independent test sets.

#The decision is no longer based on a sum of z-scores with manual weights. It now uses the
#supervised model trained for each class, keeping an evidence report grounded both in
#feature importance and in the deviation from the baseline.

#Batch mode support was also added, exporting results to CSV, which makes it easier to validate
#and analyze performance on separate test sets.

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from extractor import extractFeatures
from common import isValidImageFile

OUTPUT_DIRECTORY = "output"


def loadModel(classLabel: str):
    modelPath = os.path.join(OUTPUT_DIRECTORY, f"model_{classLabel}.joblib")
    if not os.path.exists(modelPath):
        raise FileNotFoundError(
            f"Model not found: {modelPath}. Run trainer.py --class {classLabel} first."
        )
    return joblib.load(modelPath)


def loadBaseline(classLabel: str) -> dict:
    baselinePath = os.path.join(OUTPUT_DIRECTORY, f"baseline_{classLabel}.json")
    if not os.path.exists(baselinePath):
        return {}
    with open(baselinePath, "r") as file:
        return json.load(file)


def generateEvidence(features: dict, baseline: dict, model, featureColumns: list, topN: int = 5) -> list:
#Generates a list of readable evidence, combining how much the feature deviates from the real-image baseline
#(z-score) with how important that feature is to the trained model (when available).

    importances = {}
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(featureColumns, model.feature_importances_))

    evidenceList = []
    for featureName in featureColumns:
        if featureName not in baseline or featureName not in features:
            continue
        mean = baseline[featureName]["mean"]
        stdDev = baseline[featureName]["stdDev"]
        if stdDev == 0:
            continue

        zScore = abs((features[featureName] - mean) / stdDev)
        importanceWeight = importances.get(featureName, 0.0)

        #combined score: statistical deviation weighted by the feature's relevance in the model
        evidenceScore = zScore * (importanceWeight if importances else 1.0)

        if zScore > 2:
            evidenceList.append({
                "feature": featureName,
                "zScore": round(float(zScore), 2),
                "modelImportance": round(float(importanceWeight), 4) if importances else None,
                "evidenceScore": round(float(evidenceScore), 4),
            })

    evidenceList.sort(key=lambda evidence: evidence["evidenceScore"], reverse=True)
    return evidenceList[:topN]


def analyzeImage(imagePath: str, classLabel: str, loadedModel=None, baseline=None) -> dict:
    if loadedModel is None:
        loadedModel = loadModel(classLabel)
    if baseline is None:
        baseline = loadBaseline(classLabel)

    model = loadedModel["model"]
    featureColumns = loadedModel["featureColumns"]

    features = extractFeatures(imagePath)
    numericFeatures = {key: value for key, value in features.items() if not key.startswith("_")}

    vector = np.array([[numericFeatures.get(featureName, 0.0) for featureName in featureColumns]])
    aiProbability = float(model.predict_proba(vector)[0, 1])

    evidenceList = generateEvidence(numericFeatures, baseline, model, featureColumns)

    return {
        "file": os.path.basename(imagePath),
        "classLabel": classLabel,
        "aiProbability": round(aiProbability, 4),
        "originalFormatJpeg": features.get("_metadataOriginalFormatJpeg"),
        "evidenceList": evidenceList,
    }


def singleMode(imagePath: str, classLabel: str):
    result = analyzeImage(imagePath, classLabel)
    print(json.dumps(result, indent=4, ensure_ascii=False))
    print("-" * 40)
    print(f"Estimated AI probability: {result['aiProbability'] * 100:.2f}%")
    if not result["originalFormatJpeg"]:
        print(
            "Note: the original image is not JPEG - ELA analysis has "
            "reduced reliability for this file (see the method's documentation)."
        )
    if result["evidenceList"]:
        print("\nTop evidence:")
        for evidence in result["evidenceList"]:
            print(f"  {evidence['feature']}: z={evidence['zScore']} (model importance: {evidence['modelImportance']})")
    else:
        print("\nNo relevant statistical evidence found (z > 2).")


def batchMode(directory: str, classLabel: str, outputPath: str):
    loadedModel = loadModel(classLabel)
    baseline = loadBaseline(classLabel)

    rows = []
    for fileName in sorted(os.listdir(directory)):
        if not isValidImageFile(fileName):
            continue
        fullPath = os.path.join(directory, fileName)
        try:
            result = analyzeImage(fullPath, classLabel, loadedModel, baseline)
            rows.append({
                "file": result["file"],
                "aiProbability": result["aiProbability"],
                "originalFormatJpeg": result["originalFormatJpeg"],
            })
            print(f"OK: {fileName} -> {result['aiProbability'] * 100:.2f}%")
        except Exception as error:
            print(f"ERROR: {fileName} -> {error}")

    dataFrame = pd.DataFrame(rows)
    dataFrame.to_csv(outputPath, index=False)
    print(f"\nResults saved to: {outputPath}")


def main():
    parser = argparse.ArgumentParser(description="Jenos - heuristic detection of AI-generated/edited images.")
    parser.add_argument("--class", dest="classLabel", required=True, help="Model class to use (e.g.: faces, landscapes)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path of a single image")
    group.add_argument("--dir", help="Directory with several images (batch mode)")

    parser.add_argument("--output", dest="outputPath", default="output/results_batch.csv", help="Output CSV for batch mode")
    args = parser.parse_args()

    if args.image:
        singleMode(args.image, args.classLabel)
    else:
        batchMode(args.dir, args.classLabel, args.outputPath)


if __name__ == "__main__":
    main()
