"""Train the model inferapi serves."""

import logging

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("inferapi.train")

CATEGORICAL_COLUMNS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]
NUMERIC_COLUMNS = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ]
    )
    classifier = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
    return Pipeline(steps=[("preprocess", preprocessor), ("classify", classifier)])


def main() -> None:
    # =================================================================
    # Step 1 - Load the dataset
    # =================================================================
    logger.info("Loading dataset from %s", settings.data_path)
    frame = pd.read_parquet(settings.data_path)
    logger.info("Loaded %s rows, %s columns", frame.shape[0], frame.shape[1])

    # =================================================================
    # Step 2 - Split features, target and a held-out test set
    # =================================================================
    features = frame[FEATURE_COLUMNS]
    target = frame["y"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    # =================================================================
    # Step 3 - Fit and score
    # =================================================================
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, pipeline.predict(x_test))
    logger.info("Held-out accuracy: %.4f", accuracy)

    # =================================================================
    # Step 4 - Persist the artifact
    # =================================================================
    artifact = {
        "pipeline": pipeline,
        "version": settings.model_version,
        "feature_columns": FEATURE_COLUMNS,
        "classes": list(pipeline.named_steps["classify"].classes_),
    }
    joblib.dump(artifact, settings.model_path)
    print("Logged artifact: ", artifact)
    logger.info("Wrote model artifact to %s (version=%s)", settings.model_path, settings.model_version)


if __name__ == "__main__":
    main()
