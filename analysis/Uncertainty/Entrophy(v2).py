import numpy as np
import pandas as pd
from pathlib import Path


# -------------------------------------------------------
# Load probability sheet and automatically split models
# -------------------------------------------------------
def load_models_from_excel(path, sheet_name="Probs"):

    df = pd.read_excel(
        path,
        sheet_name=sheet_name
    )

    models = {}

    columns = df.columns.tolist()

    # Find model start columns
    start_cols = [
        i for i, c in enumerate(columns)
        if str(c).endswith("_y_real")
    ]

    for idx, start in enumerate(start_cols):

        # Model name
        model_name = str(columns[start]).replace("_y_real", "")

        # End of this model block
        if idx + 1 < len(start_cols):
            end = start_cols[idx + 1]
        else:
            end = len(columns)

        block = df.iloc[:, start:end].copy()

        # Rename columns
        rename = {}

        for c in block.columns:

            c = str(c)

            if c.endswith("_y_real"):
                rename[c] = "y_true"

            elif c.endswith("_y_pred"):
                rename[c] = "y_pred"

            elif "prob_" in c:
                rename[c] = c.split("_")[-2] + "_" + c.split("_")[-1]

        block = block.rename(columns=rename)


        # Keep only valid rows
        block = block.dropna(
            subset=["y_true"]
        ).reset_index(drop=True)


        models[model_name] = block


    return models



# -------------------------------------------------------
# Normalized entropy
# -------------------------------------------------------
def normalized_entropy(probs):

    n_classes = probs.shape[1]

    if n_classes <= 1:
        return np.zeros(len(probs))

    probs = np.clip(
        probs,
        1e-12,
        1
    )

    entropy = -np.sum(
        probs * np.log(probs),
        axis=1
    )

    return entropy / np.log(n_classes)



# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():

    excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"


    print("Loading probability sheet...")

    models = load_models_from_excel(
        excel_path,
        sheet_name="Probs(ENN)"
    )


    print("\nDetected Models:")
    for m in models:
        print("-", m)


    comparison_df = pd.DataFrame()


    for model_name, df in models.items():

        print(
            f"\nProcessing {model_name}"
        )
        prob_cols = [
            c for c in df.columns
            if "prob" in c.lower()
        ]
        probs = df[prob_cols].values
        uncertainty = normalized_entropy(
            probs
        )
        if comparison_df.empty:

            comparison_df["y_true"] = df["y_true"]


        comparison_df[
            f"y_pred_{model_name}"
        ] = df["y_pred"]


        comparison_df[
            f"Uncertainty_{model_name}"
        ] = uncertainty

    print("\nPreview:")
    print(comparison_df.head())
    print("\nAverage uncertainty:")
    for model in models:

        print(
            model,
            ":",
            comparison_df[
                f"Uncertainty_{model}"
            ].mean()
        )
    comparison_df.to_clipboard(
        index=False
    )
    print(
        "\nResults copied to clipboard."
    )


if __name__ == "__main__":
    main()