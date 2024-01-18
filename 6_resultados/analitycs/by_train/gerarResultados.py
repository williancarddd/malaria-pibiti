import pandas as pd
import os
from pathlib import Path

def process_csv_file(network, results_path, subsets, columns):
    output_dir = Path(__file__).parent / "_pre_plot"
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / f"results_{network}.csv"

    all_results = []

    for subset in subsets:
        # Caminho para os CSVs de treino e refined
        csvs_path = results_path / subset / network / "metrics" / "csvs"
        refined_path = csvs_path / "_refined" / f"{network}_refined.csv"

        # Processamento dos arquivos de treino
        train_dfs = [pd.read_csv(csvs_path / file) for file in os.listdir(csvs_path) if file.endswith(".csv")]
        train_df = pd.concat(train_dfs) if train_dfs else None

        # Processamento do arquivo refined
        if os.path.isfile(refined_path):
            refined_df = pd.read_csv(refined_path)
            print(refined_df)
            refined_metrics = refined_df.mean().to_dict()
        else:
            refined_metrics = {}

        # Combinar métricas de treino e refined
        combined_metrics = {"subset": subset, "strategy": "DL", "model": network}
        combined_metrics.update({k: refined_metrics.get(k, None) for k in ["runtime", "val_runtime", "total_runtime"]})

        if train_df is not None:
            for column in columns:
                first_quartile, third_quartile = train_df[column].quantile([0.25, 0.75])
                iqr = third_quartile - first_quartile
                column_median = train_df[column].median()
                column_mean = train_df[column].mean()
                lower_whisker = max(0, first_quartile - 1.5 * iqr)
                upper_whisker = min(100, third_quartile + 1.5 * iqr)

                combined_metrics[f"{column}_mean"] = 100 * column_mean
                combined_metrics[f"{column}_median"] = 100 * column_median
                combined_metrics[f"{column}_lower"] = 100 * first_quartile
                combined_metrics[f"{column}_upper"] = 100 * third_quartile
                combined_metrics[f"{column}_lower_whisker"] = lower_whisker
                combined_metrics[f"{column}_upper_whisker"] = upper_whisker

        all_results.append(combined_metrics)

    # Salvar os resultados em um CSV
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(output_file, index=False)
    print(f"Arquivo gerado: {output_file}")

# Código principal
path_project = Path().absolute().parent.parent.parent
RESULTS_PATH = path_project / "6_resultados"

NEURAL_NETWORKS = ['DenseNet201', 'MobileNetV2', 'InceptionV3']
COLUMNS = ["accuracy", "precision", "specificity", "f1_score", "auc", "npv", "mcc", "val_accuracy", "val_precision", "val_specificity", "val_f1_score", "val_auc", "val_npv", "val_mcc"]
SUBSETS = ['Dataset01_100', 'Dataset01_95.0', 'Dataset01_90.0', 'Dataset01_85.0', 'Dataset01_80.0', 'Dataset01_75.0', 'Dataset01_70.0', 'Dataset01_65.0', 'Dataset01_60.0', 'Dataset01_55.0', 'Dataset01_50.0', 'Dataset01_45.0', 'Dataset01_40.0', 'Dataset01_35.0', 'Dataset01_30.0', 'Dataset01_25.0', 'Dataset01_20.0', 'Dataset01_15.0', 'Dataset01_10.0', 'Dataset01_5.0']

for network in NEURAL_NETWORKS:
    process_csv_file(network, RESULTS_PATH, SUBSETS, COLUMNS)
