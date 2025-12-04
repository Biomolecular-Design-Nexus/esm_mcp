"""
ESM-based protein fitness prediction tools.

This MCP Server provides 1 tool:
1. esm_predict_fitness: Predict fitness values using pre-trained ESM-based models

The tool uses pre-extracted ESM embeddings and a trained model to predict fitness.
"""

# Standard imports
from typing import Annotated, Literal
import pandas as pd
import numpy as np
import joblib
import torch
from pathlib import Path
import os
from fastmcp import FastMCP
from datetime import datetime

# MCP server instance
esm_predict_fitness_mcp = FastMCP(name="esm_predict_fitness")


def load_embeddings(data_dir, csv_file, backbone_model='esm2_t33_650M_UR50D', seq_col='seq'):
    """Load ESM embeddings for sequences in the CSV file."""
    csv_path = os.path.join(data_dir, csv_file)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df_data = pd.read_csv(csv_path)

    # Check if sequence column exists
    if seq_col not in df_data.columns:
        raise ValueError(f"Sequence column '{seq_col}' not found in CSV. Available columns: {df_data.columns.tolist()}")

    # Try to load pre-computed embeddings file
    embd_file = f'{csv_path}.{backbone_model}.npy'

    if os.path.exists(embd_file):
        prot_embd = np.load(embd_file)
    else:
        # Load individual .pt files based on sequences.fasta
        layer = 36 if 't36' in backbone_model else 33

        # Check for sequences.fasta
        fasta_file = os.path.join(data_dir, 'sequences.fasta')
        emb_dir = os.path.join(data_dir, backbone_model)

        if not os.path.exists(fasta_file):
            raise FileNotFoundError(
                f"FASTA file not found: {fasta_file}. "
                f"Please run esm_extract_embeddings_from_csv first."
            )

        if not os.path.exists(emb_dir):
            raise FileNotFoundError(
                f"Embeddings directory not found: {emb_dir}. "
                f"Please run esm_extract_embeddings_from_csv first."
            )

        # Build seq -> embedding mapping
        seq2embd = {}
        seq_ids = []
        sequences = []

        # Parse FASTA file
        with open(fasta_file, 'r') as f:
            current_id = None
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id is not None:
                        sequences.append(''.join(current_seq))
                        seq_ids.append(current_id)
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            # Don't forget the last sequence
            if current_id is not None:
                sequences.append(''.join(current_seq))
                seq_ids.append(current_id)

        # Load embeddings
        for seq_id, seq in zip(seq_ids, sequences):
            emb_file = os.path.join(emb_dir, f'{seq_id}.pt')
            if not os.path.exists(emb_file):
                raise FileNotFoundError(f"Embedding file not found: {emb_file}")
            emb = torch.load(emb_file, map_location='cpu')['mean_representations'][layer]
            seq2embd[seq] = emb

        # Get embeddings for all sequences in data.csv
        prot_embd = []
        missing_seqs = []

        for idx, seq in enumerate(df_data[seq_col]):
            if seq in seq2embd:
                prot_embd.append(seq2embd[seq])
            else:
                missing_seqs.append((idx, seq))

        if missing_seqs:
            raise ValueError(
                f"Could not find embeddings for {len(missing_seqs)} sequences. "
                f"Please ensure you ran esm_extract_embeddings_from_csv on the correct CSV file."
            )

        prot_embd = torch.stack(prot_embd, dim=0).numpy()

        # Save for future use
        np.save(embd_file, prot_embd)

    embeddings = prot_embd

    return embeddings, df_data


@esm_predict_fitness_mcp.tool
def esm_predict_fitness(
    data_csv: Annotated[str, "Path to CSV file containing protein sequences"],
    model_dir: Annotated[str, "Directory containing trained model (head_model_*.joblib and pca_model.joblib)"],
    backbone_model: Annotated[
        Literal[
            "esm2_t33_650M_UR50D",
            "esm1v_t33_650M_UR90S_1",
            "esm1v_t33_650M_UR90S_2",
            "esm1v_t33_650M_UR90S_3",
            "esm1v_t33_650M_UR90S_4",
            "esm1v_t33_650M_UR90S_5",
            "esm2_t36_3B_UR50D"
        ],
        "ESM backbone model used for embeddings"
    ] = "esm2_t33_650M_UR50D",
    seq_col: Annotated[str, "Column name containing protein sequences"] = "seq",
    output_csv: Annotated[str | None, "Output CSV file path. If None, uses <input_csv>_<model_name>_pred.csv"] = None,
    output_suffix: Annotated[str, "Suffix for prediction column name"] = "_pred",
) -> dict:
    """
    Predict fitness values using a pre-trained ESM-based model.

    This tool:
    1. Loads ESM embeddings for sequences in CSV file
    2. Loads trained PCA and head model from model directory
    3. Applies PCA transformation to embeddings
    4. Makes fitness predictions using the head model
    5. Saves results with predictions column

    Input: CSV with sequences, model directory, backbone model name
    Output: Dictionary with predictions path, statistics, and metadata
    """
    try:
        # Convert paths
        data_csv = Path(data_csv)
        model_dir = Path(model_dir)
        data_dir = data_csv.parent

        # Load the trained models
        pca_path = model_dir / 'pca_model.joblib'
        if not pca_path.exists():
            raise FileNotFoundError(f"PCA model not found: {pca_path}")
        pca_model = joblib.load(pca_path)

        # Find the head model file
        head_model_files = list(model_dir.glob('head_model_*.joblib'))
        if not head_model_files:
            raise FileNotFoundError(f"No head model found in {model_dir}")
        head_model_path = head_model_files[0]
        head_model = joblib.load(head_model_path)

        # Extract model type from filename
        head_model_type = head_model_path.stem.replace('head_model_', '')

        # Load embeddings
        embeddings, df_data = load_embeddings(
            data_dir=str(data_dir),
            csv_file=data_csv.name,
            backbone_model=backbone_model,
            seq_col=seq_col
        )

        # Apply PCA transformation
        embeddings_pca = pca_model.transform(embeddings)
        explained_var = np.sum(pca_model.explained_variance_ratio_)

        # Make predictions
        predictions = head_model.predict(embeddings_pca)

        # Add predictions to dataframe
        df_results = df_data.copy()
        df_results[f'fitness{output_suffix}'] = predictions

        # Calculate statistics
        pred_stats = {
            "count": int(len(predictions)),
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
        }

        # Determine output path
        if output_csv:
            output_path = output_csv
        else:
            # Extract model name from model_dir
            model_dir_name = model_dir.parent.name if model_dir.name == 'final_model' else model_dir.name
            output_path = str(data_dir / f"{data_csv.name}_{model_dir_name}_pred.csv")

        # Save results
        df_results.to_csv(output_path, index=False)

        return {
            "status": "success",
            "output_csv": output_path,
            "backbone_model": backbone_model,
            "head_model": head_model_type,
            "pca_components": int(pca_model.n_components_),
            "pca_explained_variance": float(explained_var),
            "embedding_shape": list(embeddings.shape),
            "total_predictions": int(len(predictions)),
            "prediction_statistics": pred_stats,
            "model_files": {
                "pca_model": str(pca_path),
                "head_model": str(head_model_path),
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "data_csv": str(data_csv),
            "model_dir": str(model_dir),
        }
