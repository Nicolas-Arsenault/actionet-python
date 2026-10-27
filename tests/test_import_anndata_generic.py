"""Regression tests for import_anndata_generic memory-safe import path."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.io import mmwrite

import actionet.preprocessing as prep
from actionet.preprocessing import io as prep_io


def _write_generic_inputs(
    tmp_path: Path,
    matrix_genes_by_cells: np.ndarray,
    *,
    gene_rows: list[list[str]],
    sample_rows: list[list[str]],
) -> tuple[str, str, str]:
    """Write MatrixMarket + annotation TSV inputs and return filenames."""
    mtx_name = "matrix.mtx"
    gene_name = "features.tsv"
    sample_name = "barcodes.tsv"

    mmwrite(tmp_path / mtx_name, sp.coo_matrix(matrix_genes_by_cells))
    pd.DataFrame(gene_rows).to_csv(
        tmp_path / gene_name, sep="\t", header=False, index=False,
    )
    pd.DataFrame(sample_rows).to_csv(
        tmp_path / sample_name, sep="\t", header=False, index=False,
    )
    return mtx_name, gene_name, sample_name


def _to_dense(x) -> np.ndarray:
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def test_import_anndata_generic_roundtrip_orientation_and_values(tmp_path):
    """MatrixMarket genes x cells input is imported as cells x genes AnnData."""
    matrix = np.array(
        [
            [1, 0],
            [2, 3],
            [0, 4],
        ],
        dtype=np.int64,
    )
    mtx_name, gene_name, sample_name = _write_generic_inputs(
        tmp_path,
        matrix,
        gene_rows=[["g1", "Gene1"], ["g2", "Gene2"], ["g3", "Gene3"]],
        sample_rows=[["c1"], ["c2"]],
    )

    adata = prep.import_anndata_generic(
        str(tmp_path),
        mtx_file=mtx_name,
        gene_annotations=gene_name,
        sample_annotations=sample_name,
        gene_headers=["ENSEMBL", "Gene"],
        sample_headers=["Barcode"],
        sep="\t",
        prefilter=False,
    )

    np.testing.assert_array_equal(
        _to_dense(adata.X),
        np.array([[1, 2, 0], [0, 3, 4]], dtype=np.int64),
    )
    assert adata.shape == (2, 3)
    assert list(adata.obs_names) == ["c1", "c2"]
    assert list(adata.var_names) == ["g1", "g2", "g3"]


def test_import_anndata_generic_makes_duplicate_obs_and_var_names_unique(tmp_path):
    matrix = np.array(
        [
            [1, 0, 0],
            [0, 1, 1],
            [2, 0, 3],
        ],
        dtype=np.int64,
    )
    mtx_name, gene_name, sample_name = _write_generic_inputs(
        tmp_path,
        matrix,
        gene_rows=[["geneA"], ["geneA"], ["geneB"]],
        sample_rows=[["cell1"], ["cell1"], ["cell2"]],
    )

    adata = prep.import_anndata_generic(
        str(tmp_path),
        mtx_file=mtx_name,
        gene_annotations=gene_name,
        sample_annotations=sample_name,
        sep="\t",
        prefilter=False,
    )

    assert list(adata.obs_names) == ["cell1", "cell1_1", "cell2"]
    assert list(adata.var_names) == ["geneA", "geneA_1", "geneB"]


def test_import_anndata_generic_returns_csr_matrix_for_sparse_input(tmp_path):
    matrix = np.array(
        [
            [0, 5],
            [6, 0],
        ],
        dtype=np.float64,
    )
    mtx_name, gene_name, sample_name = _write_generic_inputs(
        tmp_path,
        matrix,
        gene_rows=[["g1"], ["g2"]],
        sample_rows=[["c1"], ["c2"]],
    )

    adata = prep.import_anndata_generic(
        str(tmp_path),
        mtx_file=mtx_name,
        gene_annotations=gene_name,
        sample_annotations=sample_name,
        sep="\t",
        prefilter=False,
    )

    assert sp.isspmatrix_csr(adata.X)


def test_import_anndata_generic_calls_mmread_with_spmatrix_true(tmp_path, monkeypatch):
    matrix = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.float64,
    )
    mtx_name, gene_name, sample_name = _write_generic_inputs(
        tmp_path,
        matrix,
        gene_rows=[["g1"], ["g2"]],
        sample_rows=[["c1"], ["c2"]],
    )

    called: dict[str, object] = {}

    def fake_mmread(source, **kwargs):
        called["source"] = source
        called["kwargs"] = dict(kwargs)
        return sp.coo_matrix(matrix)

    monkeypatch.setattr(prep_io, "mmread", fake_mmread)

    _ = prep.import_anndata_generic(
        str(tmp_path),
        mtx_file=mtx_name,
        gene_annotations=gene_name,
        sample_annotations=sample_name,
        sep="\t",
        prefilter=False,
    )

    assert called["source"] == str(tmp_path / mtx_name)
    assert called["kwargs"] == {"spmatrix": True}


def _import(tmp_path, matrix, **kwargs):
    """Write a genes x cells matrix and import it with extra kwargs."""
    n_genes, n_cells = matrix.shape
    mtx_name, gene_name, sample_name = _write_generic_inputs(
        tmp_path,
        matrix,
        gene_rows=[[f"g{i}"] for i in range(n_genes)],
        sample_rows=[[f"c{i}"] for i in range(n_cells)],
    )
    return prep.import_anndata_generic(
        str(tmp_path),
        mtx_file=mtx_name,
        gene_annotations=gene_name,
        sample_annotations=sample_name,
        sep="\t",
        prefilter=False,
        **kwargs,
    )


def test_import_anndata_generic_default_dtypes_unchanged(tmp_path):
    """Without dtype kwargs the MatrixMarket integer field stays int64."""
    adata = _import(tmp_path, np.array([[1, 0], [2, 3]], dtype=np.int64))
    assert adata.X.dtype == np.int64


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_import_anndata_generic_data_dtype(tmp_path, dtype):
    matrix = np.array([[1, 0], [2, 3], [0, 4]], dtype=np.int64)
    adata = _import(tmp_path, matrix, data_dtype=dtype)
    assert adata.X.dtype == dtype
    np.testing.assert_array_equal(_to_dense(adata.X), matrix.T)


def test_import_anndata_generic_index_dtype(tmp_path):
    matrix = np.array([[1, 0], [2, 3], [0, 4]], dtype=np.int64)
    adata = _import(tmp_path, matrix, index_dtype=np.int32, data_dtype=np.int32)
    assert sp.isspmatrix_csr(adata.X)
    assert adata.X.indices.dtype == np.int32
    assert adata.X.indptr.dtype == np.int32
    np.testing.assert_array_equal(_to_dense(adata.X), matrix.T)


def test_import_anndata_generic_int_data_dtype_rejects_real_field(tmp_path):
    matrix = np.array([[1.5, 0.0], [0.0, 2.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="requires an integer MatrixMarket field"):
        _import(tmp_path, matrix, data_dtype=np.int32)


def test_import_anndata_generic_int_data_dtype_rejects_overflow(tmp_path):
    matrix = np.array([[2**40, 0], [0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="do not fit"):
        _import(tmp_path, matrix, data_dtype=np.int32)


def test_import_anndata_generic_index_dtype_rejects_overflow(tmp_path):
    """200 cells cannot be addressed by int8 (max 127) indices."""
    matrix = np.eye(200, dtype=np.int64)
    with pytest.raises(ValueError, match="cannot address"):
        _import(tmp_path, matrix, index_dtype=np.int8)
