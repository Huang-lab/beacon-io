"""Multi-evidence integration: combine all BEACON-IO analyses into a
unified target-ranking framework.

Evidence tiers (inspired by Open Targets):
  E1 - Expression-driven dependency (BEACON, cell-line level)
  E2 - Immune-context specificity (differential EDD hot vs cold)
  E3 - Immune evasion correlation (EDD ~ evasion programme)
  E4 - Drug sensitivity (PRISM correlation with EDD target expression)
  E5 - Clinical ICB response (meta-analysis across ICB cohorts)
  E6 - TCGA survival association
  E7 - Single-cell compartment (tumour-intrinsic confirmed)
  E8 - Druggability (DGIdb / DrugBank annotation)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from beacon_io.utils import get_logger

log = get_logger(__name__)

EVIDENCE_WEIGHTS = {
    "E1_edd": 0.15,
    "E2_immune_specific": 0.15,
    "E3_evasion_corr": 0.10,
    "E4_prism": 0.15,
    "E5_icb_response": 0.20,
    "E6_tcga_survival": 0.10,
    "E7_tumour_intrinsic": 0.05,
    "E8_druggable": 0.10,
}


def compile_evidence(
    edd_results: pd.DataFrame,
    diff_edd: pd.DataFrame,
    evasion_corr: pd.DataFrame,
    prism_hits: pd.DataFrame,
    icb_meta: pd.DataFrame,
    tcga_survival: pd.DataFrame,
    compartment: pd.DataFrame,
    druggability: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all evidence streams into a single gene-level table.

    Each evidence column is rank-normalised to [0, 1] and weighted.
    Returns DataFrame sorted by composite evidence score.
    """
    # Start with BEACON-EDD evidence base. We require an E1 (EDD rho) signal
    # so that the composite ranking is meaningful — genes with only weak
    # immune-specific or evasion evidence (and no EDD signal) would
    # otherwise dominate the table because the composite ignores missing
    # tiers rather than penalising them. If no EDD evidence is provided
    # (development scenario), fall back to the union of all evidence
    # sources excluding the genome-wide evasion table.
    genes = set()
    if not edd_results.empty and "gene" in edd_results.columns:
        genes.update(edd_results["gene"].unique())
    if not genes:
        for df in [diff_edd, prism_hits, icb_meta, tcga_survival]:
            if not df.empty and "gene" in df.columns:
                genes.update(df["gene"].unique())

    master = pd.DataFrame({"gene": sorted(genes)})

    # E1: EDD strength (most negative rho = strongest)
    if not edd_results.empty:
        e1 = edd_results.groupby("gene")["rho_posterior_median"].min().reset_index()
        e1.columns = ["gene", "E1_edd_rho"]
        master = master.merge(e1, on="gene", how="left")

    # E2: Immune-specific EDD (largest absolute delta)
    if not diff_edd.empty:
        e2 = diff_edd.groupby("gene")["delta_rho"].apply(lambda x: x.abs().max()).reset_index()
        e2.columns = ["gene", "E2_delta_rho"]
        master = master.merge(e2, on="gene", how="left")

    # E3: Evasion correlation (strongest association)
    if not evasion_corr.empty:
        e3 = evasion_corr.groupby("gene")["rho"].apply(lambda x: x.abs().max()).reset_index()
        e3.columns = ["gene", "E3_evasion_rho"]
        master = master.merge(e3, on="gene", how="left")

    # E4: PRISM drug sensitivity
    if not prism_hits.empty:
        e4 = prism_hits.groupby("gene")["rho"].min().reset_index()
        e4.columns = ["gene", "E4_prism_rho"]
        master = master.merge(e4, on="gene", how="left")

    # E5: ICB meta-analysis
    if not icb_meta.empty:
        e5 = icb_meta[["gene", "meta_rho", "meta_fdr"]].copy()
        e5.columns = ["gene", "E5_icb_rho", "E5_icb_fdr"]
        master = master.merge(e5, on="gene", how="left")

    # E6: TCGA survival
    if not tcga_survival.empty:
        e6 = tcga_survival.groupby("gene").agg(
            E6_min_cox_p=("cox_pvalue", "min"),
            E6_mean_hr=("cox_hr", "mean"),
        ).reset_index()
        master = master.merge(e6, on="gene", how="left")

    # E7: Tumour-intrinsic compartment
    if not compartment.empty:
        e7 = compartment[["gene", "primary_compartment", "tumour_fraction"]].copy()
        e7["E7_tumour_intrinsic"] = (e7["primary_compartment"] == "tumour_intrinsic").astype(float)
        master = master.merge(e7[["gene", "E7_tumour_intrinsic", "tumour_fraction"]], on="gene", how="left")

    # E8: Druggability
    if not druggability.empty:
        e8 = druggability.groupby("gene").size().reset_index(name="E8_n_drugs")
        e8["E8_druggable"] = 1.0
        master = master.merge(e8[["gene", "E8_druggable", "E8_n_drugs"]], on="gene", how="left")

    # Compute composite score
    master = _compute_composite(master)
    return master.sort_values("composite_score", ascending=False)


def _compute_composite(master: pd.DataFrame) -> pd.DataFrame:
    """Rank-normalise each evidence column and compute weighted composite.

    `smaller_is_better` indicates the semantic ordering of values
    (e.g. more negative rho = better, smaller p-value = better). For
    each column the values are converted to a [0,1] percentile so that
    1.0 = strongest evidence; missing values are filled to give the
    worst possible rank (0.0) before ranking.
    """
    score_cols = {
        # column, smaller_is_better (semantic)
        "E1_edd": ("E1_edd_rho", True),        # more negative rho = better
        "E2_immune_specific": ("E2_delta_rho", False),  # larger |delta| = better
        "E3_evasion_corr": ("E3_evasion_rho", False),
        "E4_prism": ("E4_prism_rho", True),     # more negative rho = better
        "E5_icb_response": ("E5_icb_rho", True),
        "E6_tcga_survival": ("E6_min_cox_p", True),  # smaller p = better
        "E7_tumour_intrinsic": ("E7_tumour_intrinsic", False),
        "E8_druggable": ("E8_druggable", False),
    }

    composite = np.zeros(len(master))
    for evidence_key, (col, smaller_is_better) in score_cols.items():
        weight = EVIDENCE_WEIGHTS.get(evidence_key, 0)
        if col not in master.columns:
            continue
        # Fill missing values with a "worst" value before ranking. For
        # smaller-is-better columns, NaN -> +inf-equivalent (large);
        # for larger-is-better columns, NaN -> -inf-equivalent (small).
        col_vals = master[col]
        if smaller_is_better:
            fill = col_vals.max(skipna=True) + 1 if pd.notna(col_vals.max()) else 1.0
        else:
            fill = col_vals.min(skipna=True) - 1 if pd.notna(col_vals.min()) else -1.0
        vals = col_vals.fillna(fill)
        # Convert to [0,1] where 1.0 = best evidence. For smaller-is-better
        # columns, invert ranking so the smallest (most negative) values
        # get the highest percentile.
        ranked = vals.rank(ascending=not smaller_is_better, pct=True)
        composite += weight * ranked.values

    master["composite_score"] = composite
    return master


def summarise_top_targets(
    evidence: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:
    """Pretty-print top BEACON-IO targets with all evidence."""
    cols = ["gene", "composite_score"]
    for c in evidence.columns:
        if c.startswith("E") and c not in cols:
            cols.append(c)
    available = [c for c in cols if c in evidence.columns]
    return evidence[available].head(top_n)
