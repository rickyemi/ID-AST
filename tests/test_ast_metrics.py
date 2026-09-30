"""Tier 3 MIC rule and AST metrics on hand-checked cases."""
import numpy as np

from src import config as cfg
from src.models.ast_metrics import ast_metrics, categorise, mic_from_probs


def test_mic_rule_basic_and_off_scale():
    assert mic_from_probs([.9, .9, .9, .1, .1]) == 4
    assert mic_from_probs([.1, .1, .1]) == 1          # no growth anywhere -> <= lowest
    assert mic_from_probs([.9, .9, .9]) == 4          # growth everywhere -> > highest (n+1)


def test_mic_rule_ignores_a_skipped_well():
    # well 2 looks like no growth but wells 3-4 grow: a skipped well, MIC stays at DSI 5
    assert mic_from_probs([.95, .1, .95, .95, .05, .05]) == 5


def test_categories_snap_to_dilution_grid(breakpoints):
    sa = "Staphylococcus aureus"
    assert categorise(sa, "Ampicillin", 1, breakpoints) == "S"      # <=0.12 vs S <= 0.12
    assert categorise(sa, "Ampicillin", 2, breakpoints) == "R"      # 0.25
    assert categorise(sa, "Vancomycin", 6, breakpoints) == "I"      # 8 ug/mL
    assert categorise("Enterococcus faecalis", "Vancomycin", 9, breakpoints) == "R"   # >32


def test_ast_metrics_counts_vme_and_me(breakpoints):
    org, drug = ["Staphylococcus aureus"] * 4, ["Oxacillin"] * 4   # S <= 2 (DSI 6), R >= 4 (DSI 7)
    ref = [7, 7, 3, 3]
    pred = [6, 7, 3, 8]          # VME, exact, exact, ME
    m = ast_metrics(ref, pred, org, drug, breakpoints)
    assert m["n_VME"] == 1 and m["n_ME"] == 1
    assert np.isclose(m["VME"], .5) and np.isclose(m["ME"], .5)
    assert np.isclose(m["AA"], .5) and np.isclose(m["EA"], .75)


def test_dsi_labels():
    assert cfg.dsi_to_label("Vancomycin", 1) == "<=0.25"
    assert cfg.dsi_to_label("Vancomycin", 9) == ">32"
    assert cfg.dsi_to_label("Oxacillin", 5) == "1"
