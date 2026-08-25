# Paper Update Manifest

No LaTeX source was present in the repository.

Locked-test row-level probabilities were not persisted by the original final evaluation. They were
reconstructed once from the immutable final model solely for reporting, verified against every
authoritative aggregate metric, and cached. No threshold selection, tuning, or feature selection used
locked-test labels.

## `validation_threshold_analysis.pdf` -> `validation_threshold_analysis_enhanced.pdf`

- Old metric: tau=.46
- New authoritative metric: tau=.47
- Recommended section: Model selection
- Proposed caption: Validation threshold analysis for the full-enhanced model. Panel (a) shows precision, recall, and F1; panel (b) shows false-positive, false-negative, and alert rates. The dashed line marks the validation-selected threshold tau=0.47.
- Interpretation: The figure reports predictive performance or descriptive associations for the final enhanced model.
- Warning: Validation selection, locked-test evaluation, flight classification, graph-chain evaluation, and historical decision support remain distinct; no causal, joint-chain-probability, or live-operational claim is made.

## `final_test_discrimination_calibration.pdf` -> `final_test_evaluation_enhanced.pdf`

- Old metric: AUC=.9882; AP=.8543
- New authoritative metric: AUC=.9914; AP=.9030
- Recommended section: Locked-test results
- Proposed caption: Locked-test discrimination and calibration of the full-enhanced model: (a) ROC curve, (b) precision-recall curve, and (c) equal-frequency calibration curve. The frozen threshold was selected on validation data only.
- Interpretation: The figure reports predictive performance or descriptive associations for the final enhanced model.
- Warning: Validation selection, locked-test evaluation, flight classification, graph-chain evaluation, and historical decision support remain distinct; no causal, joint-chain-probability, or live-operational claim is made.

## `final_test_roc_curve.pdf` -> `final_test_roc_curve_enhanced.pdf`

- Old metric: ROC-AUC=.9882
- New authoritative metric: ROC-AUC=.9914
- Recommended section: Locked-test results
- Proposed caption: Receiver-operating-characteristic curve for the full-enhanced model on the locked November-December 2023 test set (ROC-AUC=0.9914).
- Interpretation: The figure reports predictive performance or descriptive associations for the final enhanced model.
- Warning: Validation selection, locked-test evaluation, flight classification, graph-chain evaluation, and historical decision support remain distinct; no causal, joint-chain-probability, or live-operational claim is made.

## `final_test_precision_recall_curve.pdf` -> `final_test_precision_recall_curve_enhanced.pdf`

- Old metric: AP=.8543
- New authoritative metric: AP=.9030
- Recommended section: Locked-test results
- Proposed caption: Precision-recall curve for the full-enhanced model on the locked November-December 2023 test set (average precision=0.9030); the dashed line denotes outcome prevalence.
- Interpretation: The figure reports predictive performance or descriptive associations for the final enhanced model.
- Warning: Validation selection, locked-test evaluation, flight classification, graph-chain evaluation, and historical decision support remain distinct; no causal, joint-chain-probability, or live-operational claim is made.

## `final_test_calibration_curve.pdf` -> `final_test_calibration_curve_enhanced.pdf`

- Old metric: baseline calibration
- New authoritative metric: Brier=.016398
- Recommended section: Locked-test results
- Proposed caption: Equal-frequency calibration of the full-enhanced model on the locked test set; points compare mean predicted probability with observed propagation frequency.
- Interpretation: The figure reports predictive performance or descriptive associations for the final enhanced model.
- Warning: Validation selection, locked-test evaluation, flight classification, graph-chain evaluation, and historical decision support remain distinct; no causal, joint-chain-probability, or live-operational claim is made.

## `shap_grouped_importance_compact.pdf` -> `shap_grouped_importance_enhanced.pdf`

- Old metric: baseline SHAP
- New authoritative metric: 24-feature grouped SHAP
- Recommended section: Explainability
- Proposed caption: Grouped global mean absolute SHAP importance for 5,000 validation rotations (seed=42). One-hot levels are aggregated to their original feature groups; SHAP values are expressed in model raw-score units.
- Interpretation: The figure reports predictive performance or descriptive associations for the final enhanced model.
- Warning: Validation selection, locked-test evaluation, flight classification, graph-chain evaluation, and historical decision support remain distinct; no causal, joint-chain-probability, or live-operational claim is made.

## `shap_operational_dependence_panels.pdf` -> `shap_operational_dependence_enhanced.pdf`

- Old metric: baseline SHAP
- New authoritative metric: enhanced raw-score SHAP
- Recommended section: Explainability
- Proposed caption: Full-enhanced SHAP dependence patterns for (a) previous-delay ratio, (b) turn buffer, (c) previous arrival delay, and (d) planned turnaround. Values describe associations with the raw model score and are neither causal effects nor probability-point changes.
- Interpretation: The figure reports predictive performance or descriptive associations for the final enhanced model.
- Warning: Validation selection, locked-test evaluation, flight classification, graph-chain evaluation, and historical decision support remain distinct; no causal, joint-chain-probability, or live-operational claim is made.

## `locked_test_error_profiles.pdf` -> `locked_test_error_profiles_enhanced.pdf`

- Old metric: TP=41301; FP=13744; FN=7737
- New authoritative metric: TP=41898; FP=12160; FN=7140
- Recommended section: Error analysis
- Proposed caption: Locked-test operational feature distributions for true positives, false positives, and false negatives under the frozen tau=0.47 threshold. Boxes show medians and interquartile ranges; whiskers exclude plotted outliers for compactness.
- Interpretation: The figure reports predictive performance or descriptive associations for the final enhanced model.
- Warning: Validation selection, locked-test evaluation, flight classification, graph-chain evaluation, and historical decision support remain distinct; no causal, joint-chain-probability, or live-operational claim is made.

## `propagation_exposure_network_paper.pdf` -> `propagation_exposure_network_enhanced.pdf`

- Old metric: baseline validation graph
- New authoritative metric: enhanced validation graph
- Recommended section: Graph analysis
- Proposed caption: Validation-period propagation exposure from full-enhanced alerted rotation edges. Airport nodes are spatial aggregates and route lines are not recorded trajectories or joint chain probabilities.
- Interpretation: The figure reports predictive performance or descriptive associations for the final enhanced model.
- Warning: Validation selection, locked-test evaluation, flight classification, graph-chain evaluation, and historical decision support remain distinct; no causal, joint-chain-probability, or live-operational claim is made.

## `no direct predecessor` -> `graph_chain_evaluation_enhanced.pdf`

- Old metric: baseline graph summaries
- New authoritative metric: chain-start F1=.7089; exact=.8896; MAE=.1230
- Recommended section: Graph analysis
- Proposed caption: Locked-test graph evaluation for the full-enhanced system: (a) physical, eligible, and scored edge counts and (b) chain-start precision, recall, F1, and exact matched chain-length rate.
- Interpretation: The figure reports predictive performance or descriptive associations for the final enhanced model.
- Warning: Validation selection, locked-test evaluation, flight classification, graph-chain evaluation, and historical decision support remain distinct; no causal, joint-chain-probability, or live-operational claim is made.

## `priority_validation_comparison.pdf` -> `priority_validation_enhanced.pdf`

- Old metric: old priority rates
- New authoritative metric: P1=87.63/90.21; P2=63.84/69.29; P3=21.79/28.28; P4=.33/.50
- Recommended section: Decision support
- Proposed caption: Observed propagation rates and mean predicted probabilities across historical validation decision-priority tiers. The analysis is a retrospective decision-support demonstration, not live operational validation.
- Interpretation: The figure reports predictive performance or descriptive associations for the final enhanced model.
- Warning: Validation selection, locked-test evaluation, flight classification, graph-chain evaluation, and historical decision support remain distinct; no causal, joint-chain-probability, or live-operational claim is made.

## Additional map warning

Eight non-CONUS airport codes (BQN, GUM, PPG, PSE, SJU, SPN, STT, STX) were excluded
from the continental-US exposure map. Their underlying graph edges were not removed from model
evaluation; this exclusion affects only geographic display.
