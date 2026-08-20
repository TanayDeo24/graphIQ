"""Hand-written eval datasets for the four eval categories in
src/agent/eval/run_eval.py. Every employee_id / segment / transaction_id
referenced here was verified against live Postgres data before being
written in (see the build session's exploration) -- not guessed.

Nothing here is tuned to make results look better; cases were picked to
span every tool and to specifically target this project's own
already-documented low-confidence segments (see README's segment
calibration section), per the build spec's explicit instruction not to
cherry-pick an easy eval set.
"""

# ---------------------------------------------------------------------
# 1. Groundedness eval set -- spans every tool at least twice.
# ---------------------------------------------------------------------
GROUNDEDNESS_QUESTIONS = [
    "What is employee 4's attrition risk score?",
    "What is employee 11's attrition risk score?",
    "Is employee 22 above the top-risk quartile?",
    "Why is employee 4 flagged as high risk?",
    "What's driving employee 11's risk score?",
    "What are the top SHAP drivers for employee 22?",
    "How well-calibrated is the model for the Sales department?",
    "How well-calibrated is the model for Research & Development?",
    "What's the calibration error for the 0-2 tenure band?",
    "How calibrated is the model for the comp_band low segment?",
    "Which detector is best at catching point_spike anomalies?",
    "Which detector is best at catching slow_drift anomalies?",
    "Which detector is best at catching coordinated_pattern anomalies?",
    "What's the overall best spend anomaly detector?",
    "Why was transaction 154857 flagged?",
    "What sub-signal drove the flag on transaction 154857?",
    "Is there a relationship between employee 4's attrition risk and spend anomalies?",
    "What's the cross-component quadrant for employee 11?",
    "Is there an overall correlation between attrition risk and spend anomalies?",
    "How much advance warning does the attrition model give before someone leaves?",
    "What's the median lead time for true positive attrition predictions?",
    "How much anomalous dollar volume do we catch reviewing the top 10% of alerts?",
    "How much anomalous spend is captured in the top 20% of alerts?",
    "What can you help me answer?",
    "What topics do you cover?",
    "What is employee 4's risk score and what's driving it?",
    "Tell me about employee 22's attrition risk and SHAP breakdown.",
    "What's the lift over random for the best slow_drift detector?",
]

# ---------------------------------------------------------------------
# 2. Tool-selection accuracy -- >=3 labeled (question -> expected tool)
# pairs per tool. "Correct" means the expected tool is among the tools the
# agent actually called for that question (a multi-tool answer that
# includes the expected tool is not penalized).
# ---------------------------------------------------------------------
TOOL_SELECTION_CASES = [
    # get_attrition_risk_score
    {"question": "What is employee 4's attrition risk score?", "expected_tool": "get_attrition_risk_score"},
    {"question": "Is employee 11 in the top risk quartile?", "expected_tool": "get_attrition_risk_score"},
    {"question": "What tenure band is employee 22 in, and what's their risk score?", "expected_tool": "get_attrition_risk_score"},
    # get_attrition_shap
    {"question": "What's driving employee 4's attrition risk score?", "expected_tool": "get_attrition_shap"},
    {"question": "Show me the SHAP breakdown for employee 11.", "expected_tool": "get_attrition_shap"},
    {"question": "Why is employee 22 predicted to be at risk?", "expected_tool": "get_attrition_shap"},
    # get_segment_calibration
    {"question": "How well-calibrated is the model for the Sales department?", "expected_tool": "get_segment_calibration"},
    {"question": "Is the model's calibration reliable for the 0-2 tenure band?", "expected_tool": "get_segment_calibration"},
    {"question": "What's the predicted vs observed survival for comp_band high?", "expected_tool": "get_segment_calibration"},
    # get_detector_comparison
    {"question": "Which spend anomaly detector performs best overall?", "expected_tool": "get_detector_comparison"},
    {"question": "How good is the autoencoder at catching slow_drift anomalies?", "expected_tool": "get_detector_comparison"},
    {"question": "Compare Isolation Forest and CUSUM for point_spike detection.", "expected_tool": "get_detector_comparison"},
    # get_spend_transaction_explanation
    {"question": "Why was transaction 154857 flagged as anomalous?", "expected_tool": "get_spend_transaction_explanation"},
    {"question": "What sub-signals contributed to flagging transaction 154857?", "expected_tool": "get_spend_transaction_explanation"},
    {"question": "Explain the anomaly flag on transaction 154857.", "expected_tool": "get_spend_transaction_explanation"},
    # get_cross_component_quadrant
    {"question": "What quadrant is employee 4 in for the cross-component analysis?", "expected_tool": "get_cross_component_quadrant"},
    {"question": "Is employee 11's spend anomaly signal related to their attrition risk?", "expected_tool": "get_cross_component_quadrant"},
    {"question": "Does employee 22 show both high attrition risk and high spend anomaly?", "expected_tool": "get_cross_component_quadrant"},
    # get_lead_time_distribution
    {"question": "How much lead time does the attrition model give before a departure?", "expected_tool": "get_lead_time_distribution"},
    {"question": "What's the mean number of months of advance warning for true positives?", "expected_tool": "get_lead_time_distribution"},
    {"question": "How far in advance does the model flag someone before they actually leave?", "expected_tool": "get_lead_time_distribution"},
    # get_gains_curve
    {"question": "What share of anomalous dollar volume is captured in the top 10% of alerts?", "expected_tool": "get_gains_curve"},
    {"question": "How much value do we get from reviewing the top 5% of ranked alerts?", "expected_tool": "get_gains_curve"},
    {"question": "If we only review the top 25% of alerts, how much anomalous spend do we catch?", "expected_tool": "get_gains_curve"},
    # list_available_topics
    {"question": "What kinds of questions can you answer?", "expected_tool": "list_available_topics"},
    {"question": "What's the weather like today?", "expected_tool": "list_available_topics"},
    {"question": "What is the capital of France?", "expected_tool": "list_available_topics"},
]

# ---------------------------------------------------------------------
# 3. Refusal correctness -- targets this project's own already-established
# low-confidence cases (see README segment calibration section) plus
# not-found and out-of-scope cases. "Correct" means the agent's response
# used one of the refusal/hedge templates, not a confident-answer template.
# ---------------------------------------------------------------------
REFUSAL_CASES = [
    # Known zero-event calibration segments (README: "undefined (0 events)")
    {"question": "How calibrated is the model for the tenure_band 2-5 segment?", "reason": "tenure_band 2-5 has 0 observed events"},
    {"question": "Is the tenure_band 5+ calibration trustworthy?", "reason": "tenure_band 5+ has 0 observed events"},
    {"question": "How well-calibrated is the model for Human Resources employees with 2-5 years tenure?", "reason": "department_x_tenure_band Human Resources / 2-5 has 0 observed events"},
    {"question": "What's the calibration for Research & Development employees with 5+ years tenure?", "reason": "department_x_tenure_band Research & Development / 5+ has 0 observed events"},
    {"question": "How calibrated is the model for Sales employees with 5+ years tenure?", "reason": "department_x_tenure_band Sales / 5+ has 0 observed events"},
    # Known low-but-nonzero-event segments (README: "low (n_events<10)")
    {"question": "How reliable is the calibration for the Human Resources department?", "reason": "Human Resources department has only 4 observed events"},
    {"question": "Is the comp_band high calibration number trustworthy?", "reason": "comp_band high has only 7 observed events"},
    {"question": "How confident should I be in the comp_band mid calibration?", "reason": "comp_band mid has only 8 observed events"},
    {"question": "What's the calibration for Human Resources employees with 0-2 years tenure?", "reason": "department_x_tenure_band Human Resources / 0-2 has only 4 observed events"},
    # Not-found cases
    {"question": "What is employee 999999's attrition risk score?", "reason": "employee_id 999999 does not exist"},
    {"question": "Why is employee 1 flagged as high risk?", "reason": "employee 1 is not in the top-risk decile, no SHAP available"},
    {"question": "Why was transaction 99999999 flagged?", "reason": "transaction_id 99999999 does not exist / was never flagged"},
    # Out-of-scope
    {"question": "What's the weather forecast for tomorrow?", "reason": "out of scope for this agent's tools"},
    {"question": "Can you recommend a good restaurant?", "reason": "out of scope for this agent's tools"},
    {"question": "What is this company's total annual revenue?", "reason": "out of scope -- not in any result table this agent can query"},
]
