# definitions for different types of PSTHs
import pandas as pd
import numpy as np

event_definitions_biasedCW = {
    "inter_trial": {
        "align_event": "stimOn_times",
        "window": (-1.15, 1),
        "query": "intervals_0 > 0",
        "label": r"$\mathrm{rest}$",
    },
    "blockL": {
        "align_event": "stimOn_times",
        "window": (-0.4, 0.1),
        "query": "block_left",
        "label": r"$\mathrm{L_b}$",
    },
    "blockR": {
        "align_event": "stimOn_times",
        "window": (-0.4, 0.1),
        "query": "block_right",
        "label": r"$\mathrm{R_b}$",
    },
    "block50": {
        "align_event": "stimOn_times",
        "window": (-0.4, 0.1),
        "query": "block_center",
        "label": r"$\mathrm{50_b}$",
    },
    "quiescence": {
        "align_event": "stimOn_times",
        "window": (-0.4, 0.1),
        "query": "intervals_0 > 0",
        "label": r"$\mathrm{quies}$",
    },
    "stimLbLcL": {
        "align_event": "stimOn_times",
        "window": (0, 0.2),
        "query": "stim_left & choice_left & block_left & deep_block",
        "label": r"$\mathrm{L_sL_cL_b, s}$",
    },
    "stimLbRcL": {
        "align_event": "stimOn_times",
        "window": (0, 0.2),
        "query": "stim_left & choice_left & block_right & deep_block",
        "label": r"$\mathrm{L_sL_cR_b, s}$",
    },
    "stimLbRcR": {
        "align_event": "stimOn_times",
        "window": (0, 0.2),
        "query": "stim_left & choice_right & block_right & deep_block",
        "label": r"$\mathrm{L_sR_cR_b, s}$",
    },
    "stimLbLcR": {
        "align_event": "stimOn_times",
        "window": (0, 0.2),
        "query": "stim_left & choice_right & block_left & deep_block",
        "label": r"$\mathrm{L_sR_cL_b, s}$",
    },
    "stimRbLcL": {
        "align_event": "stimOn_times",
        "window": (0, 0.2),
        "query": "stim_right & choice_left & block_left & deep_block",
        "label": r"$\mathrm{R_sL_cL_b, s}$",
    },
    "stimRbRcL": {
        "align_event": "stimOn_times",
        "window": (0, 0.2),
        "query": "stim_right & choice_left & block_right & deep_block",
        "label": r"$\mathrm{R_sL_cR_b, s}$",
    },
    "stimRbRcR": {
        "align_event": "stimOn_times",
        "window": (0, 0.2),
        "query": "stim_right & choice_right  & block_right & deep_block",
        "label": r"$\mathrm{R_sR_cR_b, s}$",
    },
    "stimRbLcR": {
        "align_event": "stimOn_times",
        "window": (0, 0.2),
        "query": "stim_right & choice_right & block_left & deep_block",
        "label": r"$\mathrm{R_sR_cL_b, s}$",
    },
    "motor_init": {
        "align_event": "firstMovement_times",
        "window": (-0.15, 0),
        "query": "intervals_0 > 0",
        "label": r"$\mathrm{m}$",
    },
    "sLbLchoiceL": {
        "align_event": "firstMovement_times",
        "window": (-0.15, 0),
        "query": "stim_left & choice_left & block_left",
        "label": r"$\mathrm{L_sL_cL_b, m}$",
    },
    "sLbRchoiceL": {
        "align_event": "firstMovement_times",
        "window": (-0.15, 0),
        "query": "stim_left & choice_left & block_right",
        "label": r"$\mathrm{L_sL_cR_b, m}$",
    },
    "sLbRchoiceR": {
        "align_event": "firstMovement_times",
        "window": (-0.15, 0),
        "query": "stim_left & choice_right & block_right",
        "label": r"$\mathrm{L_sR_cR_b, m}$",
    },
    "sLbLchoiceR": {
        "align_event": "firstMovement_times",
        "window": (-0.15, 0),
        "query": "stim_left & choice_right & block_left",
        "label": r"$\mathrm{L_sR_cL_b, m}$",
    },
    "sRbLchoiceL": {
        "align_event": "firstMovement_times",
        "window": (-0.15, 0),
        "query": "stim_right & choice_left & block_left",
        "label": r"$\mathrm{R_sL_cL_b, m}$",
    },
    "sRbRchoiceL": {
        "align_event": "firstMovement_times",
        "window": (-0.15, 0),
        "query": "stim_right & choice_left & block_right",
        "label": r"$\mathrm{R_sL_cR_b, m}$",
    },
    "sRbRchoiceR": {
        "align_event": "firstMovement_times",
        "window": (-0.15, 0),
        "query": "stim_right & choice_right & block_right",
        "label": r"$\mathrm{R_sR_cR_b, m}$",
    },
    "sRbLchoiceR": {
        "align_event": "firstMovement_times",
        "window": (-0.15, 0),
        "query": "stim_right & choice_right & block_left",
        "label": r"$\mathrm{R_sR_cL_b, m}$",
    },
    "choiceL": {
        "align_event": "firstMovement_times",
        "window": (0, 0.15),
        "query": "choice_left",
        "label": r"$\mathrm{L_{move}}$",
    },
    "choiceR": {
        "align_event": "firstMovement_times",
        "window": (0, 0.15),
        "query": "choice_right",
        "label": r"$\mathrm{R_{move}}$",
    },
    "fback1": {
        "align_event": "feedback_times",
        "window": (0, 0.3),
        "query": "feedbackType == 1.0",
        "label": r"$\mathrm{feedbk1}$",
    },
    "fback0": {
        "align_event": "feedback_times",
        "window": (0, 0.3),
        "query": "feedbackType == -1.0",
        "label": r"$\mathrm{feedbk0}$",
    },
}

event_definitions_trainingCW = {
    "inter_trial": {
        "align_event": "stimOn_times",
        "window": (-1.15, 1),
        "query": "intervals_0 > 0",
        "label": r"$\mathrm{rest}$",
    },
    "quiescence": {
        "align_event": "stimOn_times",
        "window": (-0.4, 0.1),
        "query": "intervals_0 > 0",
        "label": r"$\mathrm{quies}$",
    },
    "stimLcL": {
        "align_event": "stimOn_times",
        "window": (0, 0.2),
        "query": "stim_left & choice_left",
        "label": r"$\mathrm{L_sL_cL_b, s}$",
    },
    "stimLcR": {
        "align_event": "stimOn_times",
        "window": (0, 0.2),
        "query": "stim_left & choice_right",
        "label": r"$\mathrm{L_sR_cR_b, s}$",
    },
    "stimRcL": {
        "align_event": "stimOn_times",
        "window": (0, 0.2),
        "query": "stim_right & choice_left",
        "label": r"$\mathrm{R_sL_cL_b, s}$",
    },
    "stimRcR": {
        "align_event": "stimOn_times",
        "window": (0, 0.2),
        "query": "stim_right & choice_right",
        "label": r"$\mathrm{R_sR_cR_b, s}$",
    },
    "motor_init": {
        "align_event": "firstMovement_times",
        "window": (-0.15, 0),
        "query": "intervals_0 > 0",
        "label": r"$\mathrm{m}$",
    },
    "sLchoiceL": {
        "align_event": "firstMovement_times",
        "window": (-0.15, 0),
        "query": "stim_left & choice_left",
        "label": r"$\mathrm{L_sL_cL_b, m}$",
    },
    "sLchoiceR": {
        "align_event": "firstMovement_times",
        "window": (-0.15, 0),
        "query": "stim_left & choice_right",
        "label": r"$\mathrm{L_sR_cR_b, m}$",
    },
    "sRchoiceL": {
        "align_event": "firstMovement_times",
        "window": (-0.15, 0),
        "query": "stim_right & choice_left",
        "label": r"$\mathrm{R_sL_cL_b, m}$",
    },
    "sRchoiceR": {
        "align_event": "firstMovement_times",
        "window": (-0.15, 0),
        "query": "stim_right & choice_right",
        "label": r"$\mathrm{R_sR_cR_b, m}$",
    },
    "choiceL": {
        "align_event": "firstMovement_times",
        "window": (0, 0.15),
        "query": "choice_left",
        "label": r"$\mathrm{L_{move}}$",
    },
    "choiceR": {
        "align_event": "firstMovement_times",
        "window": (0, 0.15),
        "query": "choice_right",
        "label": r"$\mathrm{R_{move}}$",
    },
    "fback1": {
        "align_event": "feedback_times",
        "window": (0, 0.3),
        "query": "feedbackType == 1.0",
        "label": r"$\mathrm{feedbk1}$",
    },
    "fback0": {
        "align_event": "feedback_times",
        "window": (0, 0.3),
        "query": "feedbackType == -1.0",
        "label": r"$\mathrm{feedbk0}$",
    },
}


def add_info_to_trials_table(trials_df: pd.DataFrame, N: int = 3) -> pd.DataFrame:
    trials_df = _add_categories_to_trials_table(trials_df)
    trials_df = _add_deep_in_block(trials_df, N=N)
    return trials_df


def _add_categories_to_trials_table(trials_df: pd.DataFrame) -> pd.DataFrame:
    trials_df["stim_left"] = trials_df["contrastLeft"].apply(lambda x: True if x > -1 else False)
    trials_df["stim_right"] = trials_df["contrastRight"].apply(lambda x: True if x > -1 else False)
    trials_df["choice_left"] = trials_df["choice"].apply(lambda x: True if x == -1 else False)
    trials_df["choice_right"] = trials_df["choice"].apply(lambda x: True if x == 1 else False)
    trials_df["block_left"] = trials_df["probabilityLeft"].apply(lambda x: True if x == 0.8 else False)
    trials_df["block_right"] = trials_df["probabilityLeft"].apply(lambda x: True if x == 0.2 else False)
    trials_df["block_center"] = trials_df["probabilityLeft"].apply(lambda x: True if x == 0.5 else False)
    trials_df["deep_block"] = True
    return trials_df


def _add_deep_in_block(
    trials_df: pd.DataFrame,
    N: int = 3,
) -> pd.DataFrame:
    trials_df["deep_block"] = False
    block_switches = np.diff(trials_df["probabilityLeft"]) != 0
    block_switches = np.insert(block_switches, 0, False)
    block_switch_ix = np.where(block_switches)[0]
    for i, block in enumerate(np.split(trials_df, block_switch_ix)):
        # block index could potentially also be useful eventually
        trials_df.loc[block.index, "block_ix"] = i
        trials_df.loc[block.index[N:], "deep_block"] = True
    return trials_df
