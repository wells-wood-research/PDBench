"""Plots similar to ones in visualization.py, but separate. Might be useful when writing dissertation. Not tested yet, not used anywhere else."""

import pandas as pd
from benchmark import config
import ampal
from benchmark import get_cath
import gzip
from pathlib import Path
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn import metrics
from bechmark.visualization import append_zero_residues


def compare_model_accuracy(models: list, name: str, model_labels=list):
    # plot maximum 9 models, otherwise the plot is a complete mess
    minimum = 0
    maximum = 0
    if len(models) > 8:
        models = models[0:9]
    colors = sns.color_palette()
    # combine 4 and 6 to make plots nicer. Works with any number of CATH classes.
    class_key = [x[0] for x in models[0].index]
    class_key = list(dict.fromkeys(class_key))
    if 4 in class_key and 6 in class_key:
        class_key = [x for x in class_key if x != 4 and x != 6]
        class_key.append([4, 6])
    ratios = [models[0].loc[class_key[i]].shape[0] for i in range(len(class_key))]
    fig, ax = plt.subplots(
        3,
        len(class_key),
        figsize=(12 * len(class_key), 15),
        gridspec_kw={"width_ratios": ratios},
        squeeze=False,
    )
    for i in range(len(class_key)):
        index = np.arange(0, models[0].loc[class_key[i]].shape[0])
        for j, frame in enumerate(models):
            value_accuracy = frame.loc[class_key[i]].accuracy.values
            value_recall = frame.loc[class_key[i]].recall.values
            ax[0][i].bar(
                x=index + j * 0.1,
                height=value_accuracy,
                width=0.1,
                align="center",
                color=colors[j],
                label=model_labels[j],
            )
            for e, accuracy in enumerate(value_accuracy):
                ax[0][i].text(
                    index[e] + j * 0.1,
                    accuracy + 0.3,
                    f"{accuracy:.3f}",
                    ha="center",
                    va="bottom",
                    rotation="vertical",
                    fontdict={"size": 7},
                )
            for e, recall in enumerate(value_recall):
                ax[1][i].text(
                    index[e] + j * 0.1,
                    recall * 1.2,
                    f"{recall:.3f}",
                    ha="center",
                    va="bottom",
                    rotation="vertical",
                    fontdict={"size": 7},
                )
            # show top3 accuracy if it exists
            if "top3_accuracy" in frame:
                value_top_three = frame.loc[class_key[i]].top3_accuracy.values
                ax[0][i].scatter(
                    x=index + j * 0.1,
                    y=value_top_three,
                    marker="_",
                    s=50,
                    color=colors[j],
                )
                ax[0][i].vlines(
                    x=index + j * 0.1,
                    ymin=0,
                    ymax=value_top_three,
                    color=colors[j],
                    linewidth=2,
                )
            ax[1][i].bar(
                x=index + j * 0.1,
                height=value_recall,
                width=0.1,
                align="center",
                color=colors[j],
            )
            difference = value_accuracy - value_recall
            if np.amin(difference) < minimum:
                minimum = np.amin(difference)
            if np.amax(difference) > maximum:
                maximum = np.amax(difference)
            ax[2][i].bar(
                x=index + j * 0.1,
                height=difference,
                width=0.1,
                align="center",
                color=colors[j],
            )
            for e, dif in enumerate(difference):
                if dif < 0:
                    y_coord = 0
                else:
                    y_coord = dif
                ax[2][i].text(
                    index[e] + j * 0.1,
                    y_coord + 0.01,
                    f"{dif:.3f}",
                    ha="center",
                    va="bottom",
                    rotation="vertical",
                    fontdict={"size": 7},
                )

        # Title, Label, Ticks and Ylim
        ax[0][i].set_title(config.classes[i + 1], fontdict={"size": 22})
        ax[1][i].set_title(config.classes[i + 1], fontdict={"size": 22})
        ax[2][i].set_title(config.classes[i + 1], fontdict={"size": 22})
        ax[0][i].set_ylabel("Accuracy")
        ax[1][i].set_ylabel("Recall")
        ax[2][i].set_ylabel("Accuracy-Recall")
        ax[0][i].set_xticks(index)
        ax[0][i].set_xticklabels(
            frame.loc[class_key[i]].name,
            rotation=90,
            fontdict={"horizontalalignment": "center", "size": 12},
        )
        ax[0][i].set_ylim(0, 1)
        ax[0][i].set_xlim(-0.3, index[-1] + 1)
        ax[1][i].set_xticks(index)
        ax[1][i].set_xticklabels(
            frame.loc[class_key[i]].name,
            rotation=90,
            fontdict={"horizontalalignment": "center", "size": 12},
        )
        ax[1][i].set_ylim(0, 1)
        ax[1][i].set_xlim(-0.3, index[-1] + 1)
        ax[2][i].set_xticks(index)
        ax[2][i].set_xticklabels(
            frame.loc[class_key[i]].name,
            rotation=90,
            fontdict={"horizontalalignment": "center", "size": 12},
        )
        ax[2][i].hlines(0, -0.3, index[-1] + 1, colors="k", lw=1)
        ax[2][i].set_xlim(-0.3, index[-1] + 1)
    # scale axis so that they are equal to get nice graph
    for x in range(len(ax[2])):
        ax[2][x].set_ylim(minimum * 1.2, maximum * 1.2)
    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=7, prop={"size": 8})
    fig.tight_layout()
    fig.subplots_adjust(right=0.94)
    fig.savefig(name, dpi=400)


def compare_secondary_structures(model_dicts: list, name: str, model_labels=list):
    if len(model_dicts) > 8:
        model_dicts = model_dicts[0:9]
    colors = sns.color_palette()
    maximum = 0
    minimum = 0
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    keys = ["alpha", "beta", "loops", "random"]
    index = np.array([0, 1, 2, 3])
    for j, model in enumerate(model_dicts):
        # show accuracy
        value_accuracy = [model[k] for k in keys]
        ax[0].bar(
            x=index + j * 0.1,
            height=value_accuracy,
            width=0.1,
            align="center",
            color=colors[j],
            label=model_labels[j],
        )
        # show top three accuracy
        if "alpha_three" in model:
            value = [model[k + "_three"] for k in keys]
            ax[0].scatter(x=index + j * 0.1, y=value, marker="_", s=50, color=colors[j])
            ax[0].vlines(
                x=index + j * 0.1, ymin=0, ymax=value, color=colors[j], linewidth=2
            )
        # show recall
        value_recall = [model[k + "_recall"] for k in keys]
        ax[1].bar(
            x=index + j * 0.1,
            height=value_recall,
            width=0.1,
            align="center",
            color=colors[j],
        )
        for e, accuracy in enumerate(value_accuracy):
            ax[0].text(
                index[e] + j * 0.1,
                accuracy + 0.3,
                f"{accuracy:.3f}",
                ha="center",
                va="bottom",
                rotation="vertical",
                fontdict={"size": 7},
            )
        for e, recall in enumerate(value_recall):
            ax[1].text(
                index[e] + j * 0.1,
                recall * 1.2,
                f"{recall:.3f}",
                ha="center",
                va="bottom",
                rotation="vertical",
                fontdict={"size": 7},
            )
        # show difference
        difference = np.array(value_accuracy) - np.array(value_recall)
        if np.amin(difference) < minimum:
            minimum = np.amin(difference)
        if np.amax(difference) > maximum:
            maximum = np.amax(difference)
        ax[2].bar(
            x=index + j * 0.1,
            height=difference,
            width=0.1,
            align="center",
            color=colors[j],
        )
        for e, dif in enumerate(difference):
            if dif < 0:
                y_coord = 0
            else:
                y_coord = dif
            ax[2].text(
                index[e] + j * 0.1,
                y_coord + 0.01,
                f"{dif:.3f}",
                ha="center",
                va="bottom",
                rotation="vertical",
                fontdict={"size": 7},
            )
        # Title, Label, Ticks and Ylim
        fig.suptitle("Secondary structure", fontdict={"size": 22})
        ax[0].set_ylabel("Accuracy")
        ax[0].set_xticks(index)
        ax[0].set_xticklabels(
            ["Helices", "Sheets", "Structured loops", "Random"],
            rotation=90,
            fontdict={"horizontalalignment": "center", "size": 12},
        )
        ax[0].set_ylim(0, 1)
        ax[0].set_xlim(-0.3, index[-1] + 1)

        ax[1].set_ylabel("Recall")
        ax[1].set_xticks(index)
        ax[1].set_xticklabels(
            ["Helices", "Sheets", "Structured loops", "Random"],
            rotation=90,
            fontdict={"horizontalalignment": "center", "size": 12},
        )
        ax[1].set_ylim(0, 1)
        ax[1].set_xlim(-0.3, index[-1] + 1)

        ax[2].set_ylabel("Accuracy-Recall")
        ax[2].set_xticks(index)
        ax[2].set_xticklabels(
            ["Helices", "Sheets", "Structured loops", "Random"],
            rotation=90,
            fontdict={"horizontalalignment": "center", "size": 12},
        )
        ax[2].set_xlim(-0.3, index[-1] + 1)
        ax[2].axhline(0, -0.3, index[-1] + 1, color="k", lw=1)
        ax[2].set_ylim(minimum * 1.2, maximum * 1.2)
    fig.legend(loc=7, prop={"size": 7})
    plt.tight_layout()
    fig.subplots_adjust(right=0.86)
    fig.savefig(fig.savefig(name + ".pdf"))


def plot_resolution(
    df: pd.DataFrame,
    predictions: dict,
    name: str,
    by_fragment: bool = True,
    ignore_uncommon=False,
    score_sequence=False,
):
    colors = sns.color_palette("viridis", 4)
    # combine class 4 and 6 to simplify the graph
    colors = {1: colors[0], 2: colors[1], 3: colors[2], 4: colors[3], 6: colors[3]}
    class_color = [colors[x] for x in df["class"].values]
    accuracy, recall = get_cath.score_each(
        df, predictions, by_fragment, ignore_uncommon, score_sequence
    )
    resolution = get_cath.get_resolution(df)
    corr = pd.DataFrame({0: resolution, 1: recall, 2: accuracy}).corr().to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=[10, 5])
    ax[0].scatter(accuracy, resolution, color=class_color, alpha=0.7)
    ax[0].set_ylabel("Resolution, A")
    ax[0].set_xlabel("Accuracy")
    ax[0].set_title(f"Pearson correlation: {corr[0][2]:.3f}")
    ax[1].scatter(recall, resolution, color=class_color, alpha=0.7)
    ax[1].set_title(f"Pearson correlation: {corr[0][1]:.3f}")
    ax[1].set_xlabel("Recall")
    patches = [
        mpatches.Patch(color=colors[x], label=config.classes[x]) for x in config.classes
    ]
    fig.legend(loc=1, handles=patches, prop={"size": 9})
    fig.tight_layout()
    fig.subplots_adjust(right=0.87)
    fig.savefig(name + ".pdf")


def residue_plot(
    model,
    predictions,
    name,
    by_fragment=True,
    ignore_uncommon=False,
    score_sequence=False,
):
    fig, ax = plt.subplots(2, 5, figsize=(25, 10))
    (
        sequence,
        prediction,
        dssp,
        true_secondary,
        prediction_secondary,
    ) = get_cath.format_sequence(
        model,
        predictions,
        ignore_uncommon=ignore_uncommon,
        score_sequence=score_sequence,
        by_fragment=by_fragment,
    )
    if not score_sequence:
        prediction = list(get_cath.most_likely_sequence(prediction))
        prediction_secondary = [
            list(get_cath.most_likely_sequence(ss_seq))
            for ss_seq in prediction_secondary
        ]

    seq = append_zero_residues(np.unique(sequence, return_counts=True))

    pred = append_zero_residues(np.unique(prediction, return_counts=True))
    index = np.arange(len(seq[0]))
    ax[0][4].bar(
        x=index, height=seq[1], width=0.4, label="True sequence", align="center"
    )
    ax[0][4].bar(
        x=index + 0.4, height=pred[1], width=0.4, label="Prediction", align="center"
    )
    ax[0][4].set_xticks(index)
    ax[0][4].set_xticklabels(
        pred[0], fontdict={"horizontalalignment": "center", "size": 12}
    )
    ax[0][4].set_ylabel("Amino acid count")
    ax[0][4].set_xlabel("Amino acids")
    ax[0][4].set_title("All structures")
    ax[0][4].legend()

    cm = metrics.confusion_matrix(sequence, prediction, labels=seq[0])
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax[1][4] = sns.heatmap(
        cm, xticklabels=seq[0], yticklabels=seq[0], square=True, cmap="viridis"
    )

    ss_names = ["Helices", "Sheets", "Structured loops", "Random"]
    for i, ss in enumerate(ss_names):
        seq = append_zero_residues(np.unique(true_secondary[i], return_counts=True))
        pred = append_zero_residues(
            np.unique(prediction_secondary[i], return_counts=True)
        )
        ax[0][i].bar(
            x=index, height=seq[1], width=0.4, label="True sequence", align="center"
        )
        ax[0][i].bar(
            x=index + 0.4, height=pred[1], width=0.4, label="Prediction", align="center"
        )
        ax[0][i].set_xticks(index)
        ax[0][i].set_xticklabels(
            pred[0], fontdict={"horizontalalignment": "center", "size": 12}
        )
        ax[0][i].set_ylabel("Amino acid count")
        ax[0][i].set_xlabel("Amino acids")
        ax[0][i].set_title(ss)

        cm = metrics.confusion_matrix(
            true_secondary[i], prediction_secondary[i], labels=seq[0]
        )
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        plt.sca(ax[1][i])
        sns.heatmap(
            cm, xticklabels=seq[0], yticklabels=seq[0], square=True, cmap="viridis"
        )

    fig.suptitle(name, fontdict={"size": 22})
    plt.tight_layout()
    fig.savefig(name + ".pdf")
