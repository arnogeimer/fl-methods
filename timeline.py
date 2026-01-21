import matplotlib.pyplot as plt
import numpy as np

def generate_timeline():
    plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "cmr10"],
    "axes.unicode_minus": False,
    })

    # -----------------------------
    # Timeline range
    # -----------------------------
    years = np.arange(2011, 2026)

    # Example background values (replace with your data)
    # e.g. citations per year or normalized counts
    background_values = np.array([
       0, 0, 0, 0, 0, 6, 45, 128, 391, 1144, 2283, 3796, 5711, 7391, 8378
    ])

    # -----------------------------
    # Key events on the timeline
    # -----------------------------
    events = [
        (2012.5, "GDPR\n proposed"),
        (2016 + 1/6, "Federated Learning\n(McMahan et al.)"),
        (2016.5, "GDPR\n comes into force"),
        (2020, "CCPA\n comes into force"),
        (2020.6, "Flower framework\nreleased"),
        (2024.66, "EU AI Act references\n privacy-preserving ML"),
    ]

    # -----------------------------
    # Figure setup
    # -----------------------------
    fig, ax = plt.subplots(figsize=(7, 3))

    # -----------------------------
    # Background bar plot
    # -----------------------------
    ax.bar(
        years,
        background_values,
        width=0.6,
        color="darkgreen",
        alpha=0.3,
        zorder=1,
        label = "McMahan et al. (2016) citation count"
    )

    # -----------------------------
    # Timeline baseline
    # -----------------------------

    # -----------------------------
    # Event markers + annotations
    # -----------------------------
    offset = 0.19 * max(background_values)
    direction = -1  # alternate up/down

    for year, label in events:
        # Dot on the timeline
        ax.plot(
            year,
            0,
            "o",
            color="black",
            markersize=4,
            zorder=4
        )

        if year == 2024.66:
            direction *= -1

        # Vertical connector
        ax.vlines(
            year,
            0,
            direction * offset,
            alpha=0.7,
            color="black",
            linewidth=0.8,
            linestyles="dashed",
            zorder=3
        )

        # Annotation
        ax.text(
            year,
            direction * (offset + 10),
            label,
            ha="center",
            va="bottom" if direction > 0 else "top",
            fontsize=10,
            zorder=4
        )

        direction *= -1  # alternate direction

    # -----------------------------
    # Axes formatting
    # -----------------------------
    ax.set_xlim(2011.5, 2025.5)
    ax.set_ylim(
        -0.4 * max(background_values),
        1.1 * max(background_values)
    )

    ax.set_xticks(np.arange(2012, 2026, 1))
    ax.set_yticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")

    # Remove spines for a clean academic look
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    ax.spines["bottom"].set_position(("data", 0))

    # -----------------------------
    # Layout & export
    # -----------------------------
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("./plots/fl_timeline_with_citations.pdf")
    plt.close()

generate_timeline()