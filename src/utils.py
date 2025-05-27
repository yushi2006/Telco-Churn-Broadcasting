import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def draw_box_plot(data: pd.DataFrame, y: str, save_path: str):
    sns.boxplot(data=data, y=y)
    plt.title(f"IQR of {y}")
    plt.ylabel(y)
    plt.savefig(save_path)

    plt.show()

def draw_dist_and_kde(data: pd.DataFrame, x: str, save_path: str):

    sns.histplot(data[x], bins=20, kde=True)
    plt.xlabel(x)
    plt.title(f"Distribution of {x}")
    plt.savefig(save_path)

    plt.show()

def draw_count_plot_to_churn(data: pd.DataFrame, x: str, save_path: str):
    sns.countplot(data=data, x=x, hue="Churn")
    plt.xlabel(x)
    plt.ylabel("Count")
    plt.title(f"{x} and Churn")
    plt.savefig(save_path)
    plt.show()

def draw_corr(data: pd.DataFrame, services_and_churn: list[str], save_path: str):
    # first we make those columns int64 with the values 0 and 1.
    for col in services_and_churn:
        data[col] = data[col].map({"Yes": True, "No": False, "No internet service": False})

    data[services_and_churn] = data[services_and_churn].astype(int)

    # then we compute the corelation matrix
    corr_matrix = data[services_and_churn].corr()

    # Now we do a heatmap plot of the relationships.
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation Coefficient"}
    )

    plt.title("Correlation Heatmap of Yes/No Features", fontsize=16, weight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)

    # Remove the top and right spines
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def visualize_imbalance_churn(data: pd.DataFrame):
    data['Churn'].value_counts(normalize=True).plot(kind='bar')
    plt.xlabel("Churn")
    plt.ylabel("Percentage")
    plt.title("Churn")
    plt.savefig("../figures/churn_percentage.png")
    plt.show()