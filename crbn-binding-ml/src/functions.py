import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

def scatter_plot_fn(y_test, y_pred, method = ""):
    plt.scatter(y_test, y_pred)

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        label = "Perfect prediction"
    )

    slope, intercept = np.polyfit(y_test, y_pred, 1)
    x_line = np.array([y_test.min(), y_test.max()])
    y_line = slope * x_line + intercept

    plt.plot(
        x_line,
        y_line,
        label = "Linear regression"
    )

    r, p = pearsonr(y_test, y_pred)

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Scatter Plot of Model {method}")

    plt.text(
        0.05, 0.95,
        f"r = {r:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top"
    )
    plt.legend()
    plt.show()
