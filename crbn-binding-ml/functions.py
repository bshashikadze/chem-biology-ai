import matplotlib.pyplot as plt

def scatter_plot_fn(y_test, y_pred, method = ""):
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Scatter Plot of Model {method}")
    plt.show()
