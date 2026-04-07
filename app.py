import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from linear_regression_manual import ManualLinearRegression
from linear_regression_sklearn import SklearnLinearRegression


def generate_data(n=50, noise=5):
    X = np.linspace(0, 10, n)
    y = 2 * X + 3 + np.random.randn(n) * noise
    return X, y


st.set_page_config(page_title="Linear Regression Visualizer", layout="wide")
st.title("Linear Regression Learning Visualizer")


with st.sidebar:
    st.header("Controls")

    alpha = st.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)
    epochs = st.slider("Epochs", 10, 500, 100, step=10)
    noise = st.slider("Noise", 0.0, 10.0, 3.0, step=0.5)

    st.divider()
    st.subheader("Manual Line Control")
    m_slider = st.slider("Slope (m)", -5.0, 5.0, 1.0)
    b_slider = st.slider("Intercept (b)", -10.0, 10.0, 0.0)


X, y = generate_data(noise=noise)


manual = ManualLinearRegression()
sk_model = SklearnLinearRegression()

manual.train(X, y, alpha=alpha, epochs=epochs)
sk_model.train(X, y)

y_manual = manual.predict(X)
y_sklearn = sk_model.predict(X)

y_manual_slider = m_slider * X + b_slider


m_manual, b_manual = manual.get_params()
m_sklearn, b_sklearn = sk_model.get_params()

mse_manual = np.mean((y - y_manual) ** 2)
mse_sklearn = np.mean((y - y_sklearn) ** 2)
mse_slider = np.mean((y - y_manual_slider) ** 2)


col1, col2 = st.columns([2, 1])


with col1:
    st.subheader("Data + Model Fit")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(X, y, label="Data", alpha=0.6)

    ax.plot(X, y_manual, label="Gradient Descent", linewidth=2)
    ax.plot(X, y_sklearn, label="Sklearn (Optimal)", linestyle="--")
    ax.plot(X, y_manual_slider, label="Manual Control", linestyle=":")

    # Residuals (for manual GD)
    for i in range(len(X)):
        ax.plot([X[i], X[i]], [y[i], y_manual[i]], linestyle="dotted")

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.grid(True)

    st.pyplot(fig)


with col2:
    st.subheader("Parameters")

    st.write("**Gradient Descent**")
    st.write(f"m: {m_manual:.4f}")
    st.write(f"b: {b_manual:.4f}")
    st.write(f"MSE: {mse_manual:.4f}")

    st.write("---")

    st.write("**Sklearn (Optimal)**")
    st.write(f"m: {m_sklearn:.4f}")
    st.write(f"b: {b_sklearn:.4f}")
    st.write(f"MSE: {mse_sklearn:.4f}")

    st.write("---")

    st.write("**Manual Slider Line**")
    st.write(f"MSE: {mse_slider:.4f}")

    st.metric("Slope Difference", f"{abs(m_manual - m_sklearn):.6f}")


st.subheader("Loss vs Iterations (Convergence)")

loss_history = manual.get_loss_history()
st.line_chart(loss_history)