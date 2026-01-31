 # Monte Carlo Pricing of Exotic Basket Options with Knock-Out Barriers

This project prices exotic basket options with knock-out features using a Monte Carlo simulation framework. The model calibrates a local volatility surface from synthetic market data for multiple underlying assets and evaluates knock-out options by simulating correlated asset paths.

---

## 🔍 Objective

To build an accurate and efficient pricing model for exotic multi-asset derivatives by:

- Calibrating a local volatility surface from synthetic market data.
- Simulating correlated asset paths under the calibrated model.
- Pricing basket options with knock-out features using Monte Carlo.
  
---

## 🚀 Methodology

- **Volatility Calibration:**: Calibrate local volatility surfaces from synthetic option prices using Dupire's PDE (solved via Crank-Nicolson).
- **Path Simulation:**: Generate correlated asset paths under local vol using Euler-Maruyama and Cholesky.
- **Barrier Handling:**: Apply knock-out condition dynamically during simulation.
- **Pricing:**: Average discounted terminal payoffs over surviving paths, using variance reduction (antithetic variates, adaptive time steps).

---

## 📊 Inputs

- **Calibration Data**: Market prices of plain vanilla options across different strikes and maturities.
- **Basket Option Data**: Each entry contains strike, maturity, type (call/put), and knock-out barrier.

---


## ✅ Key Results

- Constructed a local volatility surface with an average pricing error **below 2.3%** against synthetic market prices.
- Efficiently priced exotic basket options across multiple scenarios with **correlated asset paths** and **barrier knockouts**.
- Reduced pricing variance via **antithetic variates** and **adaptive time discretization**.

---


## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/Exotic-Option-Pricing-using-Monte-Carlo-Simulation.git
   cd Exotic-Option-Pricing-using-Monte-Carlo-Simulation
   ```
2. Install dependencies:
   ```bash
    pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
    python Exotic_option_pricing.py
   ```

## 🧠 Author
  Aarya Yogesh Pakhale
  Quantitative Finance Enthusiast | Dual Degree @ IIT Kharagpur
