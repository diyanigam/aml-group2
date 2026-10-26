# Fourier Neural Operator (FNO) — 1D Wave Simulation

A beginner-friendly guide and implementation of the **Fourier Neural Operator (FNO)** for predicting 1D wave physics.

---

## 1. What Are We Trying to Do?

Imagine you are watching waves in water. If you take a picture of the wave right now, can an AI model predict what the wave will look like one second from now?

* **Input:** The wave shape right now: $u(x, t)$
* **Target:** The wave shape at the next moment: $u(x, t + \Delta t)$

We are solving a physical wave equation (the **Burgers' Equation**), which describes how waves move, deform, and interact over time.

---

## 2. What is FNO and Why is it Special?

### The Problem with Normal Neural Networks (like CNNs):
A normal CNN looks at the wave like a row of pixels (e.g., exactly 128 points). If you give it a smoother, higher-resolution wave with 256 or 512 points, the CNN gets confused because its input shape is locked.

### The FNO Advantage (Thinking in Frequencies):
Think of a musical chord. You can describe a sound by listing every air pressure sample, or you can describe it by its musical notes (frequencies). 

FNO converts the wave into **frequencies (Fourier space)** using the Fast Fourier Transform (FFT):
1. **Low frequencies** capture the big, overall shape of the wave.
2. **High frequencies** capture the sharp details and steep edges.

Because frequencies are continuous physical properties that don't depend on how many pixels you use to draw them, **FNO has a superpower called Zero-Shot Super-Resolution**:
> We train the model on a coarse grid (**128 points**), but it can directly predict on fine, high-definition grids (**256 and 512 points**) without any retraining!

---

## 3. How the Model is Built

Here are the key settings explained in simple terms:

| Setting | Value | What does it mean? |
| :--- | :--- | :--- |
| **Fourier Modes** | `64` | The model keeps the 64 most important low-frequency wave patterns in each layer. |
| **Width (Channels)** | `64` | The number of hidden features/channels the model uses to think about the wave. |
| **Fourier Layers** | `4` | How many spectral blocks are stacked together. |
| **Activation** | `GELU` | A smooth math function that lets the model learn complex, nonlinear wave curves. |
| **Total Parameters** | `1,073,857` | ~1.07 million learnable weights (roughly the same size as the baseline CNN for a fair comparison). |
| **Loss Function** | Relative $L^2$ | Measures percentage error rather than absolute error, so small and large waves are treated fairly. |

---

## 4. How We Trained the Model

* **The Data:** 
  * We have 48 full wave simulations.
  * **38 simulations** (190 step-pairs) were used for training.
  * **10 simulations** (50 step-pairs) were hidden away as a **validation set** to make sure the model doesn't just memorize the answers.
* **Smart Training Rules:**
  * **Learning Rate Drop:** If the validation error stops improving for 20 epochs, the learning speed is automatically cut in half.
  * **Best Model Saving:** We automatically save the model weights that gave the lowest validation error (which happened at **Epoch 287**).
  * **Early Stopping:** Training automatically stopped at **Epoch 320** because the model was already fully converged.

---

## 5. How Well Does it Perform?

We tested the trained model on 60 brand-new, unseen wave simulations. 

We also compared it to an **"Identity Baseline"** (a simple guess that says: *"the wave won't move at all"*).

| Grid Resolution | FNO Error (Lower is better) | Identity Error | How much better is FNO? |
| :---: | :---: | :---: | :---: |
| **128 points** (Trained on this) | **0.00344** (0.34%) | 0.03253 (3.25%) | **9.46× better** |
| **256 points** (Never seen in training) | **0.00427** (0.43%) | 0.03255 (3.25%) | **7.63× better** |
| **512 points** (Never seen in training) | **0.00449** (0.45%) | 0.03255 (3.25%) | **7.24× better** |

### The Big Result:
Even when we test on **4× higher resolution (512 points)**, the error barely increases from 0.34% to 0.45%. The model learned the true continuous physics, not just a grid of pixels.

---

## 6. What Each File Does

Everything is neatly split into two clean notebooks:

* **`FNO.ipynb`** — **The Training & Testing Notebook**
  * Defines the FNO architecture.
  * Trains the model on 38 simulations.
  * Evaluates zero-shot performance across 128, 256, and 512 points.
  * Saves the best model checkpoint to `fno_modes64_38traj_seed0.pt` and errors to `fno_errors.npz`.

* **`FNO_plots.ipynb`** — **The Plotting Notebook**
  * Reads the saved checkpoint and error file.
  * Saves 3 clean image files:
    1. **`plots_training.png`**: Shows the loss going down epoch-by-epoch and marks the best epoch (287).
    2. **`plots_eval.png`**: Bar charts comparing FNO vs the baseline across all 3 resolutions.
    3. **`plots_outputs.png`**: Side-by-side plots of the input wave, true next wave, and FNO predicted wave.

---

## 7. How to Run

1. **Step 1:** Run all cells in `FNO.ipynb` to train the model and save the evaluation results.
2. **Step 2:** Run all cells in `FNO_plots.ipynb` to generate the plots and visual comparisons.
