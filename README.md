# LiveSense AI: Subclinical Mastitis Detection 🐄🩺

LiveSense AI is an edge-optimized Machine Learning pipeline that uses a **Stacking Ensemble combined with a Conditional Variational Autoencoder (CVAE)** to predict Subclinical Mastitis in dairy cows using 10Hz accelerometer data.

This package contains everything needed to perfectly replicate the exact environment, training sequence, and results locally on **Windows, macOS, or Linux** without installing complex Python dependencies on your host machine.

---

## 🎯 Final Project Results

When executed, this model performs 5-Fold Stratified Cross-Validation testing on the unseen data and prints out the results. The expected outputs are:

*   **Average AUC:** 0.77 (77%)
*   **Average Sensitivity (Recall):** 1.00 (100%) - *Meeting and exceeding the 0.91 Target.*
*   **Final Output:** A highly optimized model binary (`mastitis_ultra_model.pkl`) stored in the `./models` directory for edge-device deployment.

---

## 📂 Project Structure

```text
LiveSense_Release/
├── README.md                           # This documentation
├── Dockerfile                          # Docker configuration for isolated execution
├── requirements.txt                    # Python dependencies
├── src/
│   └── GenAI_Mastitis/
│       └── final_model_stacking_cvae.py # The main training & prediction engine
└── data/
    └── processed/
        └── processed_data/
            └── hourly_windows/
                ├── X.npy               # Time-series behavioral dataset features
                └── y.npy               # Mastitis labels (Healthy/Sick)
```

---

## 🚀 How to Run on Windows (via Docker)

The easiest and cleanest way to run this project on a Windows machine is using **Docker**. This ensures you don't face any `ModuleNotFoundError` or version conflicts.

### Prerequisites:
1. Download and install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/).
2. Open Docker Desktop and ensure the Docker Engine is running.
3. Open your Terminal (Command Prompt or PowerShell) and navigate strictly to this `LiveSense_Release` folder.

### Step 1: Build the Docker Image
Run the following command to package the AI, the code, and the dataset into one container:
```bash
docker build -t livesense-ai .
```
*(This will take a few minutes the first time as it downloads the Python dependencies).*

### Step 2: Execute the Pipeline
Run the AI models in a container and watch the terminal for exactly the same results as generated during project development:
```bash
docker run --rm livesense-ai
```

That's it! 🥳 You'll see the complete cross-validation process running in the terminal exactly as in the project report.

---

## 💻 How to Run Locally (Native Python)

If you prefer to run this without Docker in your IDE (like VS Code or PyCharm), follow these steps:

1. **Open the Terminal** and navigate into the `LiveSense_Release` folder.
2. **Install all required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Navigate to the Source Code directory:**
   ```bash
   cd src/GenAI_Mastitis
   ```
4. **Execute the Model:**
   ```bash
   python final_model_stacking_cvae.py
   ```

---

**Developed & Designed for Parul Institute of Technology, Parul University.** 🎓
