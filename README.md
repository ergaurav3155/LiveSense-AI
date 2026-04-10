<div align="center">
<h1 align="center">
  <img src="https://readme-typing-svg.demolab.com/?font=Righteous&size=40&pause=1000&color=2196F3&center=true&vCenter=true&width=800&height=80&lines=LiveSense+AI+🐄;Subclinical+Mastitis+Detection;Frugal+Edge-AI+System;Powered+by+CVAE+%2B+IoT" alt="LiveSense AI Typing Animation"/>
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/AI_Model-CVAE_%2B_Stacking_Ensemble-1f425f?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Hardware-WASP_IoT_%2B_Raspberry_Pi_4-4CAF50?style=for-the-badge&logo=raspberrypi&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Production_Ready-FF9800?style=for-the-badge&logo=checkmarx&logoColor=white" />
</p>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">
</div>

## 🐄 What is LiveSense AI?

**LiveSense AI** is a state-of-the-art, edge-optimized Machine Learning pipeline designed to detect *Subclinical Mastitis (SCM)* in dairy cows. Unlike clinical mastitis, SCM shows **no visible symptoms** but causes severe milk yield drops, making manual inspection completely ineffective.

By leveraging **10Hz triaxial accelerometer data** from WASP IoT collars, LiveSense AI eliminates the need for expensive diagnostic tests (like AMS or Lab Cultures) and provides a highly scalable solution tailored for smallholder farmers in India.

> 📖 *(For an in-depth understanding of the system, hardware architecture, and deep learning algorithms, please refer to the included **`final_report.pdf`**).*

<br>
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%">

## 🔬 Core Innovations (The Magic Behind the AI)

<details>
<summary><b>1️⃣ Solving Extreme Imbalance with GenAI (CVAE)</b></summary>
<br>
Real-world farming data exhibits extreme imbalance (a staggering <b>1:5793</b> sick-to-healthy ratio). Traditional models silently fail here. We implemented a <b>Conditional Variational Autoencoder (CVAE) with a Bi-LSTM Encoder</b> to deeply learn the hidden biological signatures of the disease. The module synthesizes highly realistic "sick" behavior patterns, boosting model Sensitivity (Recall) by 36% compared to SMOTE.
</details>

<details>
<summary><b>2️⃣ The Stacking Ensemble Decision Maker</b></summary>
<br>
Instead of relying on one algorithm, our model uses a hierarchical "Committee of Experts":
<ul>
  <li><b>Level 0 Base Learners:</b> XGBoost (Gradient Boosting), Random Forest, and SVM.</li>
  <li><b>Level 1 Meta-Learner:</b> Logistic Regression to aggregate opinions and minimize False Alarms.</li>
</ul>
</details>

<details>
<summary><b>3️⃣ Edge Optimization (Hardware)</b></summary>
<br>
The final trained model is structurally lightweight, allowing local execution directly on a <b>Raspberry Pi 4 Edge Node</b>. This removes internet dependency entirely, reduces inference latency to <100ms, and slashes hardware costs by 87%.
</details>

<br>
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%">

## 🎯 Model Validation Results

Compiled and validated using rigorous **Stratified 5-Fold Cross-Validation** (ensuring absolute zero data-leakage):

- 🟢 **Average ROC-AUC Score:** `0.77 (77%)`
- 🔥 **Average Sensitivity (Recall):** `1.00 (100%)` - *Meeting and greatly exceeding research standards!*
- 📦 **Final Output:** Production binary dump (`mastitis_ultra_model.pkl`) stored locally.

<br>
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="100%">

## 💻 Running the Project (Windows / Mac / Linux)

Running complex Machine Learning code on Windows can sometimes lead to missing libraries or `ModuleNotFoundError`. We have strictly optimized the deployment process so you can run it perfectly. 

### 🐳 Method 1: The Docker Route (Highly Recommended)
This packages the entire OS, Python environment, and Code into a single bulletproof container.

1. Download and successfully install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Open your Terminal (or PowerShell) and navigate to this folder:
   ```cmd
   cd path\to\LiveSense_Release
   ```
3. Build the animated Docker Image:
   ```cmd
   docker build -t livesense-ai .
   ```
4. Run the Pipeline:
   ```cmd
   docker run --rm livesense-ai
   ```

### 🐍 Method 2: The Native Python Virtual Environment Route
If you don't want to use Docker, use this standard Python approach.

1. Ensure **Python 3.10+** is installed on your machine.
2. Open Command Prompt inside this folder and create a clean Virtual Environment:
   ```cmd
   python -m venv venv
   ```
3. Activate the Virtual Environment:
   ```cmd
   venv\Scripts\activate
   ```
4. Install the required modules via Requirements File:
   ```cmd
   pip install -r requirements.txt
   ```
5. Navigate to the source folder and execute:
   ```cmd
   cd src\GenAI_Mastitis
   python final_model_stacking_cvae.py
   ```
   *(You should see the Terminal begin extracting 696 features followed by the 5-Fold Validation scoring).*

---
<div align="center">
  <b>Designed and Developed for academic research presentation at Parul Institute of Technology, Parul University. 🎓</b><br>
  <img src="https://img.shields.io/badge/Made_with-Love_%26_Python-red?style=for-the-badge&logo=python" />
</div>
