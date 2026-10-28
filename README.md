# ⚡ ModelPulse — Automated MLOps Retraining, Monitoring & Deployment Pipeline

[![Docker](https://img.shields.io/badge/Docker-24.0+-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Jenkins](https://img.shields.io/badge/Jenkins-CI%2FCD-D24939?logo=jenkins&logoColor=white)](https://www.jenkins.io/)
[![PyCaret](https://img.shields.io/badge/PyCaret-3.3.2-FF6F00?logo=python&logoColor=white)](https://pycaret.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2%20Ubuntu-232F3E?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/ec2/)

**ModelPulse** is an end-to-end DevOps & MLOps reference platform designed for university laboratory requirements. It demonstrates how to containerize, automate retraining, continuously test, deploy, and monitor a machine learning inference service using modern DevOps tooling across local development machines and AWS cloud infrastructure.

---

## 📋 DevOps Syllabus Alignment

| Lab Unit | Curriculum Topic | ModelPulse Implementation |
| :--- | :--- | :--- |
| **Unit 1** | **Docker Containerization** | `Dockerfile` (Python 3.10-slim + Gunicorn) & `docker-compose.yml` (multi-container orchestration, persistent volume mounts, and automated health checks). |
| **Unit 4** | **Jenkins CI/CD Automation** | Declarative `Jenkinsfile` automating SCM checkout, Docker container build with ML model retraining, container redeployment, and automated smoke testing. |
| **Unit 6** | **Automation Scripting** | `aws_setup.sh` (Bash automation for Ubuntu EC2 package provisioning) and `monitor_logs.py` (Python telemetry and anomaly log analytics). |
| **Unit 9 & 10** | **MLOps with PyCaret & Logging** | Automated model training (`train.py`), Flask Web UI & REST API (`app.py`), structured JSON inference logging (`logs/predictions.log`), and model retraining logging (`logs/model_training.log`). |

---

## 🏗️ Architecture & Pipeline Flow

```
+-----------------------------------------------------------------------------------+
|                               MODELPULSE ARCHITECTURE                             |
+-----------------------------------------------------------------------------------+
                                        
   [Developer / Git Push] 
             │
             ▼
   [AWS EC2: Jenkins CI/CD Engine] ──────────────────────────────────────────┐
             │                                                               │
             ├─► Stage 1: Checkout SCM                                       │
             │                                                               │
             ├─► Stage 2: Docker Build & Model Retraining                    │
             │            └─► Executes `train.py`                            │
             │                ├─► PyCaret loads dataset & trains LR model    │
             │                ├─► Logs metrics to `logs/model_training.log`  │
             │                └─► Exports `deployment_28042020.pkl`          │
             │                                                               │
             ├─► Stage 3: Zero-Downtime Container Deployment                 │
             │            └─► `docker run` / `docker compose`                │
             │                └─► Mounts `./logs:/app/logs`                  │
             │                                                               │
             └─► Stage 4: Smoke Testing & Telemetry Audit                    │
                          ├─► GET  /health (Status check)                    │
                          ├─► POST /predict_api (JSON inference test)        │
                          └─► `python3 monitor_logs.py` (Telemetry report)   │
                                                                             │
             ┌───────────────────────────────────────────────────────────────┘
             ▼
   [Live Containerized Microservice (Port 5000)]
   ├─► Modern Glassmorphic Web UI  (GET / & POST /predict)
   ├─► RESTful Prediction Endpoint (POST /predict_api)
   ├─► Health Check Endpoint       (GET /health)
   └─► Structured Telemetry Stream (logs/predictions.log)
```

---

## 📁 Repository Structure

```
pycaret-deployment-aws/
├── data/
│   └── insurance.csv          # Local sample insurance dataset (offline training fallback)
├── logs/                      # Persistent logs directory (mounted in container)
│   ├── model_training.log     # Records retraining timestamp, metrics, duration
│   └── predictions.log        # Records every inference call, input features, latency
├── static/
│   └── style.css              # Neo-Brutalist design system (3px black borders, hard offset shadows, cyber palette)
├── templates/
│   └── home.html              # Multi-tab Neo-Brutalist Web UI with presets, telemetry & canvas analytics
├── .dockerignore              # Excludes unnecessary build context files
├── Dockerfile                 # Multi-stage production container specification
├── docker-compose.yml         # Container orchestration with volume mounts
├── Jenkinsfile                # Declarative 4-stage automated CI/CD pipeline
├── app.py                     # Flask Web application & REST API
├── train.py                   # PyCaret automated training & serialization pipeline
├── monitor_logs.py            # CLI telemetry analyzer & anomaly monitor
├── aws_setup.sh               # Automated bash installation script for Ubuntu EC2
├── requirements.txt           # Python 3.10 compatible dependencies
└── README.md                  # Comprehensive lab documentation
```

---

## 🚀 Local Execution Guide

### Option 1: Native Python Virtual Environment

```bash
# 1. Navigate to the project directory
cd pycaret-deployment-aws

# 2. Create and activate a Python 3.10 virtual environment
# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# 3. Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Train and finalize the PyCaret regression model
python train.py

# 5. Start the Flask application
python app.py
```
> Open your browser at **`http://localhost:5000`** to access the ModelPulse Web UI.

---

### Option 2: Docker & Docker Compose (Recommended)

Make sure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is running.

```bash
# Build and launch container in the background
docker compose up --build -d

# Check running status and healthcheck
docker compose ps

# View real-time application logs
docker compose logs -f

# Stop container services
docker compose down
```

---

## 📡 API Testing & Smoke Tests

### 1. Health Check
```bash
curl -X GET http://localhost:5000/health
```
**Expected Response:**
```json
{
  "model": "deployment_28042020",
  "model_loaded": true,
  "service": "ModelPulse-MLOps",
  "status": "healthy",
  "timestamp": "2026-10-27T14:15:00.123456+00:00"
}
```

### 2. REST API Prediction Request (Single & Batch)
**Single Record:**
```bash
curl -X POST http://localhost:5000/predict_api \
  -H "Content-Type: application/json" \
  -d '{
    "age": 28,
    "sex": "male",
    "bmi": 24.5,
    "children": 1,
    "smoker": "no",
    "region": "west"
  }'
```

**Batch JSON Inference:**
```bash
curl -X POST http://localhost:5000/predict_api \
  -H "Content-Type: application/json" \
  -d '[
    {"age": 25, "sex": "female", "bmi": 21.8, "children": 0, "smoker": "no", "region": "north"},
    {"age": 52, "sex": "male", "bmi": 29.4, "children": 2, "smoker": "yes", "region": "south"}
  ]'
```

**Batch CSV Upload & Scoring:**
```bash
curl -X POST http://localhost:5000/predict_csv \
  -H "Content-Type: text/plain" \
  --data-binary @data/sample_insurance_india.csv
```

### 3. Real-Time Telemetry & Metrics Endpoints
- **Metrics Endpoint:** `curl http://localhost:5000/api/metrics`
  - Returns `total_predictions`, `avg_latency_ms`, `max_latency_ms`, `p95`, `error_rate`, and uptime.
- **Prediction History:** `curl http://localhost:5000/api/history?limit=25`
  - Returns recent inference calls with inputs and latency measurements.

### 4. Run Telemetry Monitoring Script
```bash
python monitor_logs.py --summary      # Single terminal report
python monitor_logs.py --watch 10     # Live refreshing dashboard every 10s
```
Outputs an ANSI-colored console dashboard summarizing total predictions, latency percentiles (P50/P95/P99), error rates, endpoint breakdown, and model retraining history.

---

## ☁️ AWS EC2 Deployment Guide (Step-by-Step)

### Step 1: Launch an AWS EC2 Ubuntu Instance
1. Log in to the **AWS Management Console** and open the **EC2 Dashboard**.
2. Click **Launch Instance** and configure:
   - **Name:** `ModelPulse-DevOps-Lab`
   - **AMI:** `Ubuntu Server 22.04 LTS (HVM), SSD Volume Type`
   - **Instance Type:** `t2.medium` or `t3.medium` *(Recommended: 2 vCPUs, 4GB RAM required for smooth PyCaret model training & Jenkins)*.
   - **Key Pair:** Select or create an SSH key pair (e.g. `modelpulse-key.pem`).
   - **Storage:** Minimum `20 GiB gp3` root volume.

### Step 2: Configure EC2 Security Group
Under **Network settings**, create or edit a Security Group with these **Inbound Rules**:

| Type | Protocol | Port Range | Source | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **SSH** | TCP | `22` | `My IP` (or `0.0.0.0/0`) | Remote SSH terminal access |
| **Custom TCP** | TCP | `8080` | `0.0.0.0/0` | Jenkins CI/CD Web Management Console |
| **Custom TCP** | TCP | `5000` | `0.0.0.0/0` | ModelPulse Flask Web UI & REST API |

---

### Step 3: Run the Automated Setup Script on EC2
1. Connect to your instance via SSH:
   ```bash
   ssh -i "modelpulse-key.pem" ubuntu@<YOUR-EC2-PUBLIC-IP>
   ```

2. Clone your project repository (or transfer your files):
   ```bash
   git clone <YOUR-GITHUB-REPO-URL>
   cd pycaret-deployment-aws
   ```

3. Make the automated setup script executable and run it:
   ```bash
   chmod +x aws_setup.sh
   sudo bash aws_setup.sh
   ```
   *The script automatically installs Docker, Docker Compose, Java 17, and Jenkins, adds the `ubuntu` and `jenkins` users to the `docker` group, enables system services, and displays your initial Jenkins administrator password.*

---

### Step 4: Configure the Jenkins CI/CD Pipeline
1. Open your web browser and navigate to:
   ```
   http://<YOUR-EC2-PUBLIC-IP>:8080
   ```
2. **Unlock Jenkins:** Paste the initial admin password printed by `aws_setup.sh` (or run `sudo cat /var/lib/jenkins/secrets/initialAdminPassword`).
3. Click **Install suggested plugins** and create your Jenkins admin account.
4. From the Jenkins Dashboard:
   - Click **New Item**.
   - Enter Item Name: `ModelPulse-Pipeline`.
   - Select **Pipeline** and click **OK**.
5. In the Pipeline configuration page:
   - Scroll down to the **Pipeline** section.
   - Set **Definition** to **Pipeline script from SCM**.
   - Set **SCM** to **Git**.
   - Enter your repository URL: `https://github.com/<YOUR-USERNAME>/<YOUR-REPO>.git`.
   - Specify Branch: `*/main` (or `*/master`).
   - Ensure **Script Path** is set to `Jenkinsfile`.
   - Click **Save**.

---

### Step 5: Trigger Build & Verify Deployment
1. On the pipeline dashboard, click **Build Now**.
2. Observe all 4 stages execute:
   - **Stage 1 (Checkout SCM):** Pulls fresh source code.
   - **Stage 2 (Build & Retrain):** Packages container and automatically retrains the PyCaret regression model.
   - **Stage 3 (Deploy Container):** Runs `modelpulse-container` with port forwarding on `5000:5000` and volume mounted `./logs`.
   - **Stage 4 (Smoke Test & Log Verification):** Validates `/health`, tests `/predict_api`, and executes `monitor_logs.py`.
3. Access your live application at:
   ```
   http://<YOUR-EC2-PUBLIC-IP>:5000
   ```

---

## 🛠️ Troubleshooting & Lab FAQs

### Q1: `The term 'docker' is not recognized` on Windows?
- This means Docker Desktop is not installed or not added to your Windows user `PATH`.
- Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/), enable the WSL 2 backend, and restart your PowerShell terminal.
- Alternatively, test locally using the native Python virtual environment (`python app.py`), or deploy directly to the AWS EC2 Ubuntu instance using `aws_setup.sh`.

### Q2: Jenkins build fails with `permission denied while trying to connect to the Docker daemon socket`?
- Run `sudo usermod -aG docker jenkins && sudo systemctl restart jenkins` on the EC2 instance to ensure Jenkins has access to `/var/run/docker.sock`.

### Q3: Model retraining times out on EC2?
- Ensure your EC2 instance is at least `t2.medium` or `t3.medium` (4GB RAM). If using `t2.micro`, create a swap file:
  ```bash
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  ```

---

## 📜 License
Developed for educational DevOps and MLOps laboratory demonstrations. Open-sourced under the MIT License.
