pipeline {
    agent any

    environment {
        DOCKER_IMAGE   = 'modelpulse:latest'
        CONTAINER_NAME = 'modelpulse-container'
        APP_PORT       = '5000'
    }

    stages {
        stage('Stage 1: Checkout SCM') {
            steps {
                echo 'Checking out source code from Git repository...'
                checkout scm
            }
        }

        stage('Stage 2: Build & Retrain ML Model (Docker)') {
            steps {
                echo 'Building Docker container image and triggering automated PyCaret model retraining...'
                sh '''
                    # Ensure logs directory exists on host
                    mkdir -p logs
                    
                    # Build container image (triggers RUN python train.py inside Dockerfile)
                    docker build -t ${DOCKER_IMAGE} .
                '''
            }
        }

        stage('Stage 3: Deploy Container') {
            steps {
                echo 'Deploying containerized ModelPulse service...'
                sh '''
                    # Stop and remove previous container instance if exists
                    if [ $(docker ps -aq -f name=^/${CONTAINER_NAME}$) ]; then
                        echo "Stopping and removing existing container: ${CONTAINER_NAME}"
                        docker stop ${CONTAINER_NAME} || true
                        docker rm -f ${CONTAINER_NAME} || true
                    fi

                    # Run new container with persistent logs volume and restart policy
                    echo "Starting new container: ${CONTAINER_NAME} on port ${APP_PORT}..."
                    docker run -d \
                        -p ${APP_PORT}:${APP_PORT} \
                        -v "$(pwd)/logs:/app/logs" \
                        --restart always \
                        --name ${CONTAINER_NAME} \
                        ${DOCKER_IMAGE}
                '''
            }
        }

        stage('Stage 4: API Smoke Test & Log Verification') {
            steps {
                echo 'Performing health checks, API smoke tests, and log verification...'
                sh '''
                    # Wait for Gunicorn server to initialize
                    echo "Waiting 6 seconds for container service to be ready..."
                    sleep 6

                    # 1. Health check endpoint test
                    echo "Running Health Check (GET /health)..."
                    curl -f -s http://localhost:${APP_PORT}/health | grep -q "healthy"
                    echo "Health check passed!"

                    # 2. Automated Smoke Test via REST API (POST /predict_api - Single Record)
                    echo "Running REST API Prediction Smoke Test (POST /predict_api)..."
                    PRED_RESPONSE=$(curl -f -s -X POST http://localhost:${APP_PORT}/predict_api \
                        -H "Content-Type: application/json" \
                        -d '{"age": 28, "sex": "male", "bmi": 24.5, "children": 1, "smoker": "no", "region": "south"}')
                    
                    echo "API Response: ${PRED_RESPONSE}"
                    echo "${PRED_RESPONSE}" | grep -q "prediction"
                    echo "Single prediction smoke test passed!"

                    # 3. Batch Prediction Smoke Test (POST /predict_api - Array)
                    echo "Running Batch Prediction Smoke Test..."
                    BATCH_RESPONSE=$(curl -f -s -X POST http://localhost:${APP_PORT}/predict_api \
                        -H "Content-Type: application/json" \
                        -d '[{"age": 25, "sex": "female", "bmi": 21.8, "children": 0, "smoker": "no", "region": "north"}, {"age": 52, "sex": "male", "bmi": 29.4, "children": 2, "smoker": "yes", "region": "west"}]')
                    echo "${BATCH_RESPONSE}" | grep -q "batch_size"
                    echo "Batch prediction smoke test passed!"

                    # 4. CSV Batch Upload Smoke Test (POST /predict_csv)
                    echo "Running CSV Batch Upload Smoke Test (POST /predict_csv)..."
                    CSV_RESPONSE=$(curl -f -s -X POST http://localhost:${APP_PORT}/predict_csv \
                        -H "Content-Type: text/plain" \
                        --data-binary $'age,sex,bmi,children,smoker,region\n28,male,24.5,1,no,west\n45,female,31.2,2,yes,south')
                    echo "${CSV_RESPONSE}" | grep -q "total_rows"
                    echo "CSV batch smoke test passed!"

                    # 5. Metrics & History Endpoints Verification
                    echo "Verifying /api/metrics and /api/history..."
                    curl -f -s http://localhost:${APP_PORT}/api/metrics | grep -q "total_predictions"
                    curl -f -s http://localhost:${APP_PORT}/api/history | grep -q "predictions"
                    echo "Metrics and History APIs verified!"

                    # Copy container logs to host if needed and run monitoring script
                    docker cp ${CONTAINER_NAME}:/app/logs/. ./logs/ || true

                    # 6. Telemetry & Log Monitoring verification (Unit 6.2 & 9.4)
                    echo "Executing monitor_logs.py to summarize inference telemetry..."
                    if command -v python3 &> /dev/null; then
                        python3 monitor_logs.py
                    else
                        docker exec ${CONTAINER_NAME} python monitor_logs.py
                    fi
                '''
            }
        }
    }

    post {
        always {
            echo 'Archiving ModelPulse telemetry and training logs...'
            archiveArtifacts artifacts: 'logs/*.log', allowEmptyArchive: true
        }
        success {
            echo "=========================================================================="
            echo "  SUCCESS: ModelPulse MLOps Pipeline deployed and validated successfully! "
            echo "  Web UI & REST API available at http://<EC2-PUBLIC-IP>:${APP_PORT}       "
            echo "=========================================================================="
        }
        failure {
            echo "Pipeline failed! Fetching recent container logs for troubleshooting..."
            sh "docker logs ${CONTAINER_NAME} --tail 50 || true"
        }
    }
}
