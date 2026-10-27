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

                    # 2. Automated Smoke Test via REST API (POST /predict_api)
                    echo "Running REST API Prediction Smoke Test (POST /predict_api)..."
                    PRED_RESPONSE=$(curl -f -s -X POST http://localhost:${APP_PORT}/predict_api \
                        -H "Content-Type: application/json" \
                        -d '{"age": 28, "sex": "male", "bmi": 33.0, "children": 3, "smoker": "no", "region": "southeast"}')
                    
                    echo "API Response: ${PRED_RESPONSE}"
                    echo "${PRED_RESPONSE}" | grep -q "prediction"
                    echo "Prediction smoke test passed!"

                    # Copy container logs to host if needed and run monitoring script
                    docker cp ${CONTAINER_NAME}:/app/logs/. ./logs/ || true

                    # 3. Telemetry & Log Monitoring verification (Unit 6.2 & 9.4)
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
