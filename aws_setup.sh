#!/usr/bin/env bash
# ==============================================================================
# ModelPulse — AWS Ubuntu EC2 Automated Provisioning Script (Unit 6.1)
# Installs: Docker, Docker Compose, OpenJDK 17, and Jenkins CI/CD Server
# Target OS: Ubuntu 20.04 / 22.04 / 24.04 LTS (AWS EC2 t2.medium or t3.medium)
# ==============================================================================

set -euo pipefail

# ANSI color formatting
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
BOLD="\033[1m"
NC="\033[0m" # No Color

echo -e "\n${BOLD}${CYAN}========================================================================${NC}"
echo -e "${BOLD}${CYAN}    ⚡ MODELPULSE — AWS EC2 AUTOMATED DEVOPS PROVISIONING SCRIPT ⚡      ${NC}"
echo -e "${BOLD}${CYAN}========================================================================${NC}\n"

# Verify script is run as root or with sudo
if [[ "$EUID" -ne 0 ]]; then
   echo -e "${RED}[ERROR] This setup script must be run as root or with sudo.${NC}"
   echo "Usage: sudo bash aws_setup.sh"
   exit 1
fi

# Check RAM and automatically allocate 4GB swap if running on t2.micro (<2GB RAM)
TOTAL_RAM_MB=$(free -m | awk '/^Mem:/{print $2}')
if [ "$TOTAL_RAM_MB" -lt 2500 ]; then
    echo -e "${YELLOW}>>> Low memory detected (${TOTAL_RAM_MB}MB). Configuring 4GB Swap file to prevent Jenkins/Docker OOM crashes...${NC}"
    if ! swapon --show | grep -q "/swapfile"; then
        if [ ! -f /swapfile ]; then
            fallocate -l 4G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=4096
            chmod 600 /swapfile
            mkswap /swapfile
        fi
        swapon /swapfile || true
        if ! grep -q "/swapfile" /etc/fstab; then
            echo '/swapfile none swap sw 0 0' >> /etc/fstab
        fi
        echo -e "${GREEN}[OK] 4GB Swap successfully configured and activated!${NC}"
    fi
fi

echo -e "${YELLOW}>>> Step 1/6: Updating system packages and installing prerequisites...${NC}"
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    python3 \
    python3-pip \
    python3-venv \
    software-properties-common \
    iptables

echo -e "\n${YELLOW}>>> Step 2/6: Installing Java 17 OpenJDK (Jenkins Runtime)...${NC}"
apt-get install -y openjdk-17-jdk openjdk-17-jre-headless
java -version

echo -e "\n${YELLOW}>>> Step 3/6: Installing Docker Engine & Docker Compose Plugin...${NC}"
# Remove any obsolete packages
apt-get remove -y docker docker-engine docker.io containerd runc || true

# Add official Docker GPG key & repository
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg --yes
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Enable and start Docker service
systemctl start docker
systemctl enable docker
echo -e "${GREEN}[OK] Docker installed successfully:${NC} $(docker --version)"

echo -e "\n${YELLOW}>>> Step 4/6: Installing Jenkins CI/CD Automation Server...${NC}"
# Add official Jenkins repository keyring
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key | tee \
  /usr/share/keyrings/jenkins-keyring.asc > /dev/null

echo deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] \
  https://pkg.jenkins.io/debian-stable binary/ | tee \
  /etc/apt/sources.list.d/jenkins.list > /dev/null

apt-get update -y
apt-get install -y jenkins

# Enable and start Jenkins
systemctl start jenkins
systemctl enable jenkins
echo -e "${GREEN}[OK] Jenkins service started and enabled.${NC}"

echo -e "\n${YELLOW}>>> Step 5/6: Configuring Permissions & Docker Group Membership...${NC}"
# Add 'ubuntu' user and 'jenkins' service user to 'docker' group
groupadd -f docker
usermod -aG docker ubuntu || true
usermod -aG docker jenkins || true

# Restart Jenkins so group changes take effect
systemctl restart jenkins
echo -e "${GREEN}[OK] Added 'ubuntu' and 'jenkins' users to 'docker' group.${NC}"

echo -e "\n${YELLOW}>>> Step 6/6: Verifying Installed Services...${NC}"
# Wait for initial password generation
echo "Waiting for Jenkins initial admin token generation..."
for i in {1..15}; do
    if [ -f /var/lib/jenkins/secrets/initialAdminPassword ]; then
        break
    fi
    sleep 2
done

# Fetch EC2 Public IP if on AWS metadata service v2/v1
PUBLIC_IP=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || hostname -I | awk '{print $1}')

echo -e "\n${BOLD}${GREEN}========================================================================${NC}"
echo -e "${BOLD}${GREEN}        🎉 MODELPULSE DEVOPS ENVIRONMENT SETUP COMPLETE! 🎉           ${NC}"
echo -e "${BOLD}${GREEN}========================================================================${NC}"

if [ -f /var/lib/jenkins/secrets/initialAdminPassword ]; then
    ADMIN_PWD=$(cat /var/lib/jenkins/secrets/initialAdminPassword)
    echo -e "\n${BOLD}Jenkins Initial Admin Password:${NC}"
    echo -e "${CYAN}${BOLD}${ADMIN_PWD}${NC}\n"
fi

echo -e "${BOLD}Next Steps for Deployment:${NC}"
echo -e " 1. Access Jenkins Web Console at:   ${CYAN}http://${PUBLIC_IP}:8080${NC}"
echo -e " 2. Paste the Initial Admin Password shown above and install suggested plugins."
echo -e " 3. Create a Pipeline Job pointing to your ModelPulse Git repository."
echo -e " 4. Ensure AWS Security Group allows inbound traffic for:"
echo -e "      • Port 22   (SSH Access)"
echo -e "      • Port 8080 (Jenkins CI/CD Dashboard)"
echo -e "      • Port 5000 (ModelPulse Flask Web UI & REST API)"
echo -e " 5. After the pipeline runs, access ModelPulse at: ${GREEN}http://${PUBLIC_IP}:5000${NC}\n"
