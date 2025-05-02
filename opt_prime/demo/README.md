# 🚀 Optimus<sup>Prime</sup> Training Demo Environment Setup & Execution Manual

This document explains how to set up and run the Optimus<sup>Prime</sup> training demonstration environment, designed for distributed deep learning training across multiple nodes.

## 🖥️ Server Configuration

For this demonstration, we're assuming the following server setup:
* Training Servers: 4 machines (s1, s2, s3, s4)
* Demo PC: 1 machine (pc1)

## 📦 Creating Docker Files for Demo Environment

Create a Dockerfile on all nodes in the `~/my_docker/` directory:
* Use "Dockerfile for master" on s1 (master node)
* Use "Dockerfile for worker" on s2, s3, s4 (worker nodes)

```bash
mkdir my_docker
cd my_docker
vi Dockerfile
```

### Dockerfile for master (s1)
```dockerfile
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

# Install essential packages and SSH setup
RUN apt-get update && apt-get install -y \
  net-tools iproute2 inetutils-ping openssh-server \
  vim wget git curl ca-certificates python3-pip sshfs supervisor && \
  echo 'root:****' | chpasswd && \
  sed -i 's/#Port 22/Port 22/' /etc/ssh/sshd_config && \
  sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
  echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
  mkdir -p /var/run/sshd

# Install Python packages
RUN pip install --upgrade pip && \
  pip install torch==2.5.0 --force-reinstall && \
  pip install transformers==4.42.4 datasets jupyterlab tqdm tensorboard tensorboardX
  
# Jupyter configuration
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 80" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py

# Open ports: SSH(22), Jupyter(80), TensorBoard(86)
EXPOSE 22 80 86

# Default command: Start SSH + TensorBoard + JupyterLab
CMD service ssh start && \
    tensorboard --logdir /workspace/runs --host 0.0.0.0 --port 86 & \
    jupyter lab --LabApp.disable_check_xsrf=True --ServerApp.tornado_settings="{'headers': {'Content-Security-Policy': ''}}"
```

### Dockerfile for worker (s[2-4])
```dockerfile
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

# Install essential packages and SSH setup
RUN apt-get update && apt-get install -y \
  net-tools iproute2 inetutils-ping openssh-server \
  vim wget git curl ca-certificates python3-pip sshfs supervisor && \
  echo 'root:****' | chpasswd && \
  sed -i 's/#Port 22/Port 22/' /etc/ssh/sshd_config && \
  sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
  echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
  mkdir -p /var/run/sshd

# Install Python packages
RUN pip install --upgrade pip && \
  pip install torch==2.5.0 --force-reinstall && \
  pip install transformers==4.42.4 datasets tqdm tqdm tensorboard tensorboardX

# Open ports: SSH(22)
EXPOSE 22

# Default command: Start SSH
CMD ["/usr/sbin/sshd", "-D"]
```

## 🛠️ Building Docker Images

Execute on all nodes (s1, s2, s3, s4):

```bash
# (Optional) Clean up existing Docker processes if needed
sudo docker ps  # Check running/stopped containers
sudo docker rm -f etri_demo
sudo docker rmi etri_demo

# Build Docker image
cd ~/my_docker
sudo docker build -t etri_demo .
```

## 🔄 Running Docker on Master Node (s1)

### Installing FUSE (Optional, only if not already installed)
FUSE is required to mount remote log directories from worker nodes to the master node for centralized TensorBoard monitoring.

```bash
sudo apt update
sudo apt install kmod
sudo modprobe fuse
```

## 🚀 Running Master Node Docker (s1)
```bash
sudo docker run -d --gpus all \
    --ipc=host \
    --network=host \
    --cap-add SYS_ADMIN \
    --device /dev/fuse \
    --security-opt apparmor:unconfined \
    --name etri_demo \
    etri_demo
```

## ✅ Command Explanation

| Option | Description |
|------|------|
| `sudo` | Run with root privileges |
| `docker run` | Launch a new container |
| `-d` | **Daemon mode**: Run container in background |
| `--gpus all` | Allocate all GPUs to container (for CUDA/NCCL distributed training) |
| `--ipc=host` | **Inter-process memory sharing settings**<br>Improves performance for PyTorch/NCCL and prevents shared memory errors |
| `--network=host` | **Share host network**<br>Allows container to directly communicate via `localhost`, external IP, etc.<br>Essential for `torchrun`, ssh, tensorboard, etc. |
| `--cap-add SYS_ADMIN` | Grants kernel permissions needed for **FUSE**, `sshfs`, `mount` commands, etc. |
| `--device /dev/fuse` | Enables the use of FUSE-based filesystems within the container (`sshfs`, `encfs`, etc.) |
| `--security-opt apparmor:unconfined` | Disables AppArmor security policy<br>Prevents blocking of FUSE, sshfs, and similar operations |
| `--name etri_demo` | Specify container name |
| `etri_demo` | Docker image name to use |

## 🔄 Running Worker Node Docker Containers (s2, s3, s4)
```bash
sudo docker run -d --gpus all \
    --ipc=host \
    --network=host \
    --name etri_demo \
    etri_demo
```

## 🔑 Master Docker Node Configuration: Copy SSH Keys

Access the master docker node and set up passwordless SSH access to training nodes:
```bash
# Connect to docker (s1)
sudo docker exec -it etri_demo bash

# 1. Generate SSH keys on master node (skip if already exists)
ssh-keygen -t rsa
# Default location: ~/.ssh/id_rsa
# Leave passphrase empty (press Enter)

# 2. Copy public key (specify port) // Password: ****
ssh-copy-id -p 22 root@s2
ssh-copy-id -p 22 root@s3
ssh-copy-id -p 22 root@s4

# 3. Test connections
ssh -p 22 root@s2
ssh -p 22 root@s3
ssh -p 22 root@s4

# Setup complete if you can connect without a password!

# Tip: Save port configuration in ~/.ssh/config (optional)
vi ~/.ssh/config

Host s2
    HostName s2
    Port 22
    User root

# Simple connection:
ssh s2
```

## 📁 Creating Log Directories on Master Node
```bash
mkdir -p /workspace/runs
```

## 📂 SSHFS Directory Mounting

Create log directories on training nodes using remote SSH commands:
```bash
ssh s2 "mkdir -p /workspace/runs"
ssh s3 "mkdir -p /workspace/runs"
ssh s4 "mkdir -p /workspace/runs"
```

Mount training nodes' log directories on the master docker node:
```bash
mkdir -p /workspace/runs/s2
sshfs -o reconnect,allow_other,ServerAliveInterval=15 -p 22 root@s2:/workspace/runs /workspace/runs/s2

mkdir -p /workspace/runs/s3
sshfs -o reconnect,allow_other,ServerAliveInterval=15 -p 22 root@s3:/workspace/runs /workspace/runs/s3

mkdir -p /workspace/runs/s4
sshfs -o reconnect,allow_other,ServerAliveInterval=15 -p 22 root@s4:/workspace/runs /workspace/runs/s4
```

## 📋 Clone Source Code (Execute on All Nodes)

```bash
# Clone source on master node
cd /workspace && git clone https://github.com/ai-computing/aicomp.git

# Clone source on worker nodes via SSH
ssh -p 22 -tt root@s2 "cd /workspace && git clone https://github.com/ai-computing/aicomp.git"
ssh -p 22 -tt root@s3 "cd /workspace && git clone https://github.com/ai-computing/aicomp.git"
ssh -p 22 -tt root@s4 "cd /workspace && git clone https://github.com/ai-computing/aicomp.git"
```

## 🔍 Monitoring: Access from PC
- JupyterLab access: http://s1:80
- TensorBoard access: http://s1:86

## 🧪 Running torchrun from JupyterLab

Example: Training GPT-2 on s1 and s2 nodes

Create notebook 1, add the following to a cell and execute:
```bash
!torchrun \
--nproc_per_node=4 \
--nnodes=2 \
--node_rank=0 \
--master_addr=s1 \
--master_port=29500 \
/workspace/aicomp/opt_prime/demo/pp_train_gpt2.py \
```

Create notebook 2, add the following to a cell and execute:
```bash
!ssh -tt root@s2 /opt/conda/bin/torchrun \
--nproc_per_node=4 \
--nnodes=2 \
--node_rank=1 \
--master_addr=s1 \
--master_port=29500 \
/workspace/aicomp/opt_prime/demo/pp_train_gpt2.py
```

# 📊 Docker-based GPU + InfiniBand + Node Monitoring System Setup
> Prometheus + Grafana + DCGM Exporter + InfiniBand Exporter

## 🧱 1. Prerequisites (s1, s2, s3, s4)

### 1.1 Docker Installation
```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
```

---

## 🔧 2. NVIDIA Runtime Registration (One-time setup on GPU nodes only)

### (Optional) 2.1 NVIDIA Container Toolkit Installation
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-docker2
```

### 2.2 Configure `/etc/docker/daemon.json`
```bash
sudo vi /etc/docker/daemon.json
```

```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

### 2.3 Restart Docker and Verify
```bash
sudo systemctl restart docker
docker info | grep -i runtime
# → Should include "nvidia"
```

---

## 🚀 3. Running Exporter Containers (s1, s2, s3, s4)

### 3.1 Run GPU Exporter
```bash
sudo docker run -d \
  --name dcgm-exporter \
  --runtime=nvidia \
  --gpus all \
  -p 9400:9400 \
  nvidia/dcgm-exporter:latest
```

### 3.2 Run InfiniBand Exporter
```bash
sudo docker run -d \
  --name infiniband_exporter \
  --cap-add=IPC_LOCK \
  --device=/dev/infiniband/umad0 \
  -p 9315:9315 \
  treydock/infiniband_exporter \
  --collector.switch \
  --collector.hca
```
> 💡 Verify `umad0` with: `ls /dev/infiniband`  
> 💡 You can also mount the entire directory if needed: `/dev/infiniband:/dev/infiniband`

### 3.3 Run Node Exporter
```bash
sudo docker run -d \
  --name node-exporter \
  --restart unless-stopped \
  -p 9100:9100 \
  --net="host" \
  --pid="host" \
  --cap-add="SYS_TIME" \
  -v "/:/host:ro,rslave" \
  prom/node-exporter \
  --path.rootfs=/host
```

---

## 📡 4. Running Prometheus (S1 node)

### 4.1 Prepare Configuration Directory
```bash
mkdir -p ~/monitoring-stack
cd ~/monitoring-stack
```

### 4.2 Create Prometheus Configuration File (`prometheus.yml`)
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'gpu'
    static_configs:
      - targets: [
              's1:9400',
              's2:9400',
              's3:9400',
              's4:9400'
        ]
    relabel_configs:
      - source_labels: [__address__]
        regex: '([^:]+)(:[0-9]+)?'
        target_label: instance
        replacement: '$1'

  - job_name: 'infiniband'
    static_configs:
      - targets: [
              's1:9315',
              's2:9315',
              's3:9315',
              's4:9315'
        ]
    relabel_configs:
      - source_labels: [__address__]
        regex: '([^:]+)(:[0-9]+)?'
        target_label: instance
        replacement: '$1'

  - job_name: 'node'
    static_configs:
      - targets: [
              's1:9100',
              's2:9100',
              's3:9100',
              's4:9100'
        ]
    relabel_configs:
      - source_labels: [__address__]
        regex: '([^:]+)(:[0-9]+)?'
        target_label: instance
        replacement: '$1'
```

```bash
vi prometheus.yml
```

```bash
# Note: Restart Prometheus Docker container after config changes
sudo docker restart prometheus
# Alternative:
# sudo docker stop prometheus
# sudo docker start prometheus

# To delete data if needed, run:
curl -X POST -g 'http://s1:9090/api/v1/admin/tsdb/delete_series?match[]={instance="s1:9090"}'
```

### 4.3 Run Prometheus Container
```bash
sudo docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v ~/monitoring-stack/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus:latest
```

---

## 📊 5. Running Grafana (S1 node)

```bash
sudo docker run -d \
  --name grafana \
  -p 3000:3000 \
  -v grafana-storage:/var/lib/grafana \
  grafana/grafana:latest
```

- Grafana access: `http://s1:3000`
- Default login: `admin / admin`

---

## 🔍 6. Grafana Configuration

### 6.1 Add Prometheus Data Source
- Connections > Data Sources > Add data source
- Select: Prometheus
- URL: `http://prometheus:9090` or `http://s1:9090`
- Save & Test

### 6.2 Dashboard Setup
Create a new dashboard and configure it using the dashboard configuration source code provided in the appendix at the end of this document. You can find the complete dashboard configuration JSON in the [Dashboard Configuration Source Code](#dashboard-configuration-source-code) section.

---

## ✅ 7. Verification Commands

### Container Status
```bash
docker ps
```

### Verify Metrics Collection
```bash
curl http://localhost:9400/metrics      # GPU
curl http://localhost:9315/metrics      # InfiniBand
curl http://localhost:9100/metrics      # Node
```

### Check Prometheus Target Status
```
http://s1:9090/targets
http://s2:9090/targets
http://s3:9090/targets
http://s4:9090/targets
```

---

## 📌 Port Summary

| Component           | Port   | Description                |
|--------------------|--------|----------------------------|
| Prometheus         | 9090   | Prometheus UI              |
| Grafana            | 90     | Grafana Dashboard (changed)|
| GPU Exporter       | 9400   | DCGM GPU Metrics           |
| InfiniBand Exporter| 9315   | IB Metrics Collection Port |
| Node Exporter      | 9100   | Node Information           |

---


# dashboard-configuration-source-code
```json
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": {
          "type": "grafana",
          "uid": "-- Grafana --"
        },
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 0,
  "id": 19,
  "links": [],
  "panels": [
    {
      "datasource": {
        "type": "prometheus",
        "uid": "behvby39px43kd"
      },
      "fieldConfig": {
        "defaults": {
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green"
              },
              {
                "color": "#6ED0E0",
                "value": 20
              },
              {
                "color": "#EAB839",
                "value": 40
              },
              {
                "color": "#EF843C",
                "value": 70
              },
              {
                "color": "#E24D42",
                "value": 90
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 4,
        "w": 6,
        "x": 0,
        "y": 0
      },
      "id": 1,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "percentChangeColorMode": "standard",
        "reduceOptions": {
          "calcs": [
            "lastNotNull"
          ],
          "fields": "",
          "values": false
        },
        "showPercentChange": false,
        "textMode": "auto",
        "wideLayout": true
      },
      "pluginVersion": "11.6.0",
      "targets": [
        {
          "editorMode": "code",
          "expr": "avg(DCGM_FI_DEV_GPU_UTIL{instance=~\"$node\"})",
          "interval": "",
          "legendFormat": "{{instance}}",
          "range": true,
          "refId": "A"
        }
      ],
      "title": "GPU Utilization (Average, %)",
      "type": "stat"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "behvby39px43kd"
      },
      "fieldConfig": {
        "defaults": {
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green"
              },
              {
                "color": "#EAB839",
                "value": 40
              },
              {
                "color": "#EF843C",
                "value": 60
              },
              {
                "color": "red",
                "value": 80
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 4,
        "w": 6,
        "x": 6,
        "y": 0
      },
      "id": 2,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "percentChangeColorMode": "standard",
        "reduceOptions": {
          "calcs": [
            "lastNotNull"
          ],
          "fields": "",
          "values": false
        },
        "showPercentChange": false,
        "textMode": "auto",
        "wideLayout": true
      },
      "pluginVersion": "11.6.0",
      "targets": [
        {
          "editorMode": "code",
          "expr": "avg(DCGM_FI_DEV_GPU_TEMP{instance=~\"$node\"})",
          "range": true,
          "refId": "A"
        }
      ],
      "title": "GPU Temperature (Average, C)",
      "type": "stat"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "behvby39px43kd"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green"
              },
              {
                "color": "#6ED0E0",
                "value": 30
              },
              {
                "color": "#EAB839",
                "value": 50
              },
              {
                "color": "#EF843C",
                "value": 70
              },
              {
                "color": "red",
                "value": 90
              }
            ]
          },
          "unit": "percent"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 4,
        "w": 6,
        "x": 12,
        "y": 0
      },
      "id": 15,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "percentChangeColorMode": "standard",
        "reduceOptions": {
          "calcs": [
            "lastNotNull"
          ],
          "fields": "",
          "values": false
        },
        "showPercentChange": false,
        "textMode": "auto",
        "wideLayout": true
      },
      "pluginVersion": "11.6.0",
      "targets": [
        {
          "editorMode": "code",
          "expr": "(sum(DCGM_FI_DEV_FB_USED{instance=~\"$node\"}) / (sum(DCGM_FI_DEV_FB_USED{instance=~\"$node\"}) + sum(DCGM_FI_DEV_FB_FREE{instance=~\"$node\"}))) * 100",
          "range": true,
          "refId": "A"
        }
      ],
      "title": "GPU Memory Usage (Used/Total, %)",
      "type": "stat"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "behvby39px43kd"
      },
      "fieldConfig": {
        "defaults": {
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "text"
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 4,
        "w": 6,
        "x": 18,
        "y": 0
      },
      "id": 3,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "percentChangeColorMode": "standard",
        "reduceOptions": {
          "calcs": [
            "lastNotNull"
          ],
          "fields": "",
          "values": false
        },
        "showPercentChange": false,
        "textMode": "auto",
        "wideLayout": true
      },
      "pluginVersion": "11.6.0",
      "targets": [
        {
          "editorMode": "code",
          "expr": "sum(DCGM_FI_DEV_FB_USED{instance=~\"$node\"}) / 1000",
          "range": true,
          "refId": "A"
        }
      ],
      "title": "GPU Memory Used (Total, KB)",
      "type": "stat"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "behvby39px43kd"
      },
      "fieldConfig": {
        "defaults": {
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green"
              },
              {
                "color": "#EF843C",
                "value": 100
              },
              {
                "color": "#E24D42",
                "value": 1000
              },
              {
                "color": "#6ED0E0",
                "value": 10000
              },
              {
                "color": "#EAB839",
                "value": 100000
              },
              {
                "color": "#1F78C1",
                "value": 1000000
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 4,
        "w": 6,
        "x": 0,
        "y": 4
      },
      "id": 4,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "percentChangeColorMode": "standard",
        "reduceOptions": {
          "calcs": [
            "lastNotNull"
          ],
          "fields": "",
          "values": false
        },
        "showPercentChange": false,
        "textMode": "auto",
        "wideLayout": true
      },
      "pluginVersion": "11.6.0",
      "targets": [
        {
          "editorMode": "code",
          "expr": "sum(rate(node_infiniband_port_data_transmitted_bytes_total{instance=~\"$node\"}[1m]) + rate(node_infiniband_port_data_received_bytes_total{instance=~\"$node\"}[1m])) / 1000",
          "range": true,
          "refId": "A"
        }
      ],
      "title": "Infiniband Bandwidth (Total TX+RX in 1m, KB)",
      "type": "stat"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "behvby39px43kd"
      },
      "fieldConfig": {
        "defaults": {
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green"
              },
              {
                "color": "#6ED0E0",
                "value": 30
              },
              {
                "color": "#EAB839",
                "value": 50
              },
              {
                "color": "#EF843C",
                "value": 70
              },
              {
                "color": "red",
                "value": 90
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 4,
        "w": 6,
        "x": 6,
        "y": 4
      },
      "id": 5,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "percentChangeColorMode": "standard",
        "reduceOptions": {
          "calcs": [
            "lastNotNull"
          ],
          "fields": "",
          "values": false
        },
        "showPercentChange": false,
        "textMode": "auto",
        "wideLayout": true
      },
      "pluginVersion": "11.6.0",
      "targets": [
        {
          "expr": "100 - avg(rate(node_cpu_seconds_total{mode=\"idle\", instance=~\"$node\"}[1m])) * 100",
          "refId": "A"
        }
      ],
      "title": "CPU Usage (Average, %)",
      "type": "stat"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "behvby39px43kd"
      },
      "fieldConfig": {
        "defaults": {
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green"
              },
              {
                "color": "#6ED0E0",
                "value": 30
              },
              {
                "color": "#EAB839",
                "value": 50
              },
              {
                "color": "#EF843C",
                "value": 70
              },
              {
                "color": "red",
                "value": 90
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 4,
        "w": 6,
        "x": 12,
        "y": 4
      },
      "id": 6,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "percentChangeColorMode": "standard",
        "reduceOptions": {
          "calcs": [
            "lastNotNull"
          ],
          "fields": "",
          "values": false
        },
        "showPercentChange": false,
        "textMode": "auto",
        "wideLayout": true
      },
      "pluginVersion": "11.6.0",
      "targets": [
        {
          "editorMode": "code",
          "expr": "avg((node_memory_MemTotal_bytes{instance=~\"$node\"} - node_memory_MemAvailable_bytes{instance=~\"$node\"})/node_memory_MemTotal_bytes{instance=~\"$node\"}) * 100",
          "range": true,
          "refId": "A"
        }
      ],
      "title": "Memory Usage (Average, %)",
      "type": "stat"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "behvby39px43kd"
      },
      "fieldConfig": {
        "defaults": {
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green"
              },
              {
                "color": "red",
                "value": 80
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 4,
        "w": 6,
        "x": 18,
        "y": 4
      },
      "id": 7,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "percentChangeColorMode": "standard",
        "reduceOptions": {
          "calcs": [
            "lastNotNull"
          ],
          "fields": "",
          "values": false
        },
        "showPercentChange": false,
        "textMode": "auto",
        "wideLayout": true
      },
      "pluginVersion": "11.6.0",
      "targets": [
        {
          "editorMode": "code",
          "expr": "sum(rate(node_disk_read_bytes_total{instance=~\"$node\"}[1m]) + rate(node_disk_written_bytes_total{instance=~\"$node\"}[1m])) / 1000",
          "range": true,
          "refId": "A"
        }
      ],
      "title": "Disk Bandwidth (Total R+W in 1m, KB)",
      "type": "stat"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "behvby39px43kd"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisBorderShow": false,
            "axisCenteredZero": false,
            "axisColorMode": "text",
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "barWidthFactor": 0.6,
            "drawStyle": "line",
            "fillOpacity": 0,
            "gradientMode": "none",
            "hideFrom": {
              "legend": false,
              "tooltip": false,
              "viz": false
            },
            "insertNulls": false,
            "lineInterpolation": "linear",
            "lineWidth": 1,
            "pointSize": 5,
            "scaleDistribution": {
              "type": "linear"
            },
            "showPoints": "auto",
            "spanNulls": false,
            "stacking": {
              "group": "A",
              "mode": "none"
            },
            "thresholdsStyle": {
              "mode": "off"
            }
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green"
              },
              {
                "color": "red",
                "value": 80
              }
            ]
          }
        },
        "overrides": [
          {
            "__systemRef": "hideSeriesFrom",
            "matcher": {
              "id": "byNames",
              "options": {
                "mode": "exclude",
                "names": [
                  "s4 GPU:3"
                ],
                "prefix": "All except:",
                "readOnly": true
              }
            },
            "properties": [
              {
                "id": "custom.hideFrom",
                "value": {
                  "legend": false,
                  "tooltip": false,
                  "viz": true
                }
              }
            ]
          }
        ]
      },
      "gridPos": {
        "h": 8,
        "w": 24,
        "x": 0,
        "y": 8
      },
      "id": 8,
      "options": {
        "legend": {
          "calcs": [],
          "displayMode": "list",
          "placement": "bottom",
          "showLegend": true
        },
        "tooltip": {
          "hideZeros": false,
          "mode": "single",
          "sort": "none"
        }
      },
      "pluginVersion": "11.6.0",
      "targets": [
        {
          "editorMode": "code",
          "expr": "DCGM_FI_DEV_GPU_UTIL{instance=~\"$node\"}",
          "legendFormat": "{{instance}} GPU:{{gpu}}",
          "range": true,
          "refId": "A"
        }
      ],
      "title": "GPU Utilization per Node",
      "type": "timeseries"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "behvby39px43kd"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisBorderShow": false,
            "axisCenteredZero": false,
            "axisColorMode": "text",
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "barWidthFactor": 0.6,
            "drawStyle": "line",
            "fillOpacity": 0,
            "gradientMode": "none",
            "hideFrom": {
              "legend": false,
              "tooltip": false,
              "viz": false
            },
            "insertNulls": false,
            "lineInterpolation": "linear",
            "lineWidth": 1,
            "pointSize": 5,
            "scaleDistribution": {
              "type": "linear"
            },
            "showPoints": "auto",
            "spanNulls": false,
            "stacking": {
              "group": "A",
              "mode": "none"
            },
            "thresholdsStyle": {
              "mode": "off"
            }
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green"
              },
              {
                "color": "red",
                "value": 80
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 24,
        "x": 0,
        "y": 16
      },
      "id": 9,
      "options": {
        "legend": {
          "calcs": [],
          "displayMode": "list",
          "placement": "bottom",
          "showLegend": true
        },
        "tooltip": {
          "hideZeros": false,
          "mode": "single",
          "sort": "none"
        }
      },
      "pluginVersion": "11.6.0",
      "targets": [
        {
          "editorMode": "code",
          "expr": "DCGM_FI_DEV_GPU_TEMP{instance=~\"$node\"}",
          "legendFormat": "{{instance}} GPU:{{gpu}}",
          "range": true,
          "refId": "A"
        }
      ],
      "title": "GPU Temperature per Node",
      "type": "timeseries"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "behvby39px43kd"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisBorderShow": false,
            "axisCenteredZero": false,
            "axisColorMode": "text",
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "barWidthFactor": 0.6,
            "drawStyle": "line",
            "fillOpacity": 0,
            "gradientMode": "none",
            "hideFrom": {
              "legend": false,
              "tooltip": false,
              "viz": false
            },
            "insertNulls": false,
            "lineInterpolation": "linear",
            "lineWidth": 1,
            "pointSize": 5,
            "scaleDistribution": {
              "type": "linear"
            },
            "showPoints": "auto",
            "spanNulls": false,
            "stacking": {
              "group": "A",
              "mode": "none"
            },
            "thresholdsStyle": {
              "mode": "off"
            }
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green"
              },
              {
                "color": "red",
                "value": 80
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 24,
        "x": 0,
        "y": 24
      },
      "id": 10,
      "options": {
        "legend": {
          "calcs": [],
          "displayMode": "list",
          "placement": "bottom",
          "showLegend": true
        },
        "tooltip": {
          "hideZeros": false,
          "mode": "single",
          "sort": "none"
        }
      },
      "pluginVersion": "11.6.0",
      "targets": [
        {
          "editorMode": "code",
          "expr": "DCGM_FI_DEV_FB_USED{instance=~\"$node\"}",
          "legendFormat": "{{instance}} GPU:{{gpu}}",
          "range": true,
          "refId": "A"
        }
      ],
      "title": "GPU Memory Usage per Node",
      "type": "timeseries"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "behvby39px43kd"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisBorderShow": false,
            "axisCenteredZero": false,
            "axisColorMode": "text",
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "barWidthFactor": 0.6,
            "drawStyle": "line",
            "fillOpacity": 0,
            "gradientMode": "none",
            "hideFrom": {
              "legend": false,
              "tooltip": false,
              "viz": false
            },
            "insertNulls": false,
            "lineInterpolation": "linear",
            "lineWidth": 1,
            "pointSize": 5,
            "scaleDistribution": {
              "type": "linear"
            },
            "showPoints": "auto",
            "spanNulls": false,
            "stacking": {
              "group": "A",
              "mode": "none"
            },
            "thresholdsStyle": {
              "mode": "off"
            }
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green"
              },
              {
                "color": "red",
                "value": 80
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 24,
        "x": 0,
        "y": 32
      },
      "id": 11,
      "options": {
        "legend": {
          "calcs": [],
          "displayMode": "list",
          "placement": "bottom",
          "showLegend": true
        },
        "tooltip": {
          "hideZeros": false,
          "mode": "single",
          "sort": "none"
        }
      },
      "pluginVersion": "11.6.0",
      "targets": [
        {
          "editorMode": "code",
          "exemplar": false,
          "expr": "rate(node_infiniband_port_data_transmitted_bytes_total{instance=~\"$node\"}[1m])",
          "interval": "",
          "legendFormat": "{{instance}} TX",
          "range": true,
          "refId": "A"
        },
        {
          "editorMode": "code",
          "expr": "rate(node_infiniband_port_data_received_bytes_total{instance=~\"$node\"}[1m])",
          "legendFormat": "{{instance}} RX",
          "range": true,
          "refId": "B"
        }
      ],
      "title": "Infiniband Bandwidth per Node",
      "type": "timeseries"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "behvby39px43kd"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisBorderShow": false,
            "axisCenteredZero": false,
            "axisColorMode": "text",
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "barWidthFactor": 0.6,
            "drawStyle": "line",
            "fillOpacity": 0,
            "gradientMode": "none",
            "hideFrom": {
              "legend": false,
              "tooltip": false,
              "viz": false
            },
            "insertNulls": false,
            "lineInterpolation": "linear",
            "lineWidth": 1,
            "pointSize": 5,
            "scaleDistribution": {
              "type": "linear"
            },
            "showPoints": "auto",
            "spanNulls": false,
            "stacking": {
              "group": "A",
              "mode": "none"
            },
            "thresholdsStyle": {
              "mode": "off"
            }
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green"
              },
              {
                "color": "red",
                "value": 80
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 24,
        "x": 0,
        "y": 40
      },
      "id": 12,
      "options": {
        "legend": {
          "calcs": [],
          "displayMode": "list",
          "placement": "bottom",
          "showLegend": true
        },
        "tooltip": {
          "hideZeros": false,
          "mode": "single",
          "sort": "none"
        }
      },
      "pluginVersion": "11.6.0",
      "targets": [
        {
          "editorMode": "code",
          "expr": "100 - rate(node_cpu_seconds_total{mode=\"idle\", instance=~\"$node\"}[1m]) * 100",
          "legendFormat": "{{instance}} CPU:{{cpu}}",
          "range": true,
          "refId": "A"
        }
      ],
      "title": "CPU Utilization per Node",
      "type": "timeseries"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "behvby39px43kd"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisBorderShow": false,
            "axisCenteredZero": false,
            "axisColorMode": "text",
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "barWidthFactor": 0.6,
            "drawStyle": "line",
            "fillOpacity": 0,
            "gradientMode": "none",
            "hideFrom": {
              "legend": false,
              "tooltip": false,
              "viz": false
            },
            "insertNulls": false,
            "lineInterpolation": "linear",
            "lineWidth": 1,
            "pointSize": 5,
            "scaleDistribution": {
              "type": "linear"
            },
            "showPoints": "auto",
            "spanNulls": false,
            "stacking": {
              "group": "A",
              "mode": "none"
            },
            "thresholdsStyle": {
              "mode": "off"
            }
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green"
              },
              {
                "color": "red",
                "value": 80
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 24,
        "x": 0,
        "y": 48
      },
      "id": 13,
      "options": {
        "legend": {
          "calcs": [],
          "displayMode": "list",
          "placement": "bottom",
          "showLegend": true
        },
        "tooltip": {
          "hideZeros": false,
          "mode": "single",
          "sort": "none"
        }
      },
      "pluginVersion": "11.6.0",
      "targets": [
        {
          "expr": "(node_memory_MemTotal_bytes{instance=~\"$node\"} - node_memory_MemAvailable_bytes{instance=~\"$node\"})/node_memory_MemTotal_bytes{instance=~\"$node\"} * 100",
          "legendFormat": "{{instance}}",
          "refId": "A"
        }
      ],
      "title": "Memory Usage per Node (%)",
      "type": "timeseries"
    },
    {
      "datasource": {
        "type": "prometheus",
        "uid": "behvby39px43kd"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisBorderShow": false,
            "axisCenteredZero": false,
            "axisColorMode": "text",
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "barWidthFactor": 0.6,
            "drawStyle": "line",
            "fillOpacity": 0,
            "gradientMode": "none",
            "hideFrom": {
              "legend": false,
              "tooltip": false,
              "viz": false
            },
            "insertNulls": false,
            "lineInterpolation": "linear",
            "lineWidth": 1,
            "pointSize": 5,
            "scaleDistribution": {
              "type": "linear"
            },
            "showPoints": "auto",
            "spanNulls": false,
            "stacking": {
              "group": "A",
              "mode": "none"
            },
            "thresholdsStyle": {
              "mode": "off"
            }
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green"
              },
              {
                "color": "red",
                "value": 80
              }
            ]
          }
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 24,
        "x": 0,
        "y": 56
      },
      "id": 14,
      "options": {
        "legend": {
          "calcs": [],
          "displayMode": "list",
          "placement": "bottom",
          "showLegend": true
        },
        "tooltip": {
          "hideZeros": false,
          "mode": "single",
          "sort": "none"
        }
      },
      "pluginVersion": "11.6.0",
      "targets": [
        {
          "editorMode": "code",
          "exemplar": false,
          "expr": "rate(node_disk_read_bytes_total{instance=~\"$node\"}[1m])",
          "interval": "",
          "legendFormat": "{{instance}} Read",
          "range": true,
          "refId": "A"
        },
        {
          "editorMode": "code",
          "expr": "rate(node_disk_written_bytes_total{instance=~\"$node\"}[1m])",
          "legendFormat": "{{instance}} Write",
          "range": true,
          "refId": "B"
        }
      ],
      "title": "Disk Bandwidth per Node",
      "type": "timeseries"
    }
  ],
  "preload": false,
  "refresh": "auto",
  "schemaVersion": 41,
  "tags": [
    "gpu",
    "node",
    "monitoring"
  ],
  "templating": {
    "list": [
      {
        "allowCustomValue": false,
        "current": {
          "text": [
            "s3",
            "s4"
          ],
          "value": [
            "s3",
            "s4"
          ]
        },
        "definition": "label_values(instance)",
        "description": "",
        "includeAll": true,
        "multi": true,
        "name": "node",
        "options": [],
        "query": {
          "qryType": 1,
          "query": "label_values(instance)",
          "refId": "PrometheusVariableQueryEditor-VariableQuery"
        },
        "refresh": 1,
        "regex": "",
        "type": "query"
      }
    ]
  },
  "time": {
    "from": "now-5m",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "browser",
  "title": "DASHBOARD_ETRI_OPTIMUS",
  "uid": "ceiyl6zsz5s00d",
  "version": 70
}
```
