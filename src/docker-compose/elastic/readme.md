# Elastic & Kibana Docker-Compose Configurations

## Directory Structure

### 📂 elastic
This directory handles **Elasticsearch** configurations, which may be used for logging, search indexing, or analytics.

#### 📂 logstash (currently not in use)
This subdirectory likely manages **Logstash**, responsible for processing and forwarding logs.

- **docker-compose.yml** – The primary Docker Compose file for setting up the Logstash service.
- **docker-compose (copy) -with logstash but...** – Possibly a backup or alternative configuration.
- **logstash.conf** – Configuration file for defining Logstash pipelines and log processing rules.
- **notes/** – A directory for additional Logstash-related notes or configurations.

### 📄 docker-compose.yml
The main Docker Compose file, orchestrating multiple services such as backend, frontend, and Elastic Stack.

### 📄 docker-compose.yml-bc
A variant of the main Docker Compose file, with different configurations or services.

### 📄 docker-compose.yml-https
A Docker Compose configuration, for enabling HTTPS support.
