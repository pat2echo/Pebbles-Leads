#!/bin/bash
# deploy.sh - Production deployment script

set -e  # Exit on any error

echo "🚀 Starting production deployment..."

# Create necessary directories
mkdir -p nginx
mkdir -p data/{chroma_db,memory,feedback,documents,ollama}

# Set proper permissions for data directories (UID 1000 is typically the first regular user)
echo "🔧 Setting permissions..."
sudo chown -R 1000:1000 data/ || chown -R 1000:1000 data/ 2>/dev/null || echo "Note: Could not change ownership, but Docker volumes should work"
chmod -R 755 data/
chmod -R 755 nginx/

# Environment-specific configurations
if [ "$1" = "staging" ]; then
    echo "📦 Deploying to staging environment..."
    export COMPOSE_FILE="docker-compose.yml:docker-compose.staging.yml"
elif [ "$1" = "production" ]; then
    echo "🏭 Deploying to production environment..."
    export COMPOSE_FILE="docker-compose.yml:docker-compose.production.yml"
    
    # Additional production checks
    if [ ! -f ".env.production" ]; then
        echo "❌ Missing .env.production file"
        exit 1
    fi
    
    # Load production environment
    source .env.production
else
    echo "📋 Deploying to development environment..."
fi

# Build and deploy
echo "🔨 Building Docker images..."
docker-compose build --no-cache

echo "🔄 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Pull required models
echo "🔍 Downloading ollama models..."
docker-compose exec ollama ollama pull llama3
docker-compose exec ollama ollama pull tinyllama
docker-compose exec ollama ollama pull nomic-embed-text
sleep 10

# Health checks
echo "🔍 Running health checks..."

# Check Nginx first
if curl -f "http://localhost/nginx-health" >/dev/null 2>&1; then
    echo "✅ Nginx is healthy"
else
    echo "❌ Nginx health check failed"
    docker-compose logs --tail=50 nginx
    exit 1
fi

# Check API Gateway directly (it manages other service health checks)
if curl -f "http://localhost/health" >/dev/null 2>&1; then
    echo "✅ API Gateway and downstream services are healthy"
    
    # Show detailed health status
    echo "📊 Detailed health status:"
    curl -s "http://localhost/health" | python3 -m json.tool || echo "Could not parse health response"
else
    echo "❌ API Gateway health check failed"
    echo "📋 Gateway service logs:"
    docker-compose logs --tail=50 api-gateway
    
    echo "📋 Other service logs:"
    for service in core-rag-service memory-service feedback-service indexing-service; do
        echo "--- $service ---"
        docker-compose logs --tail=20 $service
    done
    exit 1
fi

echo "🎉 Deployment completed successfully!"
echo "🌐 Application is available at: http://localhost"
echo "📊 Nginx status: http://localhost/nginx-health"

# Optional: Run smoke tests
if [ -f "smoke_tests.sh" ]; then
    echo "🧪 Running smoke tests..."
    ./smoke_tests.sh
fi