#!/bin/bash
# debug_nginx.sh - Debug nginx and frontend issues

echo "🔍 Debugging Nginx and Frontend Issues..."

echo "📊 Container Status:"
docker-compose ps

echo ""
echo "🌐 Testing Nginx Health:"
curl -s http://localhost/nginx-health || echo "❌ Nginx health check failed"

echo ""
echo "🏥 Testing API Gateway Health:"
curl -s http://localhost/health || echo "❌ API Gateway health check failed"

echo ""
echo "📝 Nginx Logs (last 10 lines):"
docker-compose logs --tail=10 nginx

echo ""
echo "🖥️ Web Frontend Logs (last 10 lines):"
docker-compose logs --tail=10 web

echo ""
echo "🔍 Testing Frontend Root:"
curl -I http://localhost/ || echo "❌ Frontend root failed"

echo ""
echo "📂 Checking if frontend files exist:"
docker-compose exec web ls -la /var/www/html/ || echo "❌ Could not check frontend files"

echo ""
echo "⚙️ Nginx Configuration Test:"
docker-compose exec nginx nginx -t || echo "❌ Nginx config test failed"

echo ""
echo "🔗 Testing API Routes:"
echo "Testing /api/query endpoint:"
curl -s -X POST http://localhost/api/query -H "Content-Type: application/json" -d '{"query": "test"}' || echo "❌ API query failed"

echo ""
echo "📋 Current Nginx Configuration:"
docker-compose exec nginx cat /etc/nginx/nginx.conf | grep -A 10 -B 5 "location"

echo ""
echo "🎯 Recommendations:"
echo "1. Check if your frontend files are properly mounted"
echo "2. Verify your web-service container is serving files correctly" 
echo "3. Check if CSS/JS files exist in your frontend directory"
echo "4. Consider updating your docker-compose.yml web service configuration"