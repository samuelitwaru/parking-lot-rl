# Docker Deployment Guide for Parking Lot RL Flask App

## Quick Start

### Build and Run with Docker
```bash
# Build the Docker image
docker build -t parking-lot-rl .

# Run the container
docker run -p 5000:5000 -v $(pwd)/models:/app/models parking-lot-rl
```

### Using Docker Compose
```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

### With Nginx Reverse Proxy
```bash
# Start with nginx
docker-compose --profile with-nginx up -d
```

## Development

### Local Development with Docker
```bash
# Build development image
docker build -t parking-lot-rl:dev .

# Run with code mounting for development
docker run -p 5000:5000 \
  -v $(pwd):/app \
  -v $(pwd)/models:/app/models \
  -e FLASK_ENV=development \
  parking-lot-rl:dev
```

### Debugging
```bash
# Run container interactively
docker run -it --entrypoint /bin/bash parking-lot-rl

# Check container logs
docker logs <container_name>

# Execute commands in running container
docker exec -it <container_name> /bin/bash
```

## Production Deployment

### Environment Variables
Set these environment variables for production:
- `FLASK_ENV=production`
- `PYTHONUNBUFFERED=1`

### Persistent Storage
Mount volumes for persistent data:
```bash
docker run -p 5000:5000 \
  -v /host/path/models:/app/models \
  -v /host/path/logs:/app/logs \
  parking-lot-rl
```

### Health Checks
The application includes health checks:
- Container health: `http://localhost:5000/api/test_connection`
- Docker health check runs every 30 seconds

### Scaling
Use Docker Compose to scale:
```bash
docker-compose up --scale parking-lot-app=3
```

## Security

### Non-root User
The container runs as a non-root user (`appuser`) for security.

### Network Security
- Only expose necessary ports
- Use environment variables for secrets
- Consider using Docker secrets in production

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Change port mapping
   docker run -p 8080:5000 parking-lot-rl
   ```

2. **Permission issues with volumes**
   ```bash
   # Fix ownership
   sudo chown -R 1000:1000 ./models
   ```

3. **Memory issues**
   ```bash
   # Increase container memory
   docker run --memory=2g parking-lot-rl
   ```

4. **PyTorch/CPU optimization**
   The container is optimized for CPU-only PyTorch to reduce image size.

### Monitoring
```bash
# Monitor resource usage
docker stats

# View application logs
docker-compose logs -f parking-lot-app
```

## Building for Different Architectures

### Multi-platform Build
```bash
# Build for multiple architectures
docker buildx build --platform linux/amd64,linux/arm64 -t parking-lot-rl .
```

## Integration with CI/CD

The Dockerfile works with the existing GitHub Actions workflows:
- Builds are reproducible
- Health checks ensure deployment success
- Proper logging for debugging