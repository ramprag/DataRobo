# Multi-stage build for React app
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./
#COPY /backend/requirements.txt ./requirements.txt

# Install dependencies
#RUN npm ci --only=production

# Copy source code
COPY . .

# Build the application
#RUN npm install
Run npm install react-scripts@latest
RUN npm run build

# Production stage with nginx
FROM nginx:alpine
RUN apk add --no-cache curl

# Copy built app from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Copy custom nginx configuration
#COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create non-root user
#RUN addgroup -g 1001 -S nodejs && \
 #   adduser -S nextjs -u 1001

# Set ownership and permissions
#RUN chown -R nextjs:nodejs /usr/share/nginx/html && \
#    chown -R nextjs:nodejs /var/cache/nginx && \
#    chown -R nextjs:nodejs /var/log/nginx && \
#    chown -R nextjs:nodejs /etc/nginx/conf.d

# Switch to non-root user
#USER nextjs

COPY <<EOF /etc/nginx/conf.d/default.conf
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html index.htm;

    location / {
        try_files \$uri \$uri/ /index.html;
    }

    location /api {
        proxy_pass http://backend:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
}
EOF

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]