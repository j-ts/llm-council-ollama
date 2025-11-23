# Frontend Dockerfile for LLM Council
FROM node:20-slim

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Expose frontend port
EXPOSE 5173

# Run development server
# Vite needs --host to be accessible outside container
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
