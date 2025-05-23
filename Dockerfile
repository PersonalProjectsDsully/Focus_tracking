# Dockerfile for Claude Code
FROM node:18

# Install git and other tools that Claude Code might need
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Claude Code globally
RUN npm install -g @anthropic-ai/claude-code

# Set working directory
WORKDIR /workspace

# Default command
CMD ["claude"]