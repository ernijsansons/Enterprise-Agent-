# Enterprise Agent Docker Image
# Multi-stage build for minimal image size

# Stage 1: Builder
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for Claude Code CLI
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt* pyproject.toml* ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; \
    elif [ -f pyproject.toml ]; then pip install poetry && poetry install; fi

# Install Claude Code CLI (optional)
RUN npm install -g @anthropic-ai/claude-code || true

# Stage 2: Runtime
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash agent

# Set working directory
WORKDIR /agent

# Copy from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/lib/node_modules /usr/lib/node_modules

# Copy application code
COPY --chown=agent:agent . .

# Create directories
RUN mkdir -p /agent/.enterprise-agent/cache \
    /agent/.enterprise-agent/logs \
    /agent/.enterprise-agent/templates \
    && chown -R agent:agent /agent

# Switch to non-root user
USER agent

# Set environment variables
ENV PYTHONPATH=/agent:$PYTHONPATH \
    USE_CLAUDE_CODE=false \
    PRIMARY_MODEL=claude-3-5-sonnet-20241022 \
    FALLBACK_MODEL=gpt-4o-mini

# Create volume mount points
VOLUME ["/workspace", "/agent/.enterprise-agent"]

# Default to interactive mode
ENTRYPOINT ["python", "enterprise_agent_cli.py"]
CMD ["interactive", "--domain", "coding"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Labels
LABEL maintainer="Enterprise Team" \
      version="3.4.0" \
      description="Enterprise Agent - Multi-domain AI agent" \
      org.opencontainers.image.source="https://github.com/yourorg/enterprise-agent"