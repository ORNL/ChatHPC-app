FROM python:3.11-slim

# Install build dependencies and clean up aggressively
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Convert SSH URLs to HTTPS in pyproject.toml
RUN bash ./scripts/pyproject_ssh_to_https.sh

# Create venv and install dependencies with uv
# Set UV cache to /app to allow hardlinking (same filesystem)
ENV UV_CACHE_DIR=/app/.uv-cache
RUN uv venv --seed --python 3.11 \
    && . .venv/bin/activate \
    && uv sync --all-groups \
    && rm -rf /app/.uv-cache \
    && find .venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find .venv -type f -name "*.pyc" -delete 2>/dev/null || true

# Activate venv by default
ENV PATH="/app/.venv/bin:$PATH"

# Set default command
CMD ["/bin/bash"]
