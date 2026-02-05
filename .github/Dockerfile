FROM python:3.11-slim

# Install build dependencies and clean up aggressively
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install hatch and build tools
RUN pip install --no-cache-dir hatch hatch-mkdocs \
    && pip cache purge

# Convert SSH URLs to HTTPS in pyproject.toml
RUN bash ./scripts/pyproject_ssh_to_https.sh

# Install the package and all dependencies
RUN pip install --no-cache-dir . \
    && pip cache purge \
    && find /usr/local/lib/python3.11/site-packages -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.11/site-packages -type d -name "*.dist-info" -prune -o -type f -name "*.pyc" -delete 2>/dev/null || true

# Set default command
CMD ["/bin/bash"]
