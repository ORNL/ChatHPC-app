FROM python:3.11-slim

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install hatch
RUN pip install --no-cache-dir hatch hatch-mkdocs

# Convert SSH URLs to HTTPS in pyproject.toml
RUN bash ./scripts/pyproject_ssh_to_https.sh

# Install the package and all dependencies
RUN pip install --no-cache-dir .

# Set default command
CMD ["/bin/bash"]
