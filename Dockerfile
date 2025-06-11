FROM python:3.12-slim AS builder

# Build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-compile --no-cache-dir --user -r requirements.txt

# Application image
FROM python:3.12-slim AS app
RUN apt-get update && apt-get install -y \
    default-jdk \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Maven
ENV MAVEN_HOME=/opt/maven
ENV PATH=$MAVEN_HOME/bin:$PATH
ENV MAVEN_VERSION=3.9.10
RUN wget https://dlcdn.apache.org/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz -O /tmp/maven.tar.gz && \
    tar xvf /tmp/maven.tar.gz -C /opt && \
    mv /opt/apache-maven-$MAVEN_VERSION $MAVEN_HOME

# Copy python dependencies over from builder
COPY --from=builder /root/.local /root/.local

WORKDIR /app

# Copy application code
COPY CellPheDashboard.py .
COPY cellpose_models/ ./cellpose_models/
COPY setup_imagej.py .

# Install Java dependencies for ImageJ
# Easier to do this via pyimagej rather than setting up a maven project
# NB: this could be done in builder but then would need to install maven in both
RUN python setup_imagej.py

# Download cyto3 CellPose model
RUN mkdir -p ~/.cellpose/models
RUN wget https://www.cellpose.org/models/cyto3 -O ~/.cellpose/models/cyto3

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["python", "-m", "streamlit", "run", "CellPheDashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
