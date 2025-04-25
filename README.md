# CellPhe Dashboard

A dashboard for the CellPhe cell phenotyping library.

# Usage

The dashboard is run locally so that you are in full control of your images and do not need to upload them anywhere.
It is cross-platform and runs in a web browser and doesn't require any coding experience beyond that required to run the [CellPhe python package](https://pypi.org/project/cellphe/).

## Prerequisites

The only prerequisite is to install Docker (or [Docker Desktop](https://www.docker.com/products/docker-desktop/)).
Optionally you can run the dashboard from the source code, although this is more involved.

## Running with Docker

Simply download the Docker image from this repository:

`docker pull ghcr.io/uoy-research/cellphe-dashboard`

Then run it:

`docker run -p 8501:8501 uoy-research/cellphe-dashboard`

## Running from source

The app can be run without Docker, although it will require setting up a suitable Python environment.

Firstly, clone this repository:

`git clone https://github.com/uoy-research/CellPhe-dashboard.git`

Then install the Python prequisites, ideally in a [new virtual environment](https://docs.python.org/3/library/venv.html):

`pip install -r requirements.txt`

Then you can run the app with:

`streamlit run CellPheDashboard.py --server.port=8501 --server.address=0.0.0.0`
