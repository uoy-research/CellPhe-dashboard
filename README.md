# CellPhe Dashboard

A dashboard for the CellPhe cell phenotyping library.

# Usage

The dashboard is run locally so that you are in full control of your images and do not need to upload them anywhere.
It is cross-platform and runs in a web browser and doesn't require any coding experience beyond that required to run the [CellPhe python package](https://pypi.org/project/cellphe/).

## Prerequisites

The only prerequisite is to install [Git](https://git-scm.com/) (or [GitHub Desktop](https://desktop.github.com/download/)) and optionally Docker (or [Docker Desktop](https://www.docker.com/products/docker-desktop/)) 

## Running with Docker

Firstly, clone this repository:

`git clone https://github.com/uoy-research/CellPhe-dashboard.git`

Then run Docker Compose from within this folder.
This command will create a Docker image with all of the dependencies installed, it will take several minutes. 

`docker compose up --build`

Once the above command finishes, you should able to access the app through a web browser at the address: `http://0.0.0.0:8501`.

For all subsequent uses you do not need to install the dependencies again and can simply run `docker compose up`.

### Caution: file system usage

This dashboard uses [Streamlit](https://streamlit.io/) which has some heavy dependencies. 
As a result, the Docker image is very large (~10GB), so ensure that you have this amount of hard drive space available.
You can always clear your Docker cache with `docker system prune`.

## Running directly with Python

The app can be run without Docker, although it will require setting up a suitable Python environment.

Firstly, clone this repository:

`git clone https://github.com/uoy-research/CellPhe-dashboard.git`

Then install the Python prequisites, ideally in a [new virtual environment](https://docs.python.org/3/library/venv.html):

`pip install -r requirements.txt`

Then you can run the app with:

`streamlit run GUI.py --server.port=8501 --server.address=0.0.0.0`
