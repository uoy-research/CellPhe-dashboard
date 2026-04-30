# CellPhe Dashboard

A dashboard for the CellPhe cell phenotyping library.

# Usage

The dashboard is run locally so that you are in full control of your images and do not need to upload them anywhere.
It is cross-platform and runs in a web browser and doesn't require any coding experience beyond that required to run the [CellPhe python package](https://pypi.org/project/cellphe/).

## Prerequisites

The only prerequisite is to install Docker (or [Docker Desktop](https://www.docker.com/products/docker-desktop/) for Windows users).
Optionally you can run the dashboard from the source code, although this is more involved and slower as it requires manually installing the dependencies.
However, if you are unable to install Docker on your computer due to the elevated permissions it needs to run, then this might be necessary.

## Running with Docker

Simply download the Docker image from this repository by running the following command in a terminal or command prompt:

```bash
docker pull ghcr.io/uoy-research/cellphe-dashboard:main
```

Then run it:

```bash
docker run --rm -p 8501:8501 ghcr.io/uoy-research/cellphe-dashboard:main
```

You can then access the app by navigating to `http://127.0.0.1:8501/` in a web-browser.

## Running from source

The app can be run without Docker, although it will require installing the dependencies by hand which takes some time.
**NB: Currently Python 3.12 is the supported version, other versions may require tweaking the Python dependencies**.

The tracking functionality depends on the ImageJ plugin TrackMate, which in turn depends on having a Java runtime available.
We recommend the **Eclipse Temurin** variant, which is built on OpenJDK and is free, stable, and widely supported.

1.  **Download:** Go to the [Adoptium Temurin website](https://adoptium.net/).
2.  **Select & Install:**
    * **Windows:** Click the "Latest LTS Release" button. Run the downloaded `.msi` file and follow the prompts. (Ensure the "Set `JAVA_HOME` variable" option is checked during installation).
    * **macOS:** Download the `.pkg` file for your chip type (Apple Silicon or Intel). Open it and follow the installation prompts.
    * **Linux (Debian/Ubuntu):** Open a terminal and run the following (NB: you can also install Canonical's build from the `openjdk-21-jdk` package)
        ```bash
        sudo apt update
        sudo apt install temurin-21-jdk
        ```

After installing, open a **new** terminal/command prompt and run:
```bash
java -version
```

If you see the version details, your Java installation is successful and you are ready to setup the Python environment.
Firstly, clone this repository and change into it:

```bash
git clone https://github.com/uoy-research/CellPhe-dashboard.git
cd CellPhe-dashboard
```

Then install the Python prequisites, ideally in a [new virtual environment](https://docs.python.org/3/library/venv.html):

```bash
pip install -r requirements.txt
```

Then you can run the app although NB: the first time tracking is run it will take some time to setup an ImageJ instance.

```bash
streamlit run CellPheDashboard.py --server.port=8501 --server.address=0.0.0.0
```

## Windows issues

Windows users might encounter some additional OS-specific issues.
This section aims to document some of these in order to remove any barriers to use.

### Installing Docker

As mentioned above, running the dashboard through Docker involves installing the fewest dependencies, however, installing Docker itself on Windows is not as straightforward as it is on other OSes.
The easiest way to install Docker on Windows is through the [Docker Desktop](https://www.docker.com/products/docker-desktop/) GUI, however, the [licence agreement](https://docs.docker.com/subscription/desktop-license/) requires a paid subscription for users in organisations with > 250 employees or > $10million in annual revenue.
An alternative is to use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) although this requires setting up WSL itself and using a potentially unfamiliar Linux environment.
The final possible hurdle with Docker that due to it requiring elevated access it might not be permitted by your organization's IT security policy, in which case running the dashboard from the Python source might be the only option.

### Installing Python

Unlike MacOS or Linux, Python doesn't come installed with Windows by default.
The best way to install Python is from the [official website](https://www.python.org/downloads/windows/), with the following considerations:

  - CellPhe isn't compatible with Python 3.14 - we'd recommend either 3.12 or 3.13
  - Check the box for adding Python to `PATH` during the install process
  - It's a good idea to also install the separate Install Manager from the same website to allow for [quicker switching between Python versions](https://docs.python.org/3/using/windows.html#installing-runtimes) in future

### Visual C++ Build Tools

If you come across an error message like the below when installing CellPhe (or other Python packages), the solution is to install C++ Build Tools from the [Microsoft website](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (tick 'Desktop development with C++').

```python
  WARNING: Failed to activate VS environment: Could not find C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe

      ..\meson.build:1:0: ERROR: Unknown compiler(s): [['icl'], ['cl'], ['cc'], ['gcc'], ['clang'], ['clang-cl'], ['pgcc']]
```

### Pip not found

If you get an error when running `pip install cellphe` about `pip` not being found ("'pip' is not recognized as an internal or external command" or similar), try running `python -m pip install cellphe` instead.
This is because `pip` isn't always added to the `PATH` in Windows.

