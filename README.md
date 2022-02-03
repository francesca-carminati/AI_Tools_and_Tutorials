<div id="top"></div>

<br />
<div align="center">
  <a href="https://github.com/Peltarion/Example">
    <img src="images/peltarion_logotype_Pi_red.svg" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Example</h3>

  <p align="center">
    Trying the new e3k
    <br />
    <a href="https://github.com/Peltarion/Example"><strong>Explore the docs »</strong></a>
    <br />
    <a href="https://github.com/Peltarion/Example/issues">Report Bug</a>
  </p>
</div>



<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#environment">Environment</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#linting">Linting</a></li>
      </ul>
    </li>
    <li>
      <a href="#features">Features</a>
      <ul>
        <li><a href="#name-tagging">Automatic container name tagging</a></li>
        <li><a href="#experiment-tracking">Experiment tracking</a></li>
        <li><a href="#gpus">GPUs</a></li>
      </ul>
    </li>
    <li>
       <a href="#variables">Environment variables</a>
    </li>
  </ol>
</details>



## About The Project

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

* [Cookiecutter](https://cookiecutter.readthedocs.io)
* [Conda](https://docs.conda.io/en/latest/miniconda.html)
* [Mamba](https://mamba.readthedocs.io)
* [Docker/Docker Compose](https://docs.docker.com/compose)
* [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

Using it is as easy as `make up`. However, before doing so there
are some environment variables that have to be set, and the container
has to be built. For your convenience, all the make commands will include the `.env` file, so
copying the `.env.template` file to `.env` and filling in the
variables is all you have to do to get going.

Having set all the variables, simply run `make build` followed by
`make up` to start the Jupyter server and the Tensorboard server.
Make sure your user is in the `docker` group and can run docker
commands without `sudo`. If not, you can configure this using [these steps](https://www.configserverfirewall.com/ubuntu-linux/add-user-to-docker-group-ubuntu/).


### Prerequisites

This project uses mamba/conda for handling the python environment. If you already have conda installed but not mamba, run the following command:

  ```sh
  conda install mamba -n base -c conda-forge
  ```


### Installation
1. Configure these variables in `.env.template`
   ```sh
   TENSORBOARD_PORT=
   JUPYTER_PORT=
   JUPYTER_PW=
   GPU_IDS=
   ```
2. Create the environment
   ```sh
   make env-create
   ```
3. Install and run pre-commit
   ```sh
   pre-commit install && pre-commit
   ```
4. Build the docker-compose project
   ```sh
   make build
   ```

### Running

1. Follow the instructions above to setup environments
2. Start docker
   ```sh
   make up
   ```
3. Run the project with one of the following:
    * Open the jupyter-notebook that is started automatically, in a browser.
    * Start bash inside the container
       ```sh
       make shell
       ```
    * Run a python file
       ```sh
       docker-compose run jupyter-server conda run -n Example python example/<file>.py
       ```
4. To shutdown your docker instance
   ```sh
   make down
   ```

### Linting

Linting:
   ```sh
   make lint
   ```
Reformat:
   ```sh
   make reformat
   ```
Test:
   ```sh
   make test
   ```

### Environment

Create environment:
   ```sh
   make env-create
   ```
Remove environment:
   ```sh
   make env-remove
   ```
Export environment:
   ```sh
   make env-export
   ```


<p align="right">(<a href="#top">back to top</a>)</p>


## Features
### Automatic container name tagging
The build will automatically tag the containers you start with your
specified project name and username. This makes it easy for other
people working on the same machine to know who is running what
container through `docker ps`.

### Experiment tracking
Out-of-the-box support covers Tensorboard.

### GPUs
To use GPUs, simply set the `GPU_IDS` variable in the `.env` file as described [here](https://github.com/NVIDIA/nvidia-container-runtime#environment-variables-oci-spec).
Setting it to a value will require a CUDA installation on the host system.

<p align="right">(<a href="#top">back to top</a>)</p>

## Environment variables
- `DATA_DIR`

  The absolute path to where you want to store data on the host machine. Commonly set to the data folder in this repository, such as `/mnt/storage/data/<first.lastname>/<project-name>/data` when working on `gpugpu` or `scorpion`.


- `MODEL_DIR`

  The absolute path to where you want to store models on the host machine. Commonly set to the data folder in this repository, such as `/mnt/storage/data/<first.lastname>/<project-name>/model` when working on `gpugpu` or `scorpion`.


- `GPU_IDS`

  The GPU ids that will be exposed inside the container as indexed by running `nvidia-smi` on the host machine. Observe that inside the container the indices will be start at 0. If needed, multiple GPUs can be used as: `GPU_IDS=0,1`


- `JUPYTER_PW`

  The password to the Jupyter lab service. You choose this yourself.


- `JUPYTER_PORT`

  The port to the Jupyter lab service. You choose this yourself, but it is a good idea to runt `docker ps`  and check which ports are already taken before you settle on one.


- `TENSORBOARD_PORT`

  The port to the Tensorboard service. You choose this yourself, but consider the same issues as when setting `JUPYTER_PORT`.


- `NEPTUNE_PROJECT`

  Used by Neptune experiment logger to indicate where your experiment metrics will be stored.
  To see how to setup a Neptune project and API token, see
  https://docs.neptune.ai/getting-started/installation#authentication-neptune-api-token


- `NEPTUNE_API_TOKEN`

  API token for using Neptune experiment logger.


- `WORKSPACE_NAME`

  Workspace name used by Neptune. Default is the `Peltarion` workspace. Assumes you have setup a Neptune account using your Peltarion email.


<p align="right">(<a href="#top">back to top</a>)</p>
