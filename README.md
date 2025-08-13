# Foundation
Neuronal recordings and Foundation models

## Description

This repository covers the training and analysis pipeline used for the foundation model described in [Wang et al., 2025](https://www.nature.com/articles/s41586-025-08829-y). While the code is available for public inspection, it is intended to run alongside lab infrastructure that is not publicly accessible. Core methods used here are imported from the [FNN](https://github.com/cajal/fnn) repository, which provides model architecture code, publicly released trained weights, and tutorials on how to fine-tune the foundation model to new data.

To inspect the state of the repo used for [Wang et al., 2025](https://www.nature.com/articles/s41586-025-08829-y), see the tag: `nature_v1`.

## Prerequisites
- Docker >= 20.10
- Docker Compose v2+

For GPU support:
- NVIDIA GPU + drivers compatible with CUDA 11.8+
- NVIDIA Container Toolkit

## Installation
### 1. Clone the repository

```bash
git clone https://github.com/cajal/foundation.git
```

### 2. Navigate to the `docker` directory
```bash
cd foundation/docker
```

### 3. Create an `.env` file. 

The `.env` file can be empty, it just needs to exist at the same level as `docker-compose.yml`.

The following lines can be added to customize the container (replace * with your own values):

**Database access:**
```
DJ_HOST=*             # database host                   if omitted, no database host will be specified
DJ_USER=*             # database username               if omitted, no database user will be specified
DJ_PASS=*             # database password               if omitted, no database password will be specified
```

**Jupyter:**
```
JUPYTER_TOKEN=*       # your desired password;          if omitted, there is no password prompt
JUPYTER_PORT_HOST=*   # your desired port on host;      if omitted, defaults to: 8888
```

**Image source & tag:**
```
IMAGE_REGISTRY=*      # your local registry;            if omitted, defaults to: ghcr.io
IMAGE_NAMESPACE=*     # desired namespace;              if omitted, defaults to: cajal
IMAGE_NAME=*          # desired image name;             if omitted, defaults to: foundation
IMAGE_TAG=*           # desired image tag (e.g. dev);   if omitted, defaults to: latest
```

### 4. Specify the Docker image

Docker compose will launch the container from the image tag specified by the `IMAGE_` environment variables with the format:

```
IMAGE_REGISTRY/IMAGE_NAMESPACE/IMAGE_NAME:IMAGE_TAG
```

By default, (i.e. if no environment variables are provided) the image tag will resolve to: `ghcr.io/cajal/foundation:latest`. 

### 5. Launch the Docker container

To pull the image from the registry and then launch the container (RECOMMENDED):

```bash
docker compose up -d foundation --no-build --pull always
```

To build the image locally and then launch the container:

```bash
docker compose up -d foundation --build
```

To launch without GPU support, replace `foundation` with `foundation-cpu`

### 6. Access the container

Jupyter lab can be accessed at: `http://<host-ip>:<JUPYTER_PORT_HOST>/`, where `JUPYTER_PORT_HOST` defaults to `8888`.

If `JUPYTER_TOKEN` is set, use it to authenticate.

## Additional Info

**Legacy image:**
To use the image that is closest to [Wang et al., 2025](https://www.nature.com/articles/s41586-025-08829-y)

```
IMAGE_REGISTRY=ghcr.io
IMAGE_NAMESPACE=cajal
IMAGE_NAME=foundation
IMAGE_TAG=nature_v1
```

Note: The `foundation:nature_v1` image installed`scipy==1.10.1` with `scikit-image==0.20.0` and `Python 3.8`, a combination that the newer libmamba solver rejects. This package incompatibility is resolved in `foundation:latest`.

To inspect the Dockerfile for the `nature_v1` image, check the `nature_v1` tag of this repository.
