#!/usr/bin/env bash

set -e

# change the account name
DOCKER_ACC="candidj0"
DOCKER_REPO="milozero"
# change the image tag ["mujoco", "envpool", "opensim", "latest"] to choose the image to build

if [ -z "$1" ]; then
    IMG_TAG="latest"
    echo "No argument supplied using $IMG_TAG as default"
else
    IMG_TAG="$1"
    echo "Argument supplied: $IMG_TAG"
fi


function test_mujoco() {
    docker run "$DOCKER_ACC/$DOCKER_REPO:$IMG_TAG" python -c "import mujoco_py; import os; discovered_path = mujoco_py.utils.discover_mujoco(); discovered_path = discovered_path[0] if isinstance(discovered_path, tuple) else discovered_path; print(discovered_path); sim = mujoco_py.MjSim(mujoco_py.load_model_from_path(os.path.join(discovered_path, 'model', 'humanoid.xml'))); print(sim.data.qpos); sim.step(); print(sim.data.qpos); print(mujoco_py.__file__); print(mujoco_py.__version__);"
}


function test_opensim() {
    docker run -v "$(dirname "$(pwd)")"/utils:/workspace/utils \
               -v "$(dirname "$(pwd)")"/data:/workspace/data \
               -v "$(pwd)"/test:/workspace \
               "$DOCKER_ACC/$DOCKER_REPO:$IMG_TAG"  python test_opensim.py
}


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"|| exit
cp ../requirements.txt .

# docker login --username=$DOCKER_ACC
docker build  --rm --no-cache -t "$DOCKER_ACC/$DOCKER_REPO:$IMG_TAG" -f "Dockerfile.$IMG_TAG" .
rm requirements.txt

if [[ $IMG_TAG == "mujoco" ]]; then
    test_mujoco
elif [[ $IMG_TAG == "opensim" ]]; then
    test_opensim
elif [[ $IMG_TAG == "latest" ]]; then
    test_mujoco
    test_opensim
else
    echo "The image tag $IMG_TAG is not known."
fi

docker push "$DOCKER_ACC/$DOCKER_REPO:$IMG_TAG"
