#!/usr/bin/env bash

set -e

function load_modules_baobab() {
    # Load modules
    module load Anaconda3/5.3.0
    module load GCC/8.3.0 OpenMPI/3.1.4
    module load Mesa/19.2.1

    # FUNCTIONS
    # conda
    # shellcheck source=/dev/null
    . /opt/ebsofts/Core/Anaconda3/5.3.0/etc/profile.d/conda.sh
}

function install_conda_env() {
    ENV_NAME=$1
    conda create -y -n "$ENV_NAME" python=3.7
}

function activate_conda_env() {
    ENV_NAME=$1
    conda activate "$ENV_NAME"
}

function install_requirements() {
    # shellcheck disable=SC2094
    while read -r requirement; 
    do
        if [ "$requirement" ] && [[ $requirement != "#"* ]];
        then
            echo "$requirement"
            if [[ "$requirement" == "mpi4py"* ]];
            then
                conda install --yes -c conda-forge "$requirement";
            # elif [[ "$requirement" == "torch"* ]];
            # then
            #      conda install --yes -c pytorch "py$requirement";
            elif [[ "$requirement" == "mujoco-py"* && "$1" = "mujoco" ]];
            then
                echo "installing mujoco-py"
                rm -rf /opt/conda/lib/python*/site-packages/mujoco_py*
                git clone https://github.com/jacr13/mujoco-py.git
                cd mujoco-py
                # cd mujoco-py || exit
                # shellcheck disable=SC2094
                pip install --no-cache-dir -r requirements.txt
                pip install --no-cache-dir -r requirements.dev.txt
                python setup.py build install
                cd ..
                rm -r mujoco-py
            else
                echo "piiiiip $requirement"
                pip install "$requirement";
            fi;
        fi; 
    done < requirements.txt
}

function install_mujoco() {
    # MUJOCO
    # Install mujoco
    PATH_MUJOCO=$HOME
    if [[ $1 == "docker" ]];
    then
        PATH_MUJOCO=/opt/mujoco
    fi

    if [ -d "$PATH_MUJOCO/.mujoco" ] 
    then
        echo ""
        echo "$PATH_MUJOCO/.mujoco already exists, we will skip this step."
    else
        mkdir -p $PATH_MUJOCO/.mujoco
        curl https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -L -o mujoco.tar.gz
        tar -xf mujoco.tar.gz -C $PATH_MUJOCO/.mujoco
        rm mujoco.tar.gz

        if [ -f "$(pwd)/mjkey.txt" ]
        then
            cp "$(pwd)/mjkey.txt" $PATH_MUJOCO/.mujoco/mjkey.txt
        else
            echo "ERROR: mjkey.txt is not present in this directory."
            echo "Please add mjkey.txt file in the same directory as this script."
            return
        fi
        if [ -f "$PATH_MUJOCO/.bashrc" ]
        then
            # shellcheck disable=SC2016
            grep -qxF 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PATH_MUJOCO/.mujoco/mujoco210/bin' $PATH_MUJOCO/.bashrc || echo 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PATH_MUJOCO/.mujoco/mujoco210/bin' >> "$PATH_MUJOCO"/.bashrc
            # shellcheck source=/dev/null
            source "$PATH_MUJOCO"/.bashrc
        fi
    fi
}

function install_myo() {
    git clone --recursive https://github.com/jacr13/myosuite.git
    cd myosuite
    pip install -v --no-cache-dir -r requirements.txt
    python setup.py build install
    python myosuite/tests/test_myo.py
    cd ..
    rm -r myosuite
}

function setup_conda_env() {
    # CONDA
    # Create conda env and install dependencies
    # check for the presence of the env, if exists ask to overwrite
    ENV_NAME=$1
    if [ -d "$HOME/.conda/envs/$ENV_NAME" ] 
    then
        echo ""
        read -p "$ENV_NAME already exists, would you like to overwrite it (y/n)? " -n 1 -r
        if [[ $REPLY =~ ^[Yy]$ ]]
        then
            conda deactivate "$ENV_NAME"
            conda remove -n "$ENV_NAME" --all
            install_conda_env "$ENV_NAME"
            activate_conda_env "$ENV_NAME"
            install_requirements
        else
            activate_conda_env "$ENV_NAME"
            install_requirements
        fi
    else
        install_conda_env "$ENV_NAME"
        activate_conda_env "$ENV_NAME"
        install_requirements
    fi

    echo "Which python: $(which python)"
    python -c "import mujoco_py; import os; discovered_path = mujoco_py.utils.discover_mujoco(); discovered_path = discovered_path[0] if isinstance(discovered_path, tuple) else discovered_path; print(discovered_path); sim = mujoco_py.MjSim(mujoco_py.load_model_from_path(os.path.join(discovered_path, 'model', 'humanoid.xml'))); print(sim.data.qpos); sim.step(); print(sim.data.qpos);"

    echo "Setup is complete!"
    echo "You can now activate the conda environment using:"
    echo "    $ conda activate $ENV_NAME"
}