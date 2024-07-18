import subprocess
import sys
import os

def run_command(command):
    """Run a command in the shell and return the output."""
    try:
        # print(f"Running command: {command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Command output: {e.output}")
        print(f"Command error: {e.stderr}")
        sys.exit(1)

def conda_env_exists(env_name):
    """Check if a Conda environment exists."""
    command = "conda env list"
    output = run_command(command)
    envs = [line.split()[0] for line in output.splitlines() if line]
    return env_name in envs

def create_conda_env(env_name):
    """Create a new Conda environment with Python 3.12.2 and cctbx-base."""
    command = f"conda create --name {env_name} python=3.12.2 cctbx-base -c conda-forge -y"
    print(f"Creating conda environment: {env_name}")
    run_command(command)

def main():
    # Change the current working directory to the directory where the executable is located
    script_dir = os.path.dirname(os.path.abspath(sys.executable))
    os.chdir(script_dir)
    
    env_name = "bkind_env"

    if conda_env_exists(env_name):
        # print(f"Conda environment '{env_name}' exists.")
        pass
    else:
        print(f"Conda environment '{env_name}' does not exist but is created. Rerun the program.")
        create_conda_env(env_name)

    # Run the shell script
    run_command("./run_bkind.sh")

if __name__ == "__main__":
    main()
