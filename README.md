# Stroop Script
Last Updated: 2-11-25

## Setup:
1. Install dependencies if necessary.  

    To check to see if python is installed, run the following command in terminal: 
* python3 --version  

    Install git from the  [official git webpage](https://git-scm.com/).

2. Clone the Repository and change directories into the repository.  

    In terminal, run the following commands (one at a time): 
* git clone https://github.com/kzsf/stroop.git
* cd /path/to/repo

3. Set up virtual environment and install dependencies from requirements.txt.  

    In terminal, run the following commands: 
* python -m venv venv
* source venv/bin/activate
* pip install -r requirements.txt

4. Set up the config file with your input and output paths.  
    Open the paths.yml file.  
    Change the 'input_path' and 'output_path' variables to your input and output file paths.  
    Save the file.

## Run the script: 
In terminal, run the following command:
* python3 stroop_script.py