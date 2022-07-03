# Use the Stable-baselines 3 with the deepbots enviroments


## Dependencies
### Without conda:
```bash
# create a virtual environment
python -m venv venv
# activate it
source venv/bin/activate
# install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### With conda:
```bash
# create a virtual environment and install requirements
conda create -y -n venv python=3.8
# activate it
conda activate venv
# install requirements
while read requirement; do if [[ $requirement != "#"* ]]; then conda install -c conda-forge --yes $requirement || pip install $requirement; fi; done < requirements.txt
```