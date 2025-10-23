# Running the python scripts
This notebook is used to run the python scripts in this repository. It is mainly used for testing and debugging purposes.
It is not meant to be used as a final product.

These notebooks assume you have python 3.12 installed on your system. You can download it from [here](https://www.python.org/downloads/release/python-3120/).

## Make a virtual environment
To make sure all dependencies are installed correctly, it is recommended to use a virtual environment. You can create a virtual environment in windows by running the following command in the terminal:
```
py -3.12 -m venv .\.venv
```

## Activate the virtual environment
To activate the virtual environment, run the following command in the terminal:
```
.\.venv\Scripts\activate
```
## Install the dependencies
To install the dependencies, run the following command in the terminal:
```
pip install -r requirements.txt
```
## Run the scripts
To run the scripts, you can use the following command in the terminal:
```
py -3.12 <script_name>.py