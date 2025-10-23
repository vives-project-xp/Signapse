# Local Setup Instructions
## Running the python scripts locally
This section describes how to run the python scripts locally on your machine. For this, you need to have Python 3.12 installed on your machine. You can download it from [here](https://www.python.org/downloads/release/python-3120/).

## Make a virtual environment
To make sure all dependencies are installed correctly, it is recommended to use a virtual environment. You can create a virtual environment in windows or linux by running the following command in the terminal:
```
# For Windows
py -3.12 -m venv .\.venv
# For Linux
python3.12 -m venv ./.venv
```
## Activate the virtual environment
To activate the virtual environment, run the following command in the terminal:
```
# For Windows
.\.venv\Scripts\activate
# For Linux
source ./.venv/bin/activate
```
## Install the dependencies
To install the dependencies, run the following command in the terminal:
```
# For Windows and Linux
pip install -r requirements.txt
```
## Run the scripts
To run the scripts, you can use the following command in the terminal:
```
# For Windows and Linux
python your_script.py
```

## Using LakeFS and Minio locally
This section describes how to set up LakeFS and Minio locally to version control your data.

Our LakeFS endpoint is: `http://100.97.85.39:8000` and the Minio endpoint is: `http://100.97.85.39:9001`.

To create a Minio bucket on the server, you need to have the Minio client installed. You can download it from [here](https://min.io/docs/minio/linux/reference/minio-mc.html).

Create a Minio bucket in the CLI with the following command:
```
# Download the Minio client
iwr https://dl.min.io/client/mc/release/windows-amd64/mc.exe -OutFile $env:USERPROFILE\mc.exe

# Add the Minio client to the PATH environment variable
$env:PATH += ";" + $env:USERPROFILE

# Verify the installation
mc.exe --version
```
Then, configure the Minio client with and alias, the access key and the secret key:
```
mc.exe alias set myminio http://100.97.85.39:9000 <access_key> <secret_key>
```
We want to version control the Minio bucket with LakeFS. Therefore, we need to create a LakeFS repository that is connected to the Minio bucket. To do this, you need to have the LakeFS CLI installed. You can download it from [here](https://github.com/treeverse/lakeFS/releases), or run the following command in powershell:
```
# Download the LakeFS CLI
$zip = "$env:TEMP\lakectl_windows_amd64.zip"
iwr https://github.com/treeverse/lakeFS/releases/download/v1.70.1/lakeFS_1.70.1_Windows_x86_64.zip -OutFile $zip

# Unzip the LakeFS CLI to the user's bin directory
$dst = "$env:USERPROFILE\bin\lakectl"
mkdir $dst -Force | Out-Null
Expand-Archive $zip -DestinationPath $dst -Force

# Add the user's bin directory to the PATH environment variable
$env:Path = "$env:Path;$dst"
setx PATH "$($env:Path)"

# Verify the installation
lakectl --version

# Configure the LakeFS CLI with the endpoint, access key and secret key
lakectl config 
```
Now we have both the Minio client and the LakeFS CLI installed. We can now create a LakeFS repository that is connected to the Minio bucket. Run the following command in the terminal:
```
mc mb myminio/<bucket_name>
lakectl repo create lakefs://<repo_name> s3://<bucket_name>
```
Replace `<bucket_name>` with the name of the Minio bucket you created and `<repo_name>` with the name of the LakeFS repository you want to create.
You can now create branches, commits and other version control operations on the LakeFS repository. For more information, you can check the [LakeFS documentation](https://docs.lakefs.io/).

# Server Setup Instructions
In this section, we will describe how to use the existing Docker setup to run the LakeFS and Minio services along with a Python 3.12 environment for running the scripts on a server.
## Prerequisites
- Docker
- Docker Compose
## Running the Docker Compose setup
To run the Docker Compose setup, follow these steps:
1. Clone the repository:
```shell
git clone <repository_url>
```
2. Navigate to the `notebooks` directory:
```shell
cd notebooks
```
3. Make a .env file with the necessary environment variables by copying the example file:
```shell
cp .env.template .env
```
4. Set the environment variables in the `.env` file according to your setup.
5. Start the Docker Compose setup:
```shell
docker compose up -d
```
This will start the LakeFS, Minio and Python 3.12 services in detached mode.
## Accessing Minio 
You can access the Minio web interface by navigating to `http://<server_ip>:9001` in your web browser. Use the access key and secret key set in the `.env` file to log in.
## Accessing LakeFS
You can access the LakeFS web interface by navigating to `http://<server_ip>:8000/setup` in your web browser. You will be prompted to create an admin user for LakeFS. After creating the admin user, you can log in using the credentials you just recived.
## Accessing Jupyter Notebook
You can access the Jupyter Notebook interface by navigating to `http://<server_ip>:8888` in your web browser. You will be prompted to enter a token for authentication. You can find the token by running the following command in the terminal:
```shell
docker logs <jupyter_container_name>
```
Replace `<jupyter_container_name>` with the name of the Jupyter container. The token will be displayed in the logs.
## Running Python scripts in the Docker container
To run Python scripts in the Docker container, you can use the following command:
```shell
docker exec -it <container_name> python <script_name.py>
```