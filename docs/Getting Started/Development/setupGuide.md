<!-- # Local Setup Instructions

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

This section describes how to set up LakeFS and Minio locally to version control your data and upload/download data from lakeFS repositories.

### Prerequisites

The docker compose file to make docker containers for LakeFS and Minio is already provided in the repository, you can find the `docker-compose.yml` [here](../../../notebooks/docker-compose.yml)

Our LakeFS endpoint is: `http://<server-ip>:8000` and the Minio endpoint is: `http://<server-ip>:9001`.

You can create a minio bucket via the Minio web interface or via the Minio client. Here we will describe how to create a Minio bucket via the Minio client.

### Minio Client Setup

#### For Windows (PowerShell)

```powershell
# Download the Minio client
iwr https://dl.min.io/client/mc/release/windows-amd64/mc.exe -OutFile $env:USERPROFILE\mc.exe

# Add the Minio client to the PATH environment variable
$env:PATH += ";" + $env:USERPROFILE

# Verify the installation
mc.exe --version

# Configure the Minio client with an alias, the access key and the secret key
mc.exe alias set myminio http://100.97.85.39:9000 <access_key> <secret_key>
```

#### For Linux (Bash)

```bash
# Download the Minio client
curl -O https://dl.min.io/client/mc/release/linux-amd64/mc

# Make it executable and move to a directory in PATH
chmod +x mc
sudo mv mc /usr/local/bin/

# Verify the installation
mc --version

# Configure the Minio client with an alias, the access key and the secret key
mc alias set myminio http://100.97.85.39:9000 <access_key> <secret_key>

```

To create a Minio bucket, run the following command in the terminal:

```bash
mc mb myminio/<bucket_name>
```

### LakeFS CLI Setup

We will now install the LakeFS CLI to interact with LakeFS repositories on our local LakeFS server.

#### For Windows (PowerShell)

```powershell
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

#### For Linux (Bash)

```bash
# Download the LakeFS CLI
curl -L https://github.com/treeverse/lakeFS/releases/download/v1.71.0/lakeFS_1.71.0_Linux_x86_64.tar.gz -o lakectl.tar.gz

# Extract and move to a directory in PATH
tar -xzf lakectl.tar.gz
chmod +x lakectl
sudo mv lakectl /usr/local/bin/

# Verify the installation
lakectl --version

# remove the downloaded tar file
rm lakectl.tar.gz

# Configure the LakeFS CLI with the endpoint, access key and secret key
lakectl config
```

To create a LakeFS repository linked to the Minio bucket you created earlier, run the following command in the terminal:

```bash
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

## Cloning a LakeFS repository
To clone a LakeFS repository locally on your machine, you can use the following command:
```shell
lakectl local clone lakefs://<repo_name> <local_directory>
```
Replace `<repo_name>` with the name of the LakeFS repository you want to clone and `<local_directory>` with the path to the local directory where you want to clone the repository.
This will create a local copy of the LakeFS repository in the specified directory.
If you want to see the changes made to the repository, you can use the following command:
```shell
lakectl status <local_directory>
``` -->

# Lokale Setup & Data Versioning

## Inleiding

Dit document beschrijft hoe je de Python-omgeving opzet en hoe je LakeFS en Minio lokaal kunt draaien voor data-versioning.

We splitsen dit op in vier duidelijke stappen:

- De Services Starten: LakeFS & Minio via Docker.

- De Services Configureren: Je lokale clients (CLIs) instellen om met de services te praten.

- De Python Scripts Draaien: Twee opties om je code uit te voeren.

- Basis Workflow: Hoe je een repository clonet.

## Vereisten

Zorg dat je de volgende tools hebt geïnstalleerd voordat je begint:

- Python 3.12

- Docker

- Docker Compose

- Git

## Start de Core Services (LakeFS & Minio)

Dit is de basis. We gebruiken één docker-compose.yml bestand om zowel de LakeFS- als de Minio-server te starten. Dit bestand start ook een Python/Jupyter-container die je later kunt gebruiken (zie Deel 3, Optie B).

Clone de repository (als je dat nog niet hebt gedaan):

```Shell
git clone <repository_url>
```

Navigeer naar de notebooks map (waar de docker-compose.yml staat):

```Shell
cd /pad/naar/je/repo/notebooks
```

Maak een .env bestand aan vanuit het template. Hierin staan je keys en wachtwoorden.

```Shell
cp .env.template .env
```

Belangrijk: Open het .env bestand en pas de variabelen (zoals MINIO_ACCESS_KEY en MINIO_SECRET_KEY) aan naar wens.

Start alle services met Docker Compose:

```Shell

docker compose up -d
```

Je services zijn nu (lokaal) bereikbaar:

- LakeFS Web UI: http://localhost:8000/setup

  - Volg de setup-instructies in de browser om je eerste admin-gebruiker aan te maken.

- Minio Web UI: http://localhost:9001

  - Log in met de MINIO_ACCESS_KEY en MINIO_SECRET_KEY uit je .env bestand.

## Configureer de Services (Eerste Gebruik)

Nu de servers draaien, moeten we ze vertellen wat ze moeten doen. Hiervoor installeren en configureren we de lokale command-line clients (mc voor Minio en lakectl voor LakeFS).

### Minio Client (mc) Setup

Met de mc client maken we de storage bucket waar LakeFS zijn data zal opslaan.

#### Installatie

Windows (PowerShell):

```PowerShell

# Download de Minio client
iwr https://dl.min.io/client/mc/release/windows-amd64/mc.exe -OutFile $env:USERPROFILE\mc.exe

# Voeg toe aan je PATH (voor deze sessie)
$env:PATH += ";" + $env:USERPROFILE

# Verifieer
mc.exe --version

```

Linux (Bash):

```Bash

# Download de Minio client
curl -O https://dl.min.io/client/mc/release/linux-amd64/mc

# Maak uitvoerbaar en verplaats
chmod +x mc
sudo mv mc /usr/local/bin/

# Verifieer
mc --version
```

#### Configuratie & Bucket Aanmaken

Configureer een alias: Vertel mc waar je Minio-server draait. Vervang <access_key> en <secret_key> met de waarden uit je .env bestand.

```Bash

# We gebruiken poort 9000, de standaard API-poort voor Minio
mc alias set myminio http://localhost:9000 <access_key> <secret_key>
```

Maak een bucket: Dit is de opslaglocatie voor je data.

```Bash

mc mb myminio/<bucket_name>
```

(Vervang <bucket_name> door een naam, bijv. lakefs-storage)

### LakeFS Client (lakectl) Setup
  
Met de lakectl client maken we een LakeFS repository die gelinkt is aan de Minio-bucket.

#### Installatie

Windows (PowerShell):

```PowerShell

# Download de LakeFS CLI (v1.71.0)
$zip = "$env:TEMP\lakectl.zip"
iwr https://github.com/treeverse/lakeFS/releases/download/v1.71.0/lakeFS_1.71.0_Windows_x86_64.zip -OutFile $zip

# Unzip naar een map
$dst = "$env:USERPROFILE\bin\lakectl"
mkdir $dst -Force | Out-Null
Expand-Archive $zip -DestinationPath $dst -Force

# Voeg toe aan je PATH (permanent)
$env:Path = "$env:Path;$dst"
setx PATH "$($env:Path)"

# Herstart je terminal en verifieer
lakectl --version
```

Linux (Bash):

```Bash

# Download de LakeFS CLI (v1.71.0)
curl -L https://github.com/treeverse/lakeFS/releases/download/v1.71.0/lakeFS_1.71.0_Linux_x86_64.tar.gz -o lakectl.tar.gz

# Uitpakken en verplaatsen
tar -xzf lakectl.tar.gz
chmod +x lakectl
sudo mv lakectl /usr/local/bin/

# Opruimen en verifiëren
rm lakectl.tar.gz
lakectl --version
```

#### Configuratie & Repository Aanmaken

Configureer de client: Dit start een interactieve wizard.

```Bash
lakectl config
```

Access Key & Secret Key: Gebruik de credentials van de admin-gebruiker die je in Start de core services (via de Web UI) hebt aangemaakt alsook het eindpoint die je daar hebt gebruikt.

#### Maak een repository: Link LakeFS aan je Minio-bucket

```Bash

lakectl repo create lakefs://<repo_name> s3://<bucket_name>
```

Vervang <repo_name> door je gewenste reponaam (bijv. mijn-data-project).

Vervang <bucket_name> door de Minio-bucketnaam (bijv. lakefs-storage).

Je bent nu klaar! De services draaien en zijn geconfigureerd.

## De Python Scripts Draaien (Kies je Methode)

Je hebt twee opties om de Python-scripts uit te voeren. Kies er één.

### Optie A: Lokaal op je Machine (met Virtual Environment)

Gebruik deze methode als je de scripts direct op je eigen besturingssysteem wilt draaien.

Maak een virtual environment:

```Bash

# Windows
py -3.12 -m venv .\.venv

# Linux
python3.12 -m venv ./.venv`
```

Activeer de virtual environment:

```Bash

# Windows
.\.venv\Scripts\activate

# Linux
source ./.venv/bin/activate
```

Installeer de dependencies:

```Bash

pip install -r requirements.txt
```

Draai je script:

```Bash

python your_script.py
```

### Optie B: Via Docker (met de Ingebouwde Jupyter/Python Container)

Gebruik deze methode als je liever binnen de geïsoleerde Docker-omgeving werkt. De docker compose up opdracht uit Start de Core Services heeft deze container al voor je gestart.

#### Jupyter Notebook Interface

Navigeer in je browser naar: http://localhost:8888

Je hebt een token nodig. Vind deze door de logs van de container te bekijken:

```Shell

# Zoek de containernaam (bijv. 'notebooks-jupyter-1')
docker ps

# Vraag de logs op (vervang containernaam)
docker logs <jupyter_container_name>
```

Kopieer de token uit de logs (het deel na ?token=...) en plak dit in je browser.

#### Direct scripts uitvoeren (via docker exec) Je kunt ook direct een commando uitvoeren binnen de draaiende Python-container:

```Shell
docker exec -it <jupyter_container_name> python your_script.py
```

(Vervang <jupyter_container_name> en your_script.py.)

## Typische Workflow met LakeFS

Nu alles is opgezet, kun je de lakectl client gebruiken om met je data te werken, vergelijkbaar met Git.

Clone een repository lokaal: Dit "mount" de LakeFS-repository als een lokale map, zodat je bestanden kunt zien en bewerken.

```Bash

lakectl local clone lakefs://<repo_name> <local_directory>
```

Vervang <repo_name> door de naam die je in stap 2b hebt gemaakt.

Vervang <local_directory> door een mapnaam (bijv. ./mijn-data).

Check de status: Nadat je bestanden hebt toegevoegd of gewijzigd in de <local_directory>, kun je de status zien:

```Bash
# Controleer de status in de lokale directory
lakectl local status <local_directory>

# Maak een commit met een bericht ( . alle wijzigingen in de directory)
lakectl local commit . -m "Je commit bericht hier"

# Pull de laatste wijzigingen van de remote repository
lakectl local pull <local_directory>

# Maak een nieuwe branch
lakectl branch create lakefs://<repo_name>/<branch_name> --source lakefs://<repo_name>/<source_branch>

# Wissel van branch (binnen een geclonede directory)
lakectl local checkout <local_directory> --ref lakefs://<repo_name>/<branch_name>

```

Voor meer commando's (zoals commit, push, merge), raadpleeg de LakeFS [documentatie](https://docs.lakefs.io/v1.60/howto/local-checkouts/)
