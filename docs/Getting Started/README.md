# Getting Started

This guide will walk you through setting up the SmartGlasses development environment on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Git**: Version control system ([Download](https://git-scm.com/downloads))
- **Visual Studio Code**: Code editor ([Download](https://code.visualstudio.com/))
- **Docker Desktop**: Container platform ([Download](https://www.docker.com/products/docker-desktop/))
- **VS Code Dev Containers Extension**: For development container support ([View on Marketplace](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers))

### Installing VS Code Extensions

1. Open Visual Studio Code
2. Go to Extensions (Ctrl+Shift+X / Cmd+Shift+X)
3. Search for and install: [**Dev Containers** by Microsoft](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

## Step 1: Clone the Repository

Clone the SmartGlasses repository to your local machine:

```bash
git clone https://github.com/vives-project-xp/SmartGlasses.git
cd SmartGlasses
```

## Step 2: Open the Workspace

The project uses a VS Code workspace file to organize the multi-folder structure:

1. Open Visual Studio Code
2. Use one of these methods:
   - **File** > **Open Workspace from File** > Select `SmartGlasses.code-workspace`
   - Or from command line: `code SmartGlasses.code-workspace`
   - Or in file explorer, right-click `SmartGlasses.code-workspace` and choose "Open with Code"

The workspace includes three main folders:

- **Root**: Main project folder with documentation and configuration
- **Server**: FastAPI backend service
- **Client**: Expo React Native mobile application

## Step 3: Start the Development Container

The project uses Docker dev containers for a consistent development environment:

**Automatic Setup**:

1. When you open the workspace, VS Code should automatically detect the `.devcontainer` configuration and prompt you to "Reopen in Container"

**Manual Setup**:

1. If not prompted automatically:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Dev Containers: Rebuild and Reopen in Container"
   - Select this option

!!! note
    The initial container build may take 5-10 minutes as it:

    - Downloads the Python 3.12 base image
    - Installs development tools (Node.js, Android SDK, FFmpeg, Docker CLI)
    - Configures VS Code extensions
    - Sets up the development environment

## Step 4: Verify Installation

Once the container is running, verify everything is set up correctly:

### Check Python Environment

```bash
python3 --version
pip3 --version
```

### Check Node.js Environment

```bash
node --version
npm --version
```

### Check Docker Access

```bash
docker --version
```

## Step 5: Install Project Dependencies

### Server Dependencies (FastAPI)

!!! note "Next Steps"
    See [Server setup & run](./Development/server.md) for the server setup and how to run the FastAPI backend.

### Client Dependencies (React Native/Expo)

!!! note "Next Steps"
    See [Client setup & run](./Development/Client/README.md) for the client setup and how to start the Expo development server.

### Notebook Dependencies (Optional - for AI/ML development)

!!! warning "Next Steps"
    See [Notebooks setup & run](./Development/notebooks.md) for details on installing notebook dependencies and running notebooks.

## Step 6: Building

!!! note "Next Steps"
    See [Building the Application](./Development/buildingGuide.md) for instructions on how to build and run the Signapse application.


