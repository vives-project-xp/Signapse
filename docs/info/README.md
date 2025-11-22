# Info Folder

This folder contains reference materials for the key tools used in this project: **Node.js, npm, npx, Expo CLI, and EAS CLI**.

## Purpose

The `info` folder serves as a centralized location for **in-depth explanations, best practices, and usage tips** for each tool. Detailed documentation is provided in the respective markdown files.

## Folder Contents

| File         | Description                                                    |
| ------------ | -------------------------------------------------------------- |
| `nodejs.md`  | Detailed explanation of Node.js concepts and workflows         |
| `npm-npx.md` | Guide to npm and npx usage, scripts, and best practices        |
| `expo.md`    | Guides and tips for working with Expo projects                 |
| `eas.md`     | Instructions and recommendations for using EAS CLI effectively |

## Usage

* Use this folder as a **reference library** when learning, troubleshooting, or understanding how each tool fits into the project.
* Keep the documentation up-to-date to reflect the latest versions and best practices.
* For installation or setup instructions, see the [`Getting Started`](../Getting%20Started/README.md)folder.

!!! note "NODE.JS Information"
    See [node.js.md](./node.js.md) for a details on installing and using Node.js.

!!! note "NPM/NPX CLI Information"
    See [npm-npx.md](./npm-npx.md) for details on installing and using npm and npx.

!!! note "EXPO CLI Information"
    See [expo.md](./expo.md) for details on installing and using the Expo Development CLI.

!!! note "EAS CLI Information"
    See [eas-cli.md](./eas-cli.md) for details on installing and using the Expo Application Services CLI.

## project Structure

A visual representation of the core tools and how they interact in this project:

````mermaid
flowchart TD
 A[Node.js<br>Runtime & npm] --> B[npm<br>Package Management]
 A --> C[npx<br>Execute Binaries] 
 B --> D[Expo CLI<br>React Native Dev]
 C --> D 
 D --> E[EAS CLI<br>Build & Deploy] 

classDef tool fill:#f9f,stroke:#333,stroke-width:2px;
class A,B,C,D,E tool;
````
## Additional Resources

* [Node.js Documentation](https://nodejs.org/en/docs/)
* [npm Documentation](https://docs.npmjs.com/)
* [npx Documentation](https://www.npmjs.com/package/npx)
* [Expo Documentation](https://docs.expo.dev/)
* [EAS CLI Documentation](https://docs.expo.dev/eas/)
