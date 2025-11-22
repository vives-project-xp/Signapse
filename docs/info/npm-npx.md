!!! warning This file serves as a reference for understanding npm and npx. For installation or setup, see the [Getting Started](../Getting%20Started/README.md) folder.

# npm & npx — Info

## What is npm?

* **npm** (Node Package Manager) is the default package manager for Node.js.
* It consists of three main components:

   - **Website**: [npmjs.com](https://www.npmjs.com/) — discover packages, manage profiles, and organizations.
   - **CLI (Command Line Interface)**: Use it in the terminal to install, publish, and manage packages.
   - **Registry**: Public database of JavaScript packages and their metadata.

## Why use npm?

* Install reusable code (packages) in your applications or share your own.
* Manage tools required in your project (linters, build tools, test frameworks).
* Run CLI tools without installing them globally using **npx**.
* Share code publicly or privately using npm organizations.
* Control dependency versions and update your applications safely.
* Discover solutions via the vast number of existing packages.

---

## How npm works in practice

### Installing Packages

* Install all dependencies in a project with a `package.json`:

```bash
npm install
```

* Install a specific package and version:

```bash
npm install <package-name>@<version>
```

### Running Scripts

Define scripts in `package.json`:

```json
"scripts": {
  "start": "node index.js",
  "test": "jest"
}
```

Execute scripts via CLI:

```bash
npm run start
npm run test
```
---

### Using npx

* **npx** allows running Node.js binaries without global installation:

```bash
npx create-react-app my-app
```

* Useful for one-off commands or temporary CLI tools.

## Sharing Packages

* **Public packages**: Available to the entire npm community.
* **Private packages**: Only accessible to your team (requires npm organizations).
* Optionally, use private registries like Verdaccio for internal packages.

## Best Practices

* Follow **semantic versioning** to control dependency updates.
* Define useful scripts in `package.json` for consistent workflows.
* Use npm organizations for team collaboration and private packages.
* Keep `package-lock.json` to maintain consistent environments across machines.

---

## package.json vs package-lock.json

### 1. package.json

**Purpose:**
Main configuration file for a Node.js project. Defines project metadata, dependencies, and scripts.

**Contents:**

* Project name, version, description
* Dependencies and devDependencies (e.g., `"react": "^18.2.0"`)
* Scripts (e.g., `"start": "node index.js"`)
* Other metadata: repository, author, license

**Key point:**
Dependencies often use **version ranges** with `^` or `~`. Example:

```json
"react": "^18.2.0"
```

This allows npm to install any compatible version ≥18.2.0 and <19.0.0.

---
### 2. package-lock.json

**Purpose:**
Ensures **exact versions** of installed packages, including nested dependencies, for consistent installs across machines.

**Contents:**

* Exact versions of all dependencies and their dependencies
* Nested dependency resolutions
* SHA-1 hashes for integrity verification

**Key point:**
Generated and updated automatically by npm when running `npm install`. Guarantees that everyone working on the project gets the same dependency tree.

---

### Short Comparison

| Feature     | package.json                            | package-lock.json                           |
| ----------- | --------------------------------------- | ------------------------------------------- |
| Purpose     | Project metadata, dependencies, scripts | Exact dependency tree and versions          |
| Manual edit | Yes                                     | No, automatically managed by npm            |
| Versioning  | Flexible ranges (`^`, `~`)              | Exact installed versions                    |
| Consistency | Not guaranteed                          | Ensures everyone installs the same versions |

💡 **Tip:**

* Never manually edit `package-lock.json`.
* Use `package.json` to add or remove dependencies.
* Let npm manage the lock file to avoid installation issues.


## Additional Resources

* [npm CLI Documentation](https://docs.npmjs.com/)
* [npm Registry](https://www.npmjs.com/)
* [Node.js npm Introduction](https://nodejs.org/docs/latest/api/documentation.html)

# npm CLI Commands — Reference

## 1. Installing Packages (`npm install`)

| Flag / Option           | Description                                                          |
| ----------------------- | -------------------------------------------------------------------- |
| `-D`, `--save-dev`      | Saves the package into `devDependencies`.                            |
| `-P`, `--save-prod`     | Saves the package into `dependencies` (production).                  |
| `-O`, `--save-optional` | Saves the package into `optionalDependencies`.                       |
| `-E`, `--save-exact`    | Saves the exact version without a semver range.                      |
| `-B`, `--save-bundle`   | Adds the dependency to `bundleDependencies`.                         |
| `--no-save`             | Installs the package without updating `package.json`.                |
| `-g`, `--global`        | Installs the package globally.                                       |
| `--legacy-peer-deps`    | Ignores peer dependency conflicts.                                   |
| `--strict-peer-deps`    | Treats peer dependency conflicts as errors.                          |
| `--package-lock-only`   | Updates `package-lock.json` without installing modules.              |
| `--omit <type>`         | Omits a type of dependency from install (`dev`, `optional`, `peer`). |
| `--ignore-scripts`      | Prevents running package lifecycle scripts.                          |
| `--dry-run`             | Simulates the install without making changes.                        |
| `--workspaces`          | Installs packages in the context of npm workspaces.                  |
| `--fund`                | Shows funding information for dependencies.                          |
| `npm ci`                | Installs dependencies exactly as per `package-lock.json`.            |
| `npm install <package>@<version>` | Installs a specific version of a package.                  |

---

## 2. Running Scripts (`npm run`)

| Command / Option   | Description                                                          |
| ------------------ | -------------------------------------------------------------------- |
| `npm run <script>` | Executes a script defined in `package.json`.                         |
| `--if-present`     | Runs the script only if it is defined.                               |
| `--silent`         | Suppresses output except for errors.                                 |
| `--`               | Passes arguments to the script. Example: `npm run build -- --watch`. |

---

## 3. Publishing & Versioning

| Command / Option | Description |
| ---------------- | ----------- |
| `npm publish`    | Publishes a package to the npm registry. |
| `npm version <update_type>` | Bumps the package version (`patch`, `minor`, `major`, or specific version). |

* Example: `npm version minor` increments `1.0.0` to `1.1.0`.
* Creates a git tag for the new version if in a git repository.
* patch — Bug fixes (1.0.0 → 1.0.1)
  * changes that do not affect the API  
* minor — New features, backward-compatible (1.0.0 → 1.1.0) 
  * adds new functionality in a backward-compatible manner
* major — Breaking changes (1.0.0 → 2.0.0)
  * changes that break backward compatibility

---

## 4. Updating & Auditing

| Command / Option       | Description                                                          |
| ---------------------- | -------------------------------------------------------------------- |
| `npm update`           | Updates installed packages to the latest compatible versions.        |
| `npm outdated`         | Shows which installed packages are outdated.                         |
| `npm audit`            | Performs a security audit of installed packages.                     |
| `npm audit fix`        | Attempts to automatically fix vulnerabilities.                       |
| `npm audit fix --force`| Applies potentially breaking changes to fix vulnerabilities.         |
| `npm doctor`           | Analyzes and suggests fixes for common issues in the project.        |

---

## 5. Uninstalling Packages (`npm uninstall`)

| Command / Option          | Description                                               |
| ------------------------- | --------------------------------------------------------- |
| `npm uninstall <package>` | Removes a package from `node_modules` and `package.json`. |
| `--save-dev`              | Removes the package from `devDependencies`.               |
| `--save-prod`             | Removes the package from `dependencies`.                  |
| `--save-optional`         | Removes the package from `optionalDependencies`.          |
| `-g`, `--global`          | Removes a globally installed package.                     |
| `--no-save`               | Uninstalls the package without updating `package.json`.   |
| `--dry-run`               | Simulates the uninstall without making changes.           |
| `npm prune`               | Removes extraneous packages not listed in `package.json`. |
| `npm cache clean --force` | Clears the npm cache.                                     |

* Example: `npm uninstall lodash --save-dev` removes `lodash` from `devDependencies`.
* Remember to check `package.json` and `package-lock.json` after uninstalling to ensure consistency.
* Use `npm cache clean --force` to clear the npm cache if you encounter issues during installs or uninstalls.

---

## 6. Managing the Lockfile (package-lock.json)

| Command / Option      | Description                                                                      |
| --------------------- | -------------------------------------------------------------------------------- |
| `npm ci`              | Installs dependencies exactly as specified in `package-lock.json`.               |
| `npm shrinkwrap`      | Creates a `npm-shrinkwrap.json` file to lock dependencies (similar to lockfile). |
| `--package-lock-only` | Updates only `package-lock.json` without installing packages.                    |

---

💡 **Tip:**

* Combine flags to customize installs or scripts:

```bash
npm install lodash --save-dev --legacy-peer-deps # Installs lodash as a dev dependency, ignoring peer deps
```

* Use `npm ci` in CI/CD pipelines to guarantee consistent installs across environments.
* Regularly run `npm audit` to keep your project secure.
* Use `npx` for one-off commands without global installs.
* Use `npm doctor` to identify and fix common project issues.

## more info
* [npm CLI Documentation](https://docs.npmjs.com/)
* [npx Documentation](https://www.npmjs.com/package/npx)



