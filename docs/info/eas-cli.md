!!! warning
	This file serves as a reference for understanding EAS-CLI. For installation or setup, see the [Getting Started](../Getting%20Started/README.md) folder of your Expo project.

# EAS-CLI — Info
(Expo Application Services Command Line Interface)




---

## What is EAS-CLI?

**EAS (Expo Application Services)** is `Expo’s` cloud-based infrastructure for building, signing, submitting, and updating **React Native applications.**
It replaces traditional local native build pipelines by offering a consistent, automated, and secure workflow.

`EAS` consists of several core services:

*   `EAS Build` — Cloud-based iOS & Android builds.

*   `EAS Submit` — Automated submission to app stores.

*   `EAS Update` — Over-the-air updates without rebuilding binaries.

*   `EAS Device` — Register physical iOS devices for internal distribution.

The **EAS CLI** is the local command-line tool used to interact with these services.

---

## Why use EAS-CLI?

-   Build native binaries without Xcode or Android Studio.
    
-   Build apps in the cloud or locally.

-   Keep signing credentials synced and automatically managed.

-   Enable app updates without publishing a new version (OTA).

-   Enforce consistent build pipelines across development teams.

-   Integrate seamlessly into CI/CD environments.

-   Support internal distribution for testing.

EAS modernizes the entire React Native production workflow.

---

## Project Structure

EAS-CLI uses a configuration file named `eas.json` located at the root of your project directory. This file defines build profiles, submission settings, and update configurations.

Here is an example `eas.json` file:

```json
{
  "cli": {
    "version": ">= 3.0.0"
  },
  "build": {
    "development": {
      "distribution": "internal",
      "android": {
        "buildType": "apk"
      },
      "ios": {
        "simulator": true
      }
    },
    "production": {
      "distribution": "store",
      "android": {
        "buildType": "aab"
      },
      "ios": {
        "image": "latest"
      }
    }
  },
  "submit": {
    "production": {}
  }
}
```
The `eas.json` file defines `profiles`. Each profile describes how a build should behave.

---

## EAS Build

EAS Build allows you to create native builds of your app in the cloud or on your local machine. You can choose different build profiles defined in your `eas.json` file.

Common build profiles include:
* `preview`: For quick testing with minimal optimizations.
* `development`: For testing with debug options enabled.
* `production`: For app store distribution with optimizations.
* `custom`: User-defined profiles for specific build configurations.
* `internal`: For internal distribution builds.
* `simulator`: For building iOS simulator binaries.

running a build is as simple as executing the following command in your project directory:

```bash
eas build
```
You can specify the platform and profile:

```bash
eas build --platform ios --profile production
```
This command will start a cloud build for iOS using the `production` profile defined in your `eas.json` file.

Or for Android:

```bash
eas build --platform android --profile development
```
This command will start a cloud build for Android using the `development` profile.

**Local Vs Cloud Builds**
-  Cloud builds are executed on Expo's servers, requiring no local setup.
-  Cloud builds are easier to set up and maintain, especially for teams without native development expertise.

-  Local builds run on your machine, requiring native toolchains (Xcode, Android Studio).
-  Local builds offer more control but need proper environment configuration.

-  Choose the build method that best fits your project's needs.

**example Build Commands**

local: `--local` flag

```bash
eas build --platform ios --local --profile development
```
❗when there is no `--local` flag, the build will be a cloud build by default.

---
## Credentials Management

eas handels credentials automatically:

```bash
eas credentials
```
This command allows you to view and manage your app signing credentials for both iOS and Android.

you can :
*  generate certificates
*  upload your own provisioning profiles
*  let eas manage everything for you

eas stores your credentials securely in the cloud, making it easy to manage them across different machines and team members.

---
## EAS Submit

EAS Submit automates the process of submitting your app to the Apple App Store and Google Play Store.

After building , you can submit your app using:

**for iOS:**

```bash
eas submit --platform ios
```
**for Android:**

```bash
eas submit --platform android
```
**use a specific profile:**

```bash
eas submit --platform ios --profile production
eas submit --platform android --profile production
```
*   submissions can also pull the correct binary from the eas server if you don't have it locally.

*   EAS Submit handles all the necessary steps, including authentication, metadata management, and binary upload.

---
## EAS Update (OTA Updates)
EAS Update allows you to push **O**ver-**T**he-**A**ir updates to your app without going through the app store review process.

**To publish an update, use the following command:**

```bash
eas update
```

**target a branch:**

```bash
eas update --branch production
```
This command will bundle your JavaScript and assets and upload them to the EAS Update servers.

or define a specific message for the update:

```bash
eas update --message "Fixed critical bug in login flow"
```
this updates all users instantly (within your rollout rules).

To integrate EAS Update into your app, you need to install the `expo-updates` package and configure it in your app's code:

```javascript
import * as Updates from 'expo-updates';
async function checkForUpdates() {
  try {
    const update = await Updates.checkForUpdateAsync();
    if (update.isAvailable) {
      await Updates.fetchUpdateAsync();
      // Notify user about the update and reload the app
      await Updates.reloadAsync();
    }
  } catch (e) {
    // Handle or log error
    console.error(e);
  }
}
```
rollout strategies:
*  `Immediate`: Apply the update as soon as it's downloaded.
*  `On Next Restart`: Apply the update the next time the app is restarted.
*  `Custom Logic`: Implement your own logic to determine when to apply the update.
*  `Rollout Percentage`: Gradually roll out the update to a percentage of users.
*  `Staged Rollout`: Release the update to specific user segments or regions first.
*  `Scheduled Rollout`: Plan updates to be released at specific times.
*  `Manual Approval`: Require manual approval before applying updates.

you can do all this configuration in the `eas.json` file with the `workflow` option:

```json
{
  "cli": {
    "version": ">= 3.0.0"
  },
  "build": {
    "production": {
      "ios": {
        "workflow": "managed"
      },
      "android": {
        "workflow": "managed"
      }
    },
    "development": {
      "ios": {
        "workflow": "generic"
      },
      "android": {
        "workflow": "generic"
      }
    }
  }
}
```
The `workflow` option specifies whether to use the `managed` or `generic` workflow for each platform in the specified build profile.

---
## eas Device.

**register physical iOS devices for internal distribution and testing.**

```bash
eas device:create
```
This command allows you to register a new iOS device with your Apple Developer account, enabling you to distribute builds directly to that device for testing purposes.

**fetch and manage registered devices:**

```bash
eas device:list
```
This command lists all the devices registered under your Apple Developer account, allowing you to manage and organize them effectively.

**Build with registered devices:**

Use any EAS build profile that uses distribution: internal or a development client:

```bash
eas build --platform ios --profile development # This  will create a build that can be installed on the registered devices for testing.
```

❗Expo will automatically handle the provisioning profiles to include the registered devices.

---

## Useful eas-cli Commands - reference

**build**

| Command                            | Description                          |
| ---------------------------------- | ------------------------------------ |
| `eas build`                        | Starts a build using default profile |
| `eas build --profile <name>`       | Uses a specific build profile        |
| `eas build --local`                | Builds locally                       |
| `eas build --platform ios/android` | Specify platform                     |
| `eas build:list`                   | Shows previous builds                |
| `eas build:cancel`                 | Cancels an active build              |
| `eas build:configure`              | Sets up eas.json file                |
| `eas build:inspect`                | Inspects a specific build            |
| `eas build:status`                 | Checks the status of the latest build|
| `eas build:logs`                   | Fetches logs for a specific build    |
| `eas build:artifacts`              | Downloads build artifacts            |
| `eas build:purge`                  | Deletes old builds                   |

**submit**

| Command                               | Description                |
| ------------------------------------- | -------------------------- |
| `eas submit`                          | Start submission flow      |
| `eas submit --platform <ios/android>` | Submit a specific platform |
| `eas submit:list`                     | View previous submissions  |
| `eas submit:inspect`                  | Inspect a specific submit  |
| `eas submit:status`                   | Check status of last submit|
| `eas submit:credentials`              | Manage submission creds    |

**Updates**

| Command                      | Description                  |
| ---------------------------- | ---------------------------- |
| `eas update`                 | Publish an OTA update        |
| `eas update --branch <name>` | Target a branch              |
| `eas update:list`            | List update history          |
| `eas update:rollback`        | Roll back to previous update |
| `eas update:inspect`         | Inspect a specific update    |
| `eas update:status`          | Check status of last update  |

**credentials**

| Command                         | Description                     |
| ------------------------------- | ------------------------------- |
| `eas credentials`                | Manage app signing credentials  |
| `eas credentials:manager`        | Interactive credential manager  |
| `eas credentials:clear`          | Clear stored credentials        |
| `eas credentials:sync`           | Sync credentials with cloud     |
| `eas credentials:download`       | Download credentials            |
| `eas credentials:upload`         | Upload local credentials        |

**devices**

| Command                     | Description                        |
| --------------------------- | ---------------------------------- |
| `eas device:create`         | Register a new iOS device          |
| `eas device:list`           | List registered iOS devices        |
| `eas device:delete`         | Delete a registered iOS device     |
| `eas device:download`       | Download registered devices as CSV |

---

## Example: Building an Android App with EAS-CLI
1. Open the terminal in your project folder.
2. Start the build: 

```bash
npx eas build --platform android --profile production
``` 
3. Follow the prompts:
   * Google account credentials
   * Choose a keystore (EAS can handle this automatically)
4. EAS will start the cloud build and provide a link to monitor the status. 
5. Once complete, download the `.aab` or `.apk` file for distribution.
6. Submit the app to the Google Play Store using EAS Submit:

```bash
eas submit --platform android --profile production
```
7. Follow the prompts to complete the submission process.
8. Monitor the submission status until your app is live on the Play Store.
9.  Celebrate your successful Android app deployment! 🎉

---

## Example: Building an iOS App with EAS-CLI
1. Open the terminal in your project folder.
2. Start the build: 

```bash
npx eas build --platform ios --profile production
```
3. Follow the prompts:
   * Apple ID and password
   * Two-factor authentication code
   * Choose a provisioning profile (EAS can handle this automatically)
4. EAS will start the cloud build and provide a link to monitor the status.
5. Once complete, download the `.ipa` file for distribution.
6. Submit the app to the App Store using EAS Submit:
```bash
eas submit --platform ios --profile production
```
7. Follow the prompts to complete the submission process.
8. Monitor the submission status until your app is live on the App Store.
9. Celebrate your successful iOS app deployment! 🎉

## best Practices

* Use seperate build profiles for development, staging, and production.
* Commit your `eas.json` - it keeps build configurations consistent across your team.
* Enable automatic credentials management for easier setup.
* Use EAS Update for quick bug fixes and feature rollouts.
* Use EAS Build for all production builds to ensure consistency.

⚠️ Store API keys and sensitive info securely using environment variables. Never hardcode them in your app.

---

## additional Resources
* [EAS-CLI Documentation](https://github.com/expo/eas-cli)
* [EAS Build Documentation](https://docs.expo.dev/build/eas-json)
* [EAS Tutorial (By the makers of expo)](https://docs.expo.dev/tutorial/eas/introduction/)




