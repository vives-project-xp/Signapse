import Constants from "expo-constants";
import { Platform } from "react-native";

// For mobile development, you need to use your computer's IP address
// Run `ipconfig` (Windows) or `ifconfig` (Mac/Linux) to find your local IP
// Example: "http://192.168.1.100:8000/"
const DEFAULT_URL = Platform.OS === "web" ? "http://127.0.0.1:8000/" : "http://127.0.0.1:8000/"; // Change this to your computer's IP for mobile testing

export const BASE_URL =
  (Constants?.expoConfig?.extra as any)?.EXPO_PUBLIC_API_URL ||
  process.env.EXPO_PUBLIC_API_URL ||
  DEFAULT_URL;

// Log the URL being used for debugging
if (__DEV__) {
  console.log(`[API] Using BASE_URL: ${BASE_URL}`);
  if (Platform.OS !== "web" && BASE_URL.includes("127.0.0.1")) {
    console.warn(
      "[API] WARNING: Using 127.0.0.1 on mobile device. " +
        "This will not work! Set EXPO_PUBLIC_API_URL to your computer's local IP address."
    );
  }
}
