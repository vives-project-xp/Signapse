import { SafeAreaView } from "react-native-safe-area-context";
import "../assets/styles/globals.css";

import { Stack } from "expo-router";

export default function RootLayout() {
  return (
    <Stack>
      <Stack.Screen
        name="index"
        options={{
          title: "Welcome",
          headerShown: false,
        }}
      />
      <Stack.Screen
        name="camera"
        options={{
          title: "Camera View",
          headerShown: false,
        }}
      />
      <Stack.Screen
        name="about"
        options={{
          title: "About",
          headerShown: false,
        }}
      />
    </Stack>
  );
}
