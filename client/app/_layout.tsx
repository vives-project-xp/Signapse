import "../assets/styles/globals.css";

import { Stack } from "expo-router";

export default function RootLayout() {
  return (
    <Stack
    screenOptions={{
    headerShown: false,
    contentStyle: { backgroundColor: "transparent" },
    }}
    >
      <Stack.Screen name="index" options={{ title: "Welcome", headerShown: false  } } />
      <Stack.Screen name="camera" options={{ title: "Camera View", headerShown: false  }} />
      <Stack.Screen name="about" options={{ title: "About", headerShown: true  }} />
      <Stack.Screen name="settings" options={{ title: "Settings", headerShown: true  }} />
    </Stack>
  );
}
