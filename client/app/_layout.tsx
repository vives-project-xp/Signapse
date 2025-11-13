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
      <Stack.Screen name="index" options={{ title: "Welkom", headerShown: false  } } />
      <Stack.Screen name="camera" options={{ title: "Cameraweergave", headerShown: false  }} />
      <Stack.Screen name="about" options={{ title: "Over", headerShown: true  }} />
      <Stack.Screen name="settings" options={{ title: "Instellingen", headerShown: true  }} />
    </Stack>
  );
}
