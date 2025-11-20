import "../assets/styles/globals.css";
import { ThemeProvider, useTheme } from "@/lib/theme";
import { AppSettingsProvider } from "@/lib/app-settings";

import { Stack } from "expo-router";

function RootStack() {
  const { colorScheme, colors } = useTheme();
  const isDark = colorScheme === "dark";

  return (
    <Stack
      screenOptions={{
        headerShown: false,
        contentStyle: {
          backgroundColor: colors.background,
        },
        statusBarStyle: isDark ? "light" : "dark",
        headerStyle: { backgroundColor: colors.headerBackground },
        headerTintColor: colors.headerText,
      }}
    >
      <Stack.Screen name="index" options={{ title: "Welkom", headerShown: false  } } />
      <Stack.Screen name="camera" options={{ title: "Cameraweergave", headerShown: false  }} />
      <Stack.Screen name="about" options={{ title: "Over", headerShown: true  }} />
      <Stack.Screen name="settings" options={{ title: "Instellingen", headerShown: true  }} />
    </Stack>
  );
}

export default function RootLayout() {
  return (
    <AppSettingsProvider>
      <ThemeProvider>
        <RootStack />
      </ThemeProvider>
    </AppSettingsProvider>
  );
}
