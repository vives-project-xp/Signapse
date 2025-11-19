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
      <Stack.Screen name="index" options={{ title: "Welcome", headerShown: false  } } />
      <Stack.Screen name="camera" options={{ title: "Camera View", headerShown: false  }} />
      <Stack.Screen name="about" options={{ title: "About", headerShown: true  }} />
      <Stack.Screen name="settings" options={{ title: "Settings", headerShown: true  }} />
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
