import React, {
  useEffect,
  useState,
  createContext,
  useContext,
  useMemo,
  type ReactNode,
} from "react";
import { Appearance } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useColorScheme as useNativeWindColorScheme } from "nativewind";

type ThemePreference = "light" | "dark" | "system";

type ThemeColors = {
  background: string;
  surface: string;
  card: string;
  text: string;
  textMuted: string;
  border: string;
  pickerBackground: string;
  memberPlaceholder: string;
  buttonBackground: string;
  buttonText: string;
  headerBackground: string;
  headerText: string;
};

type ThemeContextType = {
  preference: ThemePreference;
  colorScheme: "light" | "dark";
  setPreference: (pref: ThemePreference) => void;
  colors: ThemeColors;
};

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const THEME_PREF_KEY = "settings_THEME";

const palettes: Record<"light" | "dark", ThemeColors> = {
  light: {
    background: "#F2F2F2",
    surface: "#ffffff",
    card: "#ffffff",
    text: "#1f2937",
    textMuted: "#4b5563",
    border: "#e5e7eb",
    pickerBackground: "#ffffff",
    memberPlaceholder: "#e5e7eb",
    buttonBackground: "#2563eb",
    buttonText: "#ffffff",
    headerBackground: "#ffffff",
    headerText: "#111111",
  },
  dark: {
    background: "#181818",
    surface: "#e80707ff",
    card: "#1f1f1f",
    text: "#f3f4f6",
    textMuted: "#f3f4f6",
    border: "#2f2f2f",
    pickerBackground: "#181818",
    memberPlaceholder: "#1f1f1f",
    buttonBackground: "#3b82f6",
    buttonText: "#ffffff",
    headerBackground: "#181818",
    headerText: "#f3f4f6",
  },
};

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [preference, setPreference] = useState<ThemePreference>("system");
  const [systemScheme, setSystemScheme] = useState<"light" | "dark">(
    Appearance.getColorScheme() ?? "light"
  );
  const {
    colorScheme: nativeWindScheme,
    setColorScheme: setNativeWindScheme,
  } = useNativeWindColorScheme();

  useEffect(() => {
    (async () => {
      const saved = await AsyncStorage.getItem(THEME_PREF_KEY);
      if (saved === "light" || saved === "dark" || saved === "system") {
        setPreference(saved);
      }
    })();
  }, []);

  useEffect(() => {
    const sub = Appearance.addChangeListener(({ colorScheme }) => {
      if (colorScheme === "light" || colorScheme === "dark") {
        setSystemScheme(colorScheme);
      }
    });
    return () => sub.remove();
  }, []);

  useEffect(() => {
    AsyncStorage.setItem(THEME_PREF_KEY, preference);
  }, [preference]);

  const colorScheme: "light" | "dark" =
    preference === "system" ? systemScheme : preference;

  useEffect(() => {
    if (nativeWindScheme === colorScheme) {
      return;
    }
    setNativeWindScheme(colorScheme);
  }, [colorScheme, nativeWindScheme, setNativeWindScheme]);

  const colors = useMemo(
    () => palettes[colorScheme],
    [colorScheme]
  );

  return (
    <ThemeContext.Provider
      value={{ preference, colorScheme, setPreference, colors }}
    >
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}
