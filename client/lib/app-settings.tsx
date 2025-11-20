import AsyncStorage from "@react-native-async-storage/async-storage";
import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type AiModelSetting = "VGT" | "ASL" | "LSTM";
type AlphabetModel = "asl" | "vgt";

type SettingsContextType = {
  aiModel: AiModelSetting;
  setAiModel: (model: AiModelSetting) => void;
  alphabetModel: AlphabetModel | null;
  showLandmarksButton: boolean;
  setShowLandmarksButton: (value: boolean) => void;
};

const DEFAULT_AI_MODEL: AiModelSetting = "VGT";
const AI_MODEL_STORAGE_KEY = "setting_AI_VERSION";
const SHOW_LANDMARKS_STORAGE_KEY = "setting_SHOW_LANDMARKS_BUTTON";

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export function AppSettingsProvider({ children }: { children: ReactNode }) {
  const [aiModel, setAiModelState] = useState<AiModelSetting>(DEFAULT_AI_MODEL);
  const [showLandmarksButton, setShowLandmarksButtonState] = useState<boolean>(true);

  useEffect(() => {
    (async () => {
      try {
        const stored = await AsyncStorage.getItem(AI_MODEL_STORAGE_KEY);
        if (stored === "VGT" || stored === "ASL" || stored === "LSTM") {
          setAiModelState(stored);
        }
        const storedToggle = await AsyncStorage.getItem(SHOW_LANDMARKS_STORAGE_KEY);
        if (storedToggle === "true" || storedToggle === "false") {
          setShowLandmarksButtonState(storedToggle === "true");
        }
      } catch (error) {
        console.error("Failed to restore AI model preference:", error);
      }
    })();
  }, []);

  const persistPreference = useCallback(async (value: AiModelSetting) => {
    try {
      await AsyncStorage.setItem(AI_MODEL_STORAGE_KEY, value);
    } catch (error) {
      console.error("Failed to persist AI model preference:", error);
    }
  }, []);

  const setAiModel = useCallback(
    (value: AiModelSetting) => {
      setAiModelState(value);
      void persistPreference(value);
    },
    [persistPreference]
  );

  const setShowLandmarksButton = useCallback(async (value: boolean) => {
    setShowLandmarksButtonState(value);
    try {
      await AsyncStorage.setItem(SHOW_LANDMARKS_STORAGE_KEY, value ? "true" : "false");
    } catch (error) {
      console.error("Failed to persist landmarks toggle:", error);
    }
  }, []);

  const alphabetModel = useMemo<AlphabetModel | null>(() => {
    switch (aiModel) {
      case "ASL":
        return "asl";
      case "VGT":
        return "vgt";
      default:
        return null;
    }
  }, [aiModel]);

  const value = useMemo(
    () => ({
      aiModel,
      setAiModel,
      alphabetModel,
      showLandmarksButton,
      setShowLandmarksButton,
    }),
    [aiModel, setAiModel, alphabetModel, showLandmarksButton, setShowLandmarksButton]
  );

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useAppSettings() {
  const context = useContext(SettingsContext);
  if (!context) {
    throw new Error("useAppSettings must be used within an AppSettingsProvider");
  }
  return context;
}
