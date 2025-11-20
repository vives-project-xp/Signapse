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
};

const DEFAULT_AI_MODEL: AiModelSetting = "VGT";
const AI_MODEL_STORAGE_KEY = "setting_AI_VERSION";

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export function AppSettingsProvider({ children }: { children: ReactNode }) {
  const [aiModel, setAiModelState] = useState<AiModelSetting>(DEFAULT_AI_MODEL);

  useEffect(() => {
    (async () => {
      try {
        const stored = await AsyncStorage.getItem(AI_MODEL_STORAGE_KEY);
        if (stored === "VGT" || stored === "ASL" || stored === "LSTM") {
          setAiModelState(stored);
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
    }),
    [aiModel, setAiModel, alphabetModel]
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
