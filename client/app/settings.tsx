import { Picker } from "@react-native-picker/picker";
import React, { useEffect, useState } from "react";
import { FlatList, Image, Linking, Pressable, Text, View } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage"; // <-- Add this
import { BASE_URL } from "@/lib/const";
import api from "@/lib/api";
import { useTheme } from "@/lib/theme";
import { useAppSettings } from "@/lib/app-settings";

export default function Settings() {
  const [apiVersion, setApiVersion] = useState("-");
  const { preference, setPreference, colors } = useTheme();
  const { aiModel, setAiModel } = useAppSettings();

  useEffect(() => {
    // Fetch API version from the backend
    async function getAPIVersion() {
      const data = await api.health();
      setApiVersion(data.version);
    }
    getAPIVersion();
  }, []);

  // Item component with persistent storage
  function Item({
    item,
  }: {
    item: {
      key: string;
      title: string;
      content?: string;
      description?: string;
      image?: string;
      link?: string;
      options?: { label: string; value: string; description?: string }[];
      members?: { name: string; image?: string; link?: string }[];
    };
  }) {
    const storageKey = `setting_${item.key}`; // Unique key per setting
    const isThemeSetting = item.key === "THEME";
    const isAiSetting = item.key === "AI_VERSION";
    const [selected, setSelected] = useState(
      isThemeSetting ? preference : isAiSetting ? aiModel : item.options?.[0]?.value ?? ""
    );
    const selectedOption = item.options?.find((o) => o.value === selected);

    // Load saved value on mount
    useEffect(() => {
      if (!item.options || isThemeSetting || isAiSetting) {
        return;
      }
      const loadValue = async () => {
        try {
          const saved = await AsyncStorage.getItem(storageKey);
          if (saved !== null && item.options?.some((opt) => opt.value === saved)) {
            setSelected(saved);
          }
        } catch (error) {
          console.error("Failed to load setting:", error);
        }
      };
      if (item.options) {
        loadValue();
      }
    }, [storageKey, item.options, isThemeSetting, isAiSetting]);

    useEffect(() => {
      if (isThemeSetting && preference !== selected) {
        setSelected(preference);
      }
    }, [isThemeSetting, preference, selected]);

    useEffect(() => {
      if (isAiSetting && aiModel !== selected) {
        setSelected(aiModel);
      }
    }, [isAiSetting, aiModel, selected]);

    // Save value when changed
    const handleChange = async (value: string) => {
      setSelected(value);
      if (
        isThemeSetting &&
        (value === "light" || value === "dark" || value === "system")
      ) {
        setPreference(value as "light" | "dark" | "system");
        return;
      }
      if (
        isAiSetting &&
        (value === "VGT" || value === "ASL" || value === "LSTM")
      ) {
        setAiModel(value as "VGT" | "ASL" | "LSTM");
        return;
      }
      try {
        await AsyncStorage.setItem(storageKey, value);
      } catch (error) {
        console.error("Failed to save setting:", error);
      }
    };

    return (
      <View
        className="mb-3 w-full rounded-xl p-4 shadow-sm"
        style={{ backgroundColor: colors.card }}
      >
        <Text
          className="mb-2 text-center text-lg font-semibold"
          style={{ color: colors.text }}
        >
          {item.title}
        </Text>

        {/* Members */}
        {item.members ? (
          <View className="mt-3 flex-row flex-wrap justify-center">
            {item.members.map((m) => (
              <View key={m.name} className="mb-3 mr-3 w-24 ">
                {m.image ? (
                  <Image source={{ uri: m.image }} className="mb-2 h-16 w-16 rounded-full" />
                ) : (
                  <View
                    className="mb-2 h-16 w-16 rounded-full"
                    style={{ backgroundColor: colors.memberPlaceholder }}
                  />
                )}
                <Text className="text-xs" style={{ color: colors.textMuted }}>
                  {m.name}
                </Text>
              </View>
            ))}
          </View>
        ) : null}

        {/* Content */}
        {item.content ? (
          <Text
            className="mt-3 text-sm leading-6"
            style={{ color: colors.textMuted }}
          >
            {item.content}
          </Text>
        ) : null}

        {/* Picker with persistence */}
        {item.options ? (
          <View className="mt-4">
            <View
              className="rounded-md border"
              style={{
                borderColor: colors.border,
                backgroundColor: colors.pickerBackground,
              }}
            >
              <Picker
                selectedValue={selected}
                onValueChange={handleChange}
                mode="dropdown"
                //style={{ width: 160 }} // optioneel, bepaalt minimumbreedte
              >
                {item.options.map((opt) => (
                  <Picker.Item
                    key={opt.value}
                    label={opt.label}
                    value={opt.value}
                    color={colors.text}
                  />
                ))}
              </Picker>
            </View>
          </View>
        ) : null}

        {/* Show description of the selected option */}
        {selectedOption?.description ? (
          <Text
            className="background-white mt-2 text-sm"
            style={{ color: colors.textMuted }}
          >
            {selectedOption.description}
          </Text>
        ) : null}

        {/* Link button */}
        {item.link ? (
          <Pressable
            onPress={() => item.link && Linking.openURL(item.link)}
            accessibilityRole="link"
            className="mt-4 rounded-md px-4 py-2 shadow-sm"
            style={{ backgroundColor: colors.buttonBackground }}
          >
            <Text
              className="font-medium"
              style={{ color: colors.buttonText }}
            >
              Open link
            </Text>
          </Pressable>
        ) : null}
      </View>
    );
  }

  const SettingsData = [
    {
      key: "main",
      title: "App instellingen",
      content: "Configureer je voorkeuren hieronder.",
    },
    {
      key: "AI_VERSION",
      title: "AI Model",
      content: "Kies je voorkeurs AI-versie.",
      options: [
        {
          label: "Vlaamse gebarentaal - alfabet",
          value: "VGT",
          description: "dit gebruikt het VGT - model.\ndit werkt enkel met het alfabet.",
        },
        {
          label: "Amerikaanse gebarentaal - alfabet",
          value: "ASL",
          description: "dit gebruikt het ASL - model.\ndit werkt enkel met het alfabet.",
        },
        {
          label: "Vlaamse gebarentaal - woorden",
          value: "LSTM",
          description: "dit gebruikt het LSTM - model.\ndit werkt enkel met woorden.",
        },
      ],
    },
    {
      key: "THEME",
      title: "Thema",
      content: "Selecteer licht of donker modus.",
      options: [
        { label: "Licht", value: "light" },
        { label: "Donker", value: "dark" },
        { label: "Systeem", value: "system" },
      ],
    },
    {
      key: "apistuff",
      title: "API URL",
      content: `Server URL: ${BASE_URL}\nServer versie: v${apiVersion}`,
    },
  ];

  return (
    <View
      className="flex-1 items-center"
      style={{ backgroundColor: colors.background }}
    >
      {/* Centrale container met max breedte */}
      <View className="w-full max-w-[640px] flex-1">
        <FlatList
          data={SettingsData}
          keyExtractor={(item) => item.key}
          renderItem={({ item }) => <Item item={item} />}
          className="w-full"
          contentContainerStyle={{ paddingBottom: 28, paddingTop: 10 }}
          showsVerticalScrollIndicator={false}
        />
      </View>
    </View>
  );
}
