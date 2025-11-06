import { Picker } from "@react-native-picker/picker";
import { router } from "expo-router";
import React, { useEffect, useState } from "react";
import { FlatList, Image, Linking, Pressable, Text, View } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage"; // <-- Add this

export default function Settings() {
  const handleback = () => {
    router.push("/");
  };

  // Item component with persistent storage
  function Item({
    item,
  }: {
    item: {
      key: string;
      title: string;
      content?: string;
      image?: string;
      link?: string;
      options?: { label: string; value: string }[];
      members?: { name: string; image?: string; link?: string }[];
    };
  }) {
    const storageKey = `setting_${item.key}`; // Unique key per setting
    const [selected, setSelected] = useState(item.options?.[0]?.value ?? "");

    // Load saved value on mount
    useEffect(() => {
      const loadValue = async () => {
        try {
          const saved = await AsyncStorage.getItem(storageKey);
          if (saved !== null && item.options?.some(opt => opt.value === saved)) {
            setSelected(saved);
          }
        } catch (error) {
          console.error("Failed to load setting:", error);
        }
      };
      if (item.options) {
        loadValue();
      }
    }, [storageKey, item.options]);

    // Save value when changed
    const handleChange = async (value: string) => {
      setSelected(value);
      try {
        await AsyncStorage.setItem(storageKey, value);
      } catch (error) {
        console.error("Failed to save setting:", error);
      }
    };

    return (
      <View className="mb-3 w-full items-center rounded-xl bg-white p-4 shadow-sm">
        <Text className="mb-2 text-center text-lg font-semibold text-gray-800">
          {item.title}
        </Text>

        {/* Members */}
        {item.members ? (
          <View className="mt-3 flex-row flex-wrap justify-center">
            {item.members.map((m) => (
              <View key={m.name} className="mb-3 mr-3 w-24 items-center">
                {m.image ? (
                  <Image source={{ uri: m.image }} className="mb-2 h-16 w-16 rounded-full" />
                ) : (
                  <View className="mb-2 h-16 w-16 rounded-full bg-gray-200" />
                )}
                <Text className="text-center text-xs text-gray-700">{m.name}</Text>
              </View>
            ))}
          </View>
        ) : null}

        {/* Content */}
        {item.content ? (
          <Text className="mt-3 text-center text-sm leading-6 text-gray-600">
            {item.content}
          </Text>
        ) : null}

        {/* Picker with persistence */}
        {item.options ? (
          <View className="mt-4 w-full rounded-md border border-gray-200 bg-white">
            <Picker
              selectedValue={selected}
              onValueChange={handleChange}
              mode="dropdown"
            >
              {item.options.map((opt) => (
                <Picker.Item key={opt.value} label={opt.label} value={opt.value} />
              ))}
            </Picker>
          </View>
        ) : null}

        {/* Link button */}
        {item.link ? (
          <Pressable
            onPress={() => item.link && Linking.openURL(item.link)}
            accessibilityRole="link"
            className="mt-4 rounded-md bg-blue-600 px-4 py-2 shadow-sm"
          >
            <Text className="text-center font-medium text-white">Open link</Text>
          </Pressable>
        ) : null}
      </View>
    );
  }

  const SettingsData = [
    {
      key: "main",
      title: "App Settings",
      content: "Configure your preferences below.",
    },
    {
      key: "AI_VERSION",
      title: "AI Model",
      content: "Choose your preferred AI version.",
      options: [
        { label: "Grok 3 (Free)", value: "grok3" },
        { label: "Grok 4 (Premium)", value: "grok4" },
        { label: "Local Model", value: "local" },
      ],
    },
    {
      key: "THEME",
      title: "Theme",
      content: "Select light or dark mode.",
      options: [
        { label: "Light", value: "light" },
        { label: "Dark", value: "dark" },
        { label: "System", value: "system" },
      ],
    },
    
  ];

  return (
    <View className="flex-1 bg-gray-100 px-3">
      <FlatList
        data={SettingsData}
        keyExtractor={(item) => item.key}
        renderItem={({ item }) => <Item item={item} />}
        className="w-full"
        contentContainerStyle={{ paddingBottom: 28, paddingTop: 10 }}
        showsVerticalScrollIndicator={false}
      />
    </View>
  );
}
