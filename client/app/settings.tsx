import { Picker } from "@react-native-picker/picker"; // npm i @react-native-picker/picker
import React, { useState } from "react";
import { FlatList, Image, Linking, Pressable, Text, View } from "react-native";

export default function Settings() {
  // Item component so each list item can have its own state (picker value)
  function Item({
    item,
  }: {
    item: {
      key: string;
      title: string;
      content?: string;
      image?: string;
      link?: string;
      options?: { label: string; value: string }[]; // select options
      members?: { name: string; image?: string; link?: string }[];
    };
  }) {
    const [selected, setSelected] = useState(item.options?.[0]?.value ?? "");

    return (
      <View className="mb-3 w-full items-center rounded-xl bg-white px-4 shadow-sm">
        <Text className="mb-2 text-center text-lg font-semibold text-gray-800">{item.title}</Text>

        {/* members row */}
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

        {/* content */}
        {item.content ? (
          <Text className="mt-3 text-center text-sm leading-6 text-gray-600">{item.content}</Text>
        ) : null}

        {/* Picker (select) — wrapped so we can style with Tailwind */}
        {item.options ? (
          <View className="mt-4 w-full rounded-md border border-gray-200 bg-white">
            {/* Picker itself doesn't accept className reliably, so style wrapper */}
            <Picker
              selectedValue={selected}
              onValueChange={(v) => setSelected(String(v))}
              mode="dropdown"
              className="mb-4 pb-4"
            >
              {item.options.map((opt) => (
                <Picker.Item key={opt.value} label={opt.label} value={opt.value} />
              ))}
            </Picker>
          </View>
        ) : null}

        {/* link button */}
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

  const SettingsData: {
    key: string;
    title: string;
    content?: string;
    image?: string;
    link?: string;
    options?: { label: string; value: string }[];
    members?: { name: string; image?: string; link?: string }[];
  }[] = [
    {
      key: "main",
      title: "page for the settings",
      content: "here you can change the settings of the app.",
    },
    {
      key: "OPTIONS1",
      title: "Settings",
      content: "content.",
      options: [
        { label: "optie 1", value: "optie 1" },
        { label: "optie 2", value: "optie 2" },
        { label: "optie 3", value: "optie 3" },
      ],
    },
    {
      key: "OPTIONS2",
      title: "Settings 1",
      content: "content.",
      options: [
        { label: "optie 1", value: "optie 1" },
        { label: "optie 2", value: "optie 2" },
        { label: "optie 3", value: "optie 3" },
      ],
    },
    {
      key: "OPTIONS3",
      title: "Settings 2",
      content: "content.",
      options: [
        { label: "optie 1", value: "optie 1" },
        { label: "optie 2", value: "optie 2" },
        { label: "optie 3", value: "optie 3" },
      ],
    },
    {
      key: "OPTIONS4",
      title: "Settings 3",
      content: "content.",
      options: [
        { label: "optie 1", value: "optie 1" },
        { label: "optie 2", value: "optie 2" },
        { label: "optie 3", value: "optie 3" },
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
        contentContainerStyle={{ paddingBottom: 28, paddingTop: 10}}
        showsVerticalScrollIndicator={false}
      />
    </View>
  );
}
