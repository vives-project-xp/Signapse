import { useTheme } from "@/lib/theme";
import { router } from "expo-router";
import { FlatList, Image, Linking, Pressable, Text, View } from "react-native";
import React from "react";

export default function About() {
  const { colors } = useTheme();
  const handleback = () => {
    router.push("/");
  };

  function Item({
    item,
  }: {
    item: {
      key: string;
      title: string;
      content?: string;
      image?: string;
      link?: string;
      members?: { name: string; image?: string; link?: string }[];
    };
  }) {
    return (
      <View
        className="mb-3 w-full rounded-xl p-4 shadow-sm"
        style={{ backgroundColor: colors.card }}
      >
        {/* Titel */}
        <Text
          className="mb-2 text-center text-lg font-semibold"
          style={{ color: colors.text }}
        >
          {item.title}
        </Text>

        {/* Teamleden */}
        {item.members ? (
          <View className="mt-3 flex-row flex-wrap justify-center">
            {item.members.map((m) => (
              <View key={m.name} className="mb-3 mr-3 w-24 items-center">
                {m.image ? (
                  <Image
                    source={{ uri: m.image }}
                    className="mb-2 h-16 w-16 rounded-full"
                  />
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

        {/* Tekst / inhoud */}
        {item.content ? (
          <Text
            className="mt-3 text-sm leading-6"
            style={{ color: colors.textMuted }}
          >
            {item.content}
          </Text>
        ) : null}

        {/* Link knop */}
        {item.link ? (
          <Pressable
            onPress={() => item.link && Linking.openURL(item.link)}
            accessibilityRole="link"
            className="mt-4 rounded-md px-4 py-2 shadow-sm"
            style={{ backgroundColor: colors.buttonBackground }}
          >
            <Text
              className="text-center font-medium"
              style={{ color: colors.buttonText }}
            >
              Open link
            </Text>
          </Pressable>
        ) : null}
      </View>
    );
  }

  const aboutData = [
    {
      key: "about",
      title: "Over het project",
      content: "Signapse helpt je om gebarentaal om te zetten naar tekst in real time.\nDe app herkent handbewegingen met de camera en toont meteen wat er bedoeld wordt — zo wordt communiceren makkelijker en sneller.",
    },
    {
      key: "idea",
      title: "Idee achter het project",
      content:
        "Het idee achter Singapse is om communicatie tussen dove en horende mensen te vergemakkelijken.\nDoor AI en camerabeelden te combineren, vertaalt de app gebaren automatisch naar tekst.",
    },
    {
      key: "goals",
      title: "Doelen",
      content:
        "Onze doelen zijn:\n• Gebaren in real time herkennen\n• Herkende gebaren omzetten naar natuurlijke tekst\n• Een eenvoudige en toegankelijke interface bieden\n• Zowel lokaal als via een server efficiënt werken.",
    },
    {
      key: "features",
      title: "Kernfuncties",
      content:
        "• Herkenning van gebaren via de camera\n• AI-model dat gebaren omzet naar tekst\n• Live weergave van herkende woorden en zinnen\n• Geschiedenis van herkende tekst\n• Delen of kopiëren van resultaten",
    },
    {
      key: "roadmap",
      title: "Roadmap",
      content:
        "1. Prototype camera opslag\n2. Integreer extractie van oriëntatiepunten\n3. Train/test classifier\n4. Koppel classifier aan app\n5. Verbeter nauwkeurigheid & smoothing\n6. Verfijn UI, voeg geschiedenis & instellingen toe",
    },
    {
      key: "mkdocs",
      title: "MKDocs",
      content:
        "Alle informatie over het project vind je op onze documentatiesite",
      link: "https://github.com/vives-project-xp/SmartGlasses/tree/main/docs",
    },
    {
      key: "githubrepo",
      title: "GitHub-repository",
      content: "Dit is de hoofdrepository voor het project.",
      link: "https://github.com/vives-project-xp/SmartGlasses",
    },
    {
      key: "team",
      title: "Project team",
      members: [
        { name: "Simon Stijnen", image: "https://github.com/SimonStnn.png" },
        { name: "Timo Plets", image: "https://github.com/TimoPlts.png" },
        { name: "Lynn Delaere", image: "https://github.com/lynndelaere.png" },
        { name: "Olivier Westerman", image: "https://github.com/olivierwesterman.png" },
        { name: "Kyell De Windt", image: "https://github.com/kyell182.png" },
      ],
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
          data={aboutData}
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
