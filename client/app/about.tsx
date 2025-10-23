import { router } from "expo-router";
import { FlatList, Image, Linking, Pressable, Text, View } from "react-native";

export default function About() {
  const handleback = () => {
    router.push("/");
  };

  const renderItem = ({
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
  }) => (
    // Card: full width, consistent padding, centered content
    <View className="mb-3 w-full items-center rounded-xl bg-white p-4 shadow-sm">
      <Text className="mb-2 text-center text-lg font-semibold text-gray-800">{item.title}</Text>

      {/* members row: centered, wrapped */}
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

      {/* content: centered with consistent line-height */}
      {item.content ? (
        <Text className="mt-3 text-center text-sm leading-6 text-gray-600">{item.content}</Text>
      ) : null}

      {/* link button: only when link exists, consistent styling */}
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

  const aboutData: {
    key: string;
    title: string;
    content?: string;
    image?: string;
    link?: string;
    members?: { name: string; image?: string; link?: string }[];
  }[] = [
    {
      key: "about",
      title: "About the project",
      content: "SmartGlasses — Sign Language → Text",
    },
        {
      key: "idea",
      title: "Idea",
      content:
        "Capture sign language in real time using wearable/phone cameras, translate it to written text using a machine learning model, and present the result in a clean React Native UI for quick communication.",
    },
    {
      key: "goals",
      title: "Goals",
      content:
        "• Real-time recognition of a set of sign language gestures.\n• Translate recognized signs into natural text.\n• Provide an accessible React Native interface for display and interaction.\n• Support low-latency on-device inference or an efficient server API fallback.",
    },
    {
      key: "features",
      title: "Core features",
      content:
        "• Camera capture pipeline for video frames (or streamed landmarks).\n• ML inference: sign classification → text output (sentence builder).\n• UI: live camera view, recognized text overlay, history/log, language and settings.\n• Export/share recognized text (clipboard, messaging).\n• Basic user onboarding and privacy notice.",
    },
    {
      key: "roadmap",
      title: "Roadmap",
      content:
        "1. Prototype camera capture\n2. Integrate landmark extraction\n3. Train/test classifier\n4. Hook classifier to app\n5. Improve accuracy & smoothing\n6. Polish UI, add history & settings",
    },
    {
      key: "mkdocs",
      title: "MKDocs",
      content: "This documentation site contains all the info about the project",
      link: "https://github.com/vives-project-xp/SmartGlasses/tree/main/docs",
    },
    {
      key: "githubrepo",
      title: "GitHub Repository",
      content: "This is the main repository for the project.",
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
    <View className="flex-1 bg-gray-100 px-3">
      <FlatList
        data={aboutData}
        keyExtractor={(item) => item.key}
        renderItem={renderItem}
        className="w-full"
        contentContainerStyle={{ paddingBottom: 28, paddingTop: 10 }}
        showsVerticalScrollIndicator={false}
      />
    </View>
  );
}
