import { Button } from "@/components/Button";
import { router } from "expo-router";
import { Image, Text, View } from "react-native";

export default function Index() {
  const handleGetStarted = () => {
    router.push("/camera");
  };

  const handleAbout = () => {
    router.push("/about");
  };

  const handleSettings = () => {
    router.push("/settings");
  };

  return (
    <View className="flex-1 bg-[#F2F2F2] justify-center items-center px-6 sm:px-8 md:px-12">
      <View className="w-full max-w-[640px] items-center">
        <Image
          source={require('../assets/images/icon.png')}
          resizeMode="contain"
          className="w-24 h-24 md:w-28 md:h-28 lg:w-32 lg:h-32 max-w-[128px] max-h-[128px]"
        />

        <Text className="mt-4 text-3xl md:text-4xl lg:text-5xl font-bold text-[#333333] text-center">Signapse</Text>
        <Text className="mt-2 text-base md:tesxt-lg text-[#666666] text-center leading-6">
          Gebaren naar tekst & spraak
        </Text>

        <View className="mt-6 w-full">
          <Button
            size="lg"
            className="w-full bg-black rounded-lg"
            label="Start detectie"
            labelClasses="text-white text-lg font-semibold"
            onPress={handleGetStarted}
          />
        </View>

        <View className="mt-3 w-full flex-col md:flex-row gap-3">
          <Button
            label="Over"
            className="w-full md:w-auto md:flex-1 bg-white px-10 border-2 rounded-lg border-[#B1B1B1]"
            labelClasses="text-black text-lg font-semibold"
            onPress={handleAbout}
            size="lg"
            variant="secondary"
          />
          <Button
            label="Instellingen"
            className="w-full md:w-auto md:flex-1 bg-white px-10 border-2 rounded-lg border-[#B1B1B1]"
            labelClasses="text-black text-lg font-semibold"
            onPress={handleSettings}
            size="lg"
            variant="secondary"
          />
        </View>
      </View>
    </View>
  );
};
