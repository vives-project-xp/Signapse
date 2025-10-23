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
    <View className="flex-1 bg-[#F2F2F2] justify-center items-center p-5">
      <View className="items-center max-w-[300px]">
        <Image
          source={require('../assets/images/Handsymbol.png')}
          style={{ width: 100, height: 100 }}
        />

        <Text className="text-4xl font-bold text-[#333333] text-center mb-5">Smartglasses</Text>
        <Text className="text-base text-[#666666] text-center leading-6 mb-10">
          Gestures to text & speech
        </Text>

        <View className="self-stretch">
          <Button
            size="lg"
            className="bg-black py-4 rounded-lg w-full mb-5"
            label="Start detection"
            labelClasses="text-white text-lg font-semibold"
            onPress={handleGetStarted}
          />
        </View>

        <View className="flex-row self-stretch">
          <Button
            label="About"
            className="bg-white px-10 py-4 border-2 rounded-lg border-[#B1B1B1] mr-3 flex-1"
            labelClasses="text-black text-lg font-semibold"
            onPress={handleAbout}
            size="default"
            variant="secondary"
          />
          <Button
            label="Settings"
            className="bg-white px-10 py-4 border-2 rounded-lg border-[#B1B1B1] flex-1"
            labelClasses="text-black text-lg font-semibold"
            onPress={handleSettings}
            size="default"
            variant="secondary"
          />
        </View>
      </View>
    </View>
  );
}
