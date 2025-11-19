import { Button } from "@/components/Button";
import { useTheme } from "@/lib/theme";
import { router } from "expo-router";
import { Image, Text, View } from "react-native";

export default function Index() {
  const { colors } = useTheme();

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
    <View
      className="flex-1 justify-center items-center px-6 sm:px-8 md:px-12"
      style={{ backgroundColor: colors.background }}
    >
      <View className="w-full max-w-[640px] items-center">
        <Image
          source={require('../assets/images/Handsymbol.png')}
          resizeMode="contain"
          className="w-24 h-24 md:w-28 md:h-28 lg:w-32 lg:h-32 max-w-[128px] max-h-[128px]" 
        />

        <Text
          className="mt-4 text-3xl md:text-4xl lg:text-5xl font-bold text-center"
          style={{ color: colors.text }}
        >
          Smartglasses
        </Text>
        <Text
          className="mt-2 text-base md:tesxt-lg text-center leading-6"
          style={{ color: colors.textMuted }}
        >
          Gestures to text & speech
        </Text>

        <View className="mt-6 w-full">
          <Button
            size="lg"
            className="w-full rounded-lg"
            style={{ backgroundColor: colors.buttonBackground }}
            label="Start detection"
            labelClasses="text-lg font-semibold"
            labelStyle={{ color: colors.buttonText }}
            onPress={handleGetStarted}
          />
        </View>

        <View className="mt-3 w-full flex-col md:flex-row gap-3">
          <Button
            label="About"
            className="w-full md:w-auto md:flex-1 rounded-lg px-10 border-2"
            style={{
              backgroundColor: colors.card,
              borderColor: colors.border,
            }}
            labelClasses="text-lg font-semibold"
            labelStyle={{ color: colors.text }}
            onPress={handleAbout}
            size="lg"
            variant="secondary"
          />
          <Button
            label="Settings"
            className="w-full md:w-auto md:flex-1 rounded-lg px-10 border-2"
            style={{
              backgroundColor: colors.card,
              borderColor: colors.border,
            }}
            labelClasses="text-lg font-semibold"
            labelStyle={{ color: colors.text }}
            onPress={handleSettings}
            size="lg"
            variant="secondary"
          />
        </View>
      </View>
    </View>
  );
};
