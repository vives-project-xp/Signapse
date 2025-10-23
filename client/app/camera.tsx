import { Button } from "@/components/Button";
import { CameraType, CameraView, useCameraPermissions } from "expo-camera";
import { router } from "expo-router";
import { useEffect, useState } from "react";
import { Text, View} from "react-native";

import { SafeAreaView } from "react-native-safe-area-context";

export default function CameraScreen() {
  const [facing, setFacing] = useState<CameraType>("back");
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [showTranslation, setShowTranslation] = useState(false);  // voor vertalingd

  // Request permission automatically on mount so camera opens when entering screen
  useEffect(() => {
    if (!cameraPermission?.granted) {
      // calling requestPermission will prompt the user on first run
      requestCameraPermission();
    }
  }, [cameraPermission, requestCameraPermission]);

  if (!cameraPermission) {
    // Camera permissions are still loading.
    return <Text>Loading camera...</Text>
  }

  if (!cameraPermission.granted) {
    // Camera permissions are not granted yet.
    return (
      <View className="mb-3 aspect-square w-full rounded-2xl border border-[#B1B1B1] bg-black">

      </View>
    );
  }

  function toggleCameraFacing() {
    setFacing((current) => (current === "back" ? "front" : "back"));
  }

  // Always attempt to show the camera (permission is requested automatically)
  return (

      <View className="flex-1 bg-[#F2F2F2]">
        <CameraView facing={facing} style={{flex: 1}} />

        {/* Overlay controls */}
        <SafeAreaView pointerEvents="box-none"
        className="absolute inset-x-0 bottom-4 sm:bottom-6 md:bottom-8 px-4 sm:px-6 md:px-8" >

          <View className="w-full px-4 sm:px-6 md:px-8">

            {showTranslation && (
                <View className="mb-3 bg-white rounded-xl border border-[#B1B1B1] items-center justify-center w-full max-w-2xl self-center h-56 md:h-64 lg:h-72 px-4">
                    <Text className="text-black text-xl md:text-2xl font-semibold text-center">Vertaling</Text>
                </View>
            )}

            <View className="w-full max-w-2xl self-center">
              <View className="flex-row justify-between items-center gap-2 md:gap-3">
                <Button
                  label="Back"
                  className="flex-1 h-12 sm:h-14 md:h-16 bg-white border-2 rounded-lg border-[#B1B1B1]"
                  labelClasses="text-black text-base sm:text-lg md:text-xl font-semibold"
                  onPress={() => router.back()}
                  size="lg"
                  variant="secondary"
                />
                <Button
                  label="Text"
                  className="flex-1 h-12 sm:h-14 md:h-16 bg-white border-2 rounded-lg border-[#B1B1B1]"
                  labelClasses="text-black text-base sm:text-lg md:text-xl font-semibold"
                  onPress={() => setShowTranslation((v) => !v)}
                  size="lg"
                  variant="secondary"
                />
                <Button
                  label="Flip"
                  className="flex-1 h-12 sm:h-14 md:h-16 bg-white border-2 rounded-lg border-[#B1B1B1]"
                  labelClasses="text-black text-base sm:text-lg md:text-xl font-semibold"
                  onPress={toggleCameraFacing}
                  size="lg"
                  variant="secondary"
                />
              </View>
            </View>
          </View>
        </SafeAreaView>
      </View>
  );
}
