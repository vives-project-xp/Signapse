import { Button } from "@/components/Button";
import { CameraType, CameraView, useCameraPermissions } from "expo-camera";
import { router } from "expo-router";
import { useEffect, useState } from "react";
import { Text, TouchableOpacity, View } from "react-native";

import { SafeAreaProvider,  SafeAreaView} from "react-native-safe-area-context";

export default function CameraScreen() {
  const [facing, setFacing] = useState<CameraType>("back");
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [showTranslation, setShowTranslation] = useState(false);  // voor vertalingd

  // Request permission automatically on mount so camera opens when entering screen
  useEffect(() => {
    if (!cameraPermission || !cameraPermission.granted) {
      // calling requestPermission will prompt the user on first run
      requestCameraPermission();
    }
  }, [cameraPermission, requestCameraPermission]);



  if (!cameraPermission) {
    // Camera permissions are still loading.
    return (
      <View>
        <Text>Loading camera...</Text>
      </View>
    );
  }

  if (!cameraPermission.granted) {
    // Camera permissions are not granted yet.
    return (
      <View >
        <Text>We need your permission to show the camera</Text>
        <TouchableOpacity onPress={requestCameraPermission}>
          <Text>Grant Permission</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={() => router.back()}>
          <Text>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  function toggleCameraFacing() {
    setFacing((current) => (current === "back" ? "front" : "back"));
  }



  // Always attempt to show the camera (permission is requested automatically)
  return (

      <View className="flex-1 bg-[#F2F2F2]">
        <CameraView facing={facing} className="absolute inset-0" />


        {/* Overlay controls */}
        <SafeAreaView pointerEvents="box-none"
        className="absolute left-5 right-5 bottom-0 px-4 pb-3 items-center">

          {showTranslation && (
              <View className="mb-3 bg-white rounded-xl border border-[#B1B1B1] items-center justify-center h-60 max-w-md w-full">
                <View className="px-4 py-4">
                  <Text className="text-black text-xl font-semibold text-center">Vertaling</Text>
                </View>
              </View>
            )}

          <View className="items-center px-4 py-3 self-stretch">
            <View className="flex-row justify-center gap-3">
              <Button
                label="Back"
                className="bg-white px-10 py-4 border-2 rounded-lg border-[#B1B1B1]"
                labelClasses="text-black text-lg font-semibold"
                onPress={() => router.back()}
                size="lg"
                variant="secondary"
              />
              <Button
                label="Text"
                className="bg-white px-5 py-4 border-2 rounded-lg border-[#B1B1B1]"
                labelClasses="text-black text-lg font-semibold"
                onPress={() => setShowTranslation((v) => !v)}
                size="lg"
                variant="secondary"
              />
              <Button
                label="Flip"
                className="bg-white px-10 py-4 border-2 rounded-lg border-[#B1B1B1]"
                labelClasses="text-black text-lg font-semibold"
                onPress={toggleCameraFacing}
                size="lg"
                variant="secondary"
              />
            </View>
          </View>
        </SafeAreaView>
      </View>

  );
}
