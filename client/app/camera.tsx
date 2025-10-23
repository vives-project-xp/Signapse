import { Button } from "@/components/Button";
import { CameraType, CameraView, useCameraPermissions } from "expo-camera";
import { useEffect, useState } from "react";
import { Text, TouchableOpacity, View } from "react-native";



export default function CameraScreen() {
  const [facing, setFacing] = useState<CameraType>("back");
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [showTranslation, setShowTranslation] = useState(false);

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
      </View>
    );
  }

  function toggleCameraFacing() {
    setFacing((current) => (current === "back" ? "front" : "back"));
  }



  // Always attempt to show the camera (permission is requested automatically)
  return (
    <View className="flex-1 bg-[#F2F2F2]">
      <View className="flex-1 items-center justify-center px-5">
        <View // Camera 
          className="w-full max-w-full rounded-2xl overflow-hidden bg-black border border-[#B1B1B1] mb-3"
          style={{ aspectRatio: 1, borderStyle: "dashed" }}
        >
          <CameraView facing={facing} className="w-full h-full" />
        </View> 

        {showTranslation && ( // Vertaling
          <View className="w-full bg-white rounded-xl border border-[#B1B1B1] p-6 mb-5">
            <Text className="text-black text-xl font-semibold text-center" style={{ aspectRatio: 2 }}>
              Vertaling
            </Text>
          </View>
        )}

      </View> 
      

      {/* Overlay controls */}
      <View pointerEvents="box-none" className="absolute left-0 right-0 bottom-0">
        <View className="items-center bg-white px-4 py-3 self-stretch">
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
      </View>
    </View>
  );
}
