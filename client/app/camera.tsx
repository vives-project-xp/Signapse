import { Button } from "@/components/Button";
import { LandmarksOverlay } from "@/components/LandmarksOverlay";
import api, { HttpError } from "@/lib/api";
import { CameraType, CameraView, useCameraPermissions } from "expo-camera";
import { router } from "expo-router";
import { useEffect, useRef, useState } from "react";
import { Platform, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

const CAPTURE_INTERVAL = 500;

export default function CameraScreen() {
  const [facing, setFacing] = useState<CameraType>("back");
  const [permission, requestPermission] = useCameraPermissions();
  const [showTranslation, setShowTranslation] = useState(false);
  const [showLandmarks, setShowLandmarks] = useState(false);
  const [prediction, setPrediction] = useState<string>("");
  const [classes, setClasses] = useState<string[]>([]);
  const [landmarks, setLandmarks] = useState<{ x: number; y: number; z: number }[]>([]);

  const cameraRef = useRef<CameraView>(null);

  // Request permission and load classes on mount
  useEffect(() => {
    if (!permission?.granted) requestPermission();

    api
      .classes("vgt")
      .then((resp) => setClasses(resp.classes || []))
      .catch(console.warn);
  }, [permission, requestPermission]);

  // Capture and process images
  const capture = async () => {
    if (!cameraRef.current || !permission?.granted) return;

    try {
      const cam = cameraRef.current as any;
      const photo =
        (await cam.takePictureAsync?.({
          quality: 0.4,
          skipProcessing: true,
          base64: false,
        })) ?? (await cam.takePhoto?.({ qualityPrioritization: "speed" }));

      if (!photo?.uri) return;

      // Prepare image based on platform
      let imageData = photo.uri;
      if (Platform.OS === "web") {
        const response = await fetch(photo.uri);
        const blob = await response.blob();
        imageData = new File([blob], "frame.jpg", { type: "image/jpeg" });
      }

      // Get landmarks and predict
      const { landmarks: detectedLandmarks } = await api.keypointsFromImage(imageData);

      if (detectedLandmarks.length === 21) {
        setLandmarks(detectedLandmarks);
        const result = await api.predict("vgt", detectedLandmarks);
        const index = Number(result?.prediction);
        setPrediction(!isNaN(index) && classes[index] ? classes[index] : result?.prediction || "");
      } else {
        setLandmarks([]);
      }
    } catch (error) {
      if (error instanceof HttpError && error.statusCode === 404) {
        setPrediction(""); // No hand detected
        setLandmarks([]);
      }
    }
  };

  // Auto-capture loop
  useEffect(() => {
    if (!permission?.granted) return;

    const interval = setInterval(capture, CAPTURE_INTERVAL);
    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [permission?.granted, classes]);

  if (!permission) {
    return <Text>Loading camera...</Text>;
  }

  if (!permission.granted) {
    return (
      <View className="flex-1 items-center justify-center bg-black p-6">
        <Text className="mb-4 text-center text-white">Camera permission is required</Text>
        <Button onPress={requestPermission} label="Grant permission" />
      </View>
    );
  }

  const toggleCameraFacing = () => {
    setFacing((current) => (current === "back" ? "front" : "back"));
  };

  return (
    <View className="flex-1 bg-[#F2F2F2]">
      <CameraView ref={cameraRef} facing={facing} style={{ flex: 1 }} />
      <LandmarksOverlay landmarks={landmarks} visible={showLandmarks} />

      <SafeAreaView
        pointerEvents="box-none"
        className="absolute inset-x-0 bottom-4 px-4 sm:bottom-6 sm:px-6 md:bottom-8 md:px-8"
      >
        <View className="w-full px-4 sm:px-6 md:px-8">
          {showTranslation && (
            <View className="mb-3 h-56 w-full max-w-2xl items-center justify-center self-center rounded-xl border border-[#B1B1B1] bg-white px-4 md:h-64 lg:h-72">
              <Text className="text-center text-xl font-semibold text-black md:text-2xl">
                {prediction || "—"}
              </Text>
            </View>
          )}

          <View className="w-full max-w-2xl self-center">
            <View className="flex-row items-center justify-between gap-2 md:gap-3">
              <Button
                label="Back"
                className="h-12 flex-1 rounded-lg border-2 border-[#B1B1B1] bg-white sm:h-14 md:h-16"
                labelClasses="text-black text-base sm:text-lg md:text-xl font-semibold"
                onPress={() => router.back()}
                size="lg"
                variant="secondary"
              />
              <Button
                label="Text"
                className="h-12 flex-1 rounded-lg border-2 border-[#B1B1B1] bg-white sm:h-14 md:h-16"
                labelClasses="text-black text-base sm:text-lg md:text-xl font-semibold"
                onPress={() => setShowTranslation((v) => !v)}
                size="lg"
                variant="secondary"
              />
              <Button
                label={`Landmarks (${showLandmarks ? "On" : "Off"})`}
                className="h-12 flex-1 rounded-lg border-2 border-[#B1B1B1] bg-white sm:h-14 md:h-16"
                labelClasses="text-black text-base sm:text-lg md:text-xl font-semibold"
                onPress={() => setShowLandmarks((v) => !v)}
                size="lg"
                variant="secondary"
              />
              <Button
                label="Flip"
                className="h-12 flex-1 rounded-lg border-2 border-[#B1B1B1] bg-white sm:h-14 md:h-16"
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
