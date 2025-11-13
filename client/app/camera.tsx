import { Button } from "@/components/Button";
import { LandmarksOverlay } from "@/components/LandmarksOverlay";
import api, { HttpError, NetworkError } from "@/lib/api";
import { useWordBuilder } from "@/lib/useWordBuilder";
import { CameraType, CameraView, useCameraPermissions } from "expo-camera";
import { router } from "expo-router";
import { useEffect, useRef, useState } from "react";
import { Platform, ScrollView, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

const CAPTURE_INTERVAL = 1000;

export default function CameraScreen() {
  const [facing, setFacing] = useState<CameraType>("back");
  const [permission, requestPermission] = useCameraPermissions();
  const [showLandmarks, setShowLandmarks] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [classes, setClasses] = useState<string[]>([]);
  const [landmarks, setLandmarks] = useState<{ x: number; y: number; z: number }[]>([]);

  const cameraRef = useRef<CameraView>(null);

  // Word builder hook with optimized settings for sign language
  const { currentWord, letterBuffer, addLetter, clearWord, deleteLastLetter, commitBuffer } =
    useWordBuilder({
      dwellTime: 800, // Hold letter for 800ms before adding
      idleTimeout: 2500, // 2.5s of no detection creates word boundary
      repeatWindow: 1500, // 1.5s window to detect intentional repeats
    });

  // Request permission and load classes on mount
  useEffect(() => {
    if (!permission?.granted) requestPermission();

    api
      .classes("vgt")
      .then((resp) => setClasses(resp.classes || []))
      .catch(console.warn);
  }, [permission?.granted, requestPermission]);

  // Capture and process images
  const capture = async () => {
    if (!cameraRef.current || !permission?.granted) return;

    try {
      const cam = cameraRef.current;
      const photo =
        (await cam.takePictureAsync?.({
          quality: 0.4,
          skipProcessing: true,
          base64: false,
          shutterSound: false,
          imageType: "jpg",
          exif: false,
        })) ??
        (await (cam as any).takePhoto?.({
          qualityPrioritization: "speed",
          flash: "off",
        }));

      if (!photo?.uri) throw new Error("Failed to capture photo");

      // Prepare image based on platform
      let imageData: string | File = photo.uri;
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
        const detectedLetter =
          !isNaN(index) && classes[index] ? classes[index] : result?.prediction || null;
        setPrediction(detectedLetter);

        // Feed letter to word builder
        addLetter(detectedLetter);
      } else {
        setLandmarks([]);
        setPrediction(null);

        // No hand detected
        addLetter(null);
      }
    } catch (error) {
      if (error instanceof HttpError && error.statusCode === 404) {
        setPrediction(null); // No hand detected
        setLandmarks([]);
        addLetter(null);
      } else if (error instanceof NetworkError) {
        // Network connectivity issue - show more helpful message
        console.error("Network error - Cannot reach server:", error.message);
        setPrediction(null);
        setLandmarks([]);
        addLetter(null);
      } else if (error instanceof Error && error.message === "Failed to capture photo") {
        // Ignore capture errors
      } else {
        console.warn("Capture error:", error);
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
    return (
      <View className="flex-1 items-center justify-center p-6">
        <Text className="mb-4 text-center">camera laden...</Text>
      </View>
    );
  }

  if (!permission || !permission.granted) {
    return (
      <View className="flex-1 items-center justify-center p-6">
        <Text className="mb-4 text-center">camera toegang is noodzakelijk</Text>
        <Button onPress={requestPermission} label="Toestemming geven" />
        <Button onPress={() => router.push("/")} label="Terug" />
      </View>
    );
  }

  const toggleCameraFacing = () => {
    setFacing((current) => (current === "back" ? "front" : "back"));
  };

  return (
    <View className="flex-1 bg-[#F2F2F2]">
      <CameraView ref={cameraRef} facing={facing} style={{ flex: 1 }} animateShutter={false} />
      <LandmarksOverlay
        landmarks={landmarks}
        visible={showLandmarks}
        mirrored={facing === "front"}
      />

      <SafeAreaView
        pointerEvents="box-none"
        className="absolute inset-x-0 bottom-4 px-4 sm:bottom-6 sm:px-6 md:bottom-8 md:px-8"
      >
        <View className="w-full px-4 sm:px-6 md:px-8">
          {/* Current Word Display */}
          <View className="mb-3 min-h-28 w-full max-w-2xl self-center rounded-xl border border-[#B1B1B1] bg-white px-4 py-3 md:min-h-32">
            <Text className="mb-1 text-xs font-medium text-gray-500 md:text-sm">Word</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} className="flex-1">
              <Text className="text-2xl font-bold text-black md:text-3xl">
                {currentWord || "—"}
                {letterBuffer && <Text className="text-blue-500 opacity-60">{letterBuffer}</Text>}
              </Text>
            </ScrollView>
          </View>

          {/* Current Letter Display */}
          <View className="mb-3 h-28 w-full max-w-2xl items-center justify-center self-center rounded-xl border border-[#B1B1B1] bg-white px-4 md:h-32">
            <Text className="mb-1 text-xs font-medium text-gray-500 md:text-sm">
              Current Letter
            </Text>
            <Text className="text-center text-4xl font-bold text-black md:text-5xl">
              {prediction || "—"}
            </Text>
          </View>

          {/* Word Controls */}
          <View className="mb-3 w-full max-w-2xl self-center">
            <View className="flex-row items-center justify-between gap-2 md:gap-3">
              <Button
                label="⌫ Delete"
                className="h-12 flex-1 rounded-lg border-2 border-[#B1B1B1] bg-white sm:h-14 md:h-16"
                labelClasses="text-black text-base sm:text-lg md:text-xl font-semibold"
                onPress={deleteLastLetter}
                size="lg"
                variant="secondary"
              />
              <Button
                label="✓ Add"
                className="h-12 flex-1 rounded-lg border-2 border-[#B1B1B1] bg-white sm:h-14 md:h-16"
                labelClasses="text-black text-base sm:text-lg md:text-xl font-semibold"
                onPress={commitBuffer}
                size="lg"
                variant="secondary"
              />
              <Button
                label="✗ Clear"
                className="h-12 flex-1 rounded-lg border-2 border-[#B1B1B1] bg-white sm:h-14 md:h-16"
                labelClasses="text-black text-base sm:text-lg md:text-xl font-semibold"
                onPress={clearWord}
                size="lg"
                variant="secondary"
              />
            </View>
          </View>

          {/* Camera Controls */}
          <View className="w-full max-w-2xl self-center">
            <View className="flex-row items-center justify-between gap-2 md:gap-3">
              <Button
                label="Back"
                className="h-12 flex-1 rounded-lg border-2 border-[#B1B1B1] bg-white sm:h-14 md:h-16"
                labelClasses="text-black text-base sm:text-lg md:text-xl font-semibold"
                onPress={() => router.push("/")}
                size="lg"
                variant="secondary"
              />
              <Button
                label={`Landmarks ${showLandmarks ? "On" : "Off"}`}
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
