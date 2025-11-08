import { Button } from "@/components/Button";
import { CameraType, CameraView, useCameraPermissions } from "expo-camera";
import { router } from "expo-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { AppState, AppStateStatus, Platform, Text, View } from "react-native";

import { SafeAreaView } from "react-native-safe-area-context";

import { api } from "@/lib/api";

export default function CameraScreen() {
  const [facing, setFacing] = useState<CameraType>("back");
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [showTranslation, setShowTranslation] = useState(false);  // voor vertalingd

  const [landmarks, setLandmarks] = useState<Array<{ x: number; y: number; z: number }>>([]);
  const [prediction, setPrediction] = useState<string | null>(null);

  const [classes, setClasses] = useState<string[]>([]);

  const cameraRef = useRef<CameraView | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const uploadingRef = useRef(false);
  const appStateRef = useRef<AppStateStatus>(AppState.currentState);

  // Request permission automatically on mount so camera opens when entering screen
  useEffect(() => {
    if (!cameraPermission?.granted) {
      // calling requestPermission will prompt the user on first run
      requestCameraPermission();
    }
  }, [cameraPermission, requestCameraPermission]);

useEffect(() => {
    (async () => {
      try {
        const resp = await api.aslClasses(); // { classes: string[] }
        setClasses(resp.classes || []);
        console.log("ASL classes:", resp.classes);
      } catch (e) {
        console.warn("aslClasses failed:", e);
      }
    })();
  }, []);

  useEffect(() => {
    const sub = AppState.addEventListener("change", (next) => {
      const becameInactive =
        appStateRef.current === "active" && next.match(/inactive|background/);
      const becameActive =
        appStateRef.current.match(/inactive|background/) && next === "active";
      appStateRef.current = next;

      if (becameInactive) stopCaptureLoop();
      if (becameActive) startCaptureLoop();
    });
    return () => sub.remove();
  }, []);

  const startCaptureLoop = useCallback(() => {
    if (intervalRef.current) return;
    intervalRef.current = setInterval(captureAndSend, 700);
  }, []);

  const stopCaptureLoop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

    useEffect(() => {
    if (cameraPermission?.granted) startCaptureLoop();
    return stopCaptureLoop;
  }, [cameraPermission?.granted, startCaptureLoop, stopCaptureLoop]);

    const normalizePrediction = (raw: string | null, cls: string[]) => {
    if (!raw) return "";
    const maybeIdx = Number(raw);
    if (!Number.isNaN(maybeIdx) && Number.isInteger(maybeIdx) && cls[maybeIdx]) {
      return cls[maybeIdx];
    }
    return raw;
  };

  const captureAndSend = useCallback(async () => {
    if (uploadingRef.current) return;
    const cam = cameraRef.current as any;
    if (!cam) return;

    try {
      uploadingRef.current = true;

      const photo =
        (await cam.takePictureAsync?.({
          quality: 0.4,
          skipProcessing: true,
          base64: false,
        })) ??
        (await cam.takePhoto?.({ qualityPrioritization: "speed" }));

      if (!photo?.uri) {
        console.warn("Geen foto-URI ontvangen");
        return;
      }

      let keypointsResp:
        | { landmarks: { x: number; y: number; z: number }[] }
        | undefined;

      if (Platform.OS === "web") {
        const resp = await fetch(photo.uri);
        const blob = await resp.blob();
        const file = new File([blob], "frame.jpg", { type: blob.type || "image/jpeg" });
        const form = new FormData();
        form.append("image", file);
        keypointsResp = await api.keypointsFromImageForm(form);
      } else {
        keypointsResp = await api.keypointsFromImage(photo.uri);
      }

      const kp = keypointsResp?.landmarks ?? [];
      setLandmarks(kp);

      if (kp.length === 21) {
        const pred = await api.aslPredict(kp);
        const pretty = normalizePrediction(pred?.prediction ?? "", classes);
        setPrediction(pretty);
      } else {
        setPrediction("");
      }
    } catch (e) {
      console.warn("captureAndSend error:", e);
    } finally {
      uploadingRef.current = false;
    }
  }, [classes]);

  if (!cameraPermission) {
    return <Text>Camera laden...</Text>
  }

if (!cameraPermission?.granted) {
  return (
    <View className="flex-1 items-center justify-center bg-black p-6">
      <Text className="text-white text-center mb-4">
        We hebben uw toestemming nodig om de camera te laten zien.
      </Text>
      <Button onPress={requestCameraPermission} label="Grant permission" />
    </View>
  );
}

  function toggleCameraFacing() {
    setFacing((current) => (current === "back" ? "front" : "back"));
  }

  return (

      <View className="flex-1 bg-[#F2F2F2]">
        <CameraView ref={cameraRef} facing={facing} style={{flex: 1}} />

        {/* Overlay controls */}
        <SafeAreaView pointerEvents="box-none"
        className="absolute inset-x-0 bottom-4 sm:bottom-6 md:bottom-8 px-4 sm:px-6 md:px-8" >

          <View className="w-full px-4 sm:px-6 md:px-8">

            {showTranslation && (
                <View className="mb-3 bg-white rounded-xl border border-[#B1B1B1] items-center justify-center w-full max-w-2xl self-center h-56 md:h-64 lg:h-72 px-4">
                    <Text className="text-black text-xl md:text-2xl font-semibold text-center">{prediction || "—"}</Text>
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
