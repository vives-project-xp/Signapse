// api.ts
import Constants from "expo-constants";

const BASE_URL =
  (Constants?.expoConfig?.extra as any)?.EXPO_PUBLIC_API_URL ||
  process.env.EXPO_PUBLIC_API_URL ||
  "http://127.0.0.1:8000/";

type FetchOptions = {
  method?: "GET" | "POST";
  headers?: Record<string, string>;
  body?: any;
  timeoutMs?: number;
};

async function safeReadText(res: Response) {
  try {
    return await res.text();
  } catch {
    return "";
  }
}

export async function apiFetch<T = any>(
  path: string,
  { method = "GET", headers = {}, body, timeoutMs = 8000 }: FetchOptions = {}
): Promise<T> {
  const url = BASE_URL.replace(/\/+$/, "") + "/" + path.replace(/^\/+/, "");

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  const res = await fetch(url, {
    method,
    headers: {
      Accept: "application/json",
      ...(body && !(body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...headers,
    },
    body: body && !(body instanceof FormData) ? JSON.stringify(body) : body,
    signal: controller.signal,
  }).catch((e) => {
    clearTimeout(timer);
    throw new Error(`Netwerkfout: ${e?.message ?? e}`);
  });

  clearTimeout(timer);

  if (!res.ok) {
    const msg = await safeReadText(res);
    throw new Error(`HTTP ${res.status}: ${msg || res.statusText}`);
  }

  return (await res.json()) as T;
}

export const api = {
  health: () => apiFetch<{ version: string }>("/health"),
  aslClasses: () => apiFetch<{ classes: string[] }>("/alphabet/asl/classes"),

  aslPredict: (landmarks: Array<{ x: number; y: number; z: number }>) =>
    apiFetch<{ prediction: string }>(
      "/alphabet/asl/predict",
      { method: "POST", body: { landmarks } }
    ),

  // iOS/Android
  keypointsFromImage: (uri: string) => {
    const form = new FormData();
    form.append(
      "image",
      {
        uri,
        name: "frame.jpg",
        type: "image/jpeg",
      } as any
    );
    return apiFetch<{ landmarks: { x: number; y: number; z: number }[] }>(
      "/keypoints/",
      { method: "POST", body: form }
    );
  },

  // Web File/Blob
  keypointsFromImageForm: (form: FormData) =>
    apiFetch<{ landmarks: { x: number; y: number; z: number }[] }>(
      "/keypoints/",
      { method: "POST", body: form }
    ),
};
