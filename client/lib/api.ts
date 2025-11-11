import Constants from "expo-constants";

const BASE_URL =
  (Constants?.expoConfig?.extra as any)?.EXPO_PUBLIC_API_URL ||
  process.env.EXPO_PUBLIC_API_URL ||
  "http://127.0.0.1:8000/";

// Custom error classes for better error handling
export class ApiError extends Error {
  constructor(
    message: string,
    public code?: string
  ) {
    super(message);
    this.name = "ApiError";
    Object.setPrototypeOf(this, ApiError.prototype);
  }
}

export class NetworkError extends ApiError {
  constructor(
    message: string,
    public originalError?: any
  ) {
    super(message, "NETWORK_ERROR");
    this.name = "NetworkError";
    Object.setPrototypeOf(this, NetworkError.prototype);
  }
}

export class TimeoutError extends ApiError {
  constructor(message: string = "Request timed out") {
    super(message, "TIMEOUT_ERROR");
    this.name = "TimeoutError";
    Object.setPrototypeOf(this, TimeoutError.prototype);
  }
}

export class HttpError extends ApiError {
  public status: number;
  constructor(
    message: string,
    public statusCode: number,
    public statusText: string,
    public responseBody?: string
  ) {
    super(message, `HTTP_${statusCode}`);
    this.name = "HttpError";
    this.status = statusCode;
    Object.setPrototypeOf(this, HttpError.prototype);
  }
}

export class ParseError extends ApiError {
  constructor(
    message: string,
    public originalError?: any
  ) {
    super(message, "PARSE_ERROR");
    this.name = "ParseError";
    Object.setPrototypeOf(this, ParseError.prototype);
  }
}

type FetchOptions = {
  method?: "GET" | "POST";
  headers?: Record<string, string>;
  body?: any;
  timeoutMs?: number;
};

async function safeReadText(res: Response): Promise<string> {
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

  let res: Response;

  try {
    res = await fetch(url, {
      method,
      headers: {
        Accept: "application/json",
        ...(body && !(body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
        ...headers,
      },
      body: body && !(body instanceof FormData) ? JSON.stringify(body) : body,
      signal: controller.signal,
    });
  } catch (error: any) {
    clearTimeout(timer);

    // Handle AbortController timeout
    if (error.name === "AbortError") {
      throw new TimeoutError(`Request to ${path} timed out after ${timeoutMs}ms`);
    }

    // Handle network errors
    throw new NetworkError(
      `Network error while fetching ${path}: ${error?.message ?? "Unknown error"}`,
      error
    );
  } finally {
    clearTimeout(timer);
  }

  // Handle HTTP errors
  if (!res.ok) {
    const responseText = await safeReadText(res);

    // Try to extract error details from response
    let errorDetails;
    try {
      errorDetails = responseText ? JSON.parse(responseText) : null;
    } catch {
      // Response is not JSON, use as-is
      errorDetails = responseText;
    }

    throw new HttpError(
      errorDetails?.detail || errorDetails || res.statusText,
      res.status,
      res.statusText,
      responseText
    );
  }

  // Parse JSON response
  try {
    return (await res.json()) as T;
  } catch (error: any) {
    throw new ParseError(
      `Failed to parse JSON response from ${path}: ${error?.message ?? "Unknown error"}`,
      error
    );
  }
}

/**
 * API client for the SmartGlasses backend
 *
 * All methods can throw the following errors:
 * - NetworkError: Network connectivity issues
 * - TimeoutError: Request exceeded timeout limit
 * - HttpError: HTTP error responses (4xx, 5xx)
 * - ParseError: Failed to parse response
 */
const api = {
  /**
   * Check API health status
   * @throws {NetworkError} If network request fails
   * @throws {TimeoutError} If request times out
   * @throws {HttpError} If server returns error status
   * @throws {ParseError} If response cannot be parsed
   */
  health: () => apiFetch<{ version: string }>("/health"),

  /**
   * Get available classes for a sign language model
   * @param model - The model to query ("asl" or "vgt")
   * @throws {NetworkError} If network request fails
   * @throws {TimeoutError} If request times out
   * @throws {HttpError} If server returns error status
   * @throws {ParseError} If response cannot be parsed
   */
  classes: (model: "asl" | "vgt") => {
    return apiFetch<{ classes: string[] }>(`/alphabet/${model}/classes`);
  },

  /**
   * Predict sign language gesture from landmarks
   * @param model - The model to use ("asl" or "vgt")
   * @param landmarks - Array of 3D landmark coordinates
   * @throws {NetworkError} If network request fails
   * @throws {TimeoutError} If request times out
   * @throws {HttpError} If server returns error status
   * @throws {ParseError} If response cannot be parsed
   */
  predict: (model: "asl" | "vgt", landmarks: { x: number; y: number; z: number }[]) => {
    return apiFetch<{ prediction: string }>(`/alphabet/${model}/predict`, {
      method: "POST",
      body: { landmarks },
    });
  },

  /**
   * Upload image to keypoints endpoint and extract landmarks
   * @param image - Image URI (iOS/Android) or Blob/File (web)
   * @throws {NetworkError} If network request fails
   * @throws {TimeoutError} If request times out
   * @throws {HttpError} If server returns error status
   * @throws {ParseError} If response cannot be parsed
   */
  keypointsFromImage: (image: string | Blob | File) => {
    const form = new FormData();

    if (typeof image === "string") {
      // React Native style file object
      form.append("image", {
        uri: image,
        name: "frame.jpg",
        type: "image/jpeg",
      } as any);
    } else {
      // Web: append actual File/Blob
      const file =
        image instanceof File ? image : new File([image], "frame.jpg", { type: "image/jpeg" });
      form.append("image", file);
    }

    return apiFetch<{ landmarks: { x: number; y: number; z: number }[] }>("/keypoints/", {
      method: "POST",
      body: form,
    });
  },
};

export default api;
