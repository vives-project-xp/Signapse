import { Dimensions } from "react-native";
import Svg, { Circle, Line } from "react-native-svg";

// Hand landmark connections (MediaPipe hand model)
const HAND_CONNECTIONS = [
  // Thumb
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  // Index finger
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  // Middle finger
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  // Ring finger
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  // Pinky
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  // Palm
  [5, 9],
  [9, 13],
  [13, 17],
];

interface Landmark {
  x: number;
  y: number;
  z: number;
}

interface LandmarksOverlayProps {
  landmarks: Landmark[];
  visible?: boolean;
  connectionColor?: string;
  pointColor?: string;
  wristColor?: string;
  connectionWidth?: number;
  pointRadius?: number;
  mirrored?: boolean;
}

export function LandmarksOverlay({
  landmarks,
  visible = true,
  connectionColor = "#00FF00",
  pointColor = "#00FF00",
  wristColor = "#FF0000",
  connectionWidth = 2,
  pointRadius = 4,
  mirrored = true,
}: LandmarksOverlayProps) {
  if (!visible || landmarks.length !== 21) return null;

  const { width, height } = Dimensions.get("window");

  return (
    <Svg
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width,
        height,
      }}
      pointerEvents="none"
    >
      {/* Draw connections */}
      {HAND_CONNECTIONS.map(([start, end], idx) => {
        const startPoint = landmarks[start];
        const endPoint = landmarks[end];

        return (
          <Line
            key={`line-${idx}`}
            x1={mirrored ? width - startPoint.x * width : startPoint.x * width}
            y1={startPoint.y * height}
            x2={mirrored ? width - endPoint.x * width : endPoint.x * width}
            y2={endPoint.y * height}
            stroke={connectionColor}
            strokeWidth={connectionWidth}
          />
        );
      })}

      {/* Draw landmark points */}
      {landmarks.map((landmark, idx) => (
        <Circle
          key={`point-${idx}`}
          cx={mirrored ? width - landmark.x * width : landmark.x * width}
          cy={landmark.y * height}
          r={pointRadius}
          fill={idx === 0 ? wristColor : pointColor}
          stroke="#FFFFFF"
          strokeWidth="1"
        />
      ))}
    </Svg>
  );
}
