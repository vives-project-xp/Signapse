import { type VariantProps, cva } from "class-variance-authority";
import { Text, TouchableOpacity } from "react-native";

import { cn } from "../lib/utils";

const buttonVariants = cva("flex flex-row items-center justify-center rounded-md", {
  variants: {
    variant: {
      default: "bg-primary",
      secondary: "bg-secondary",
      destructive: "bg-destructive",
      ghost: "bg-slate-700",
      link: "text-primary underline-offset-4",
    },
    size: {
      // reduce default padding so buttons are less tall by default
      default: "py-1 px-4",
      sm: "py-1 px-2",
      lg: "py-2 px-6",
    },
  },
  defaultVariants: {
    variant: "default",
    size: "default",
  },
});

const buttonTextVariants = cva("text-center font-medium", {
  variants: {
    variant: {
      default: "text-primary-foreground",
      secondary: "text-secondary-foreground",
      destructive: "text-destructive-foreground",
      ghost: "text-primary-foreground",
      link: "text-primary-foreground underline",
    },
    size: {
      default: "text-base",
      sm: "text-md",
      lg: "text-xl",
    },
  },
  defaultVariants: {
    variant: "default",
    size: "default",
  },
});

interface ButtonProps
  extends React.ComponentPropsWithoutRef<typeof TouchableOpacity>,
    VariantProps<typeof buttonVariants> {
  label: string;
  labelClasses?: string;
}
function Button({ label, labelClasses, className, variant, size, ...props }: ButtonProps) {
  // ensure a consistent fontSize per size variant
  const computedFontSize = size === "lg" ? 18 : size === "sm" ? 14 : 16;
  return (
    <TouchableOpacity className={cn(buttonVariants({ variant, size, className }))} {...props}>
      <Text
        className={cn(buttonTextVariants({ variant, size, className: labelClasses }))}
        allowFontScaling
        style={{ fontSize: computedFontSize, textAlign: "center", width: "100%", flexWrap: "wrap" }}
      >
        {label}
      </Text>
    </TouchableOpacity>
  );
}

export { Button, buttonTextVariants, buttonVariants };
