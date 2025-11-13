import { useCallback, useRef, useState } from "react";

/**
 * Configuration for the word builder algorithm
 */
interface WordBuilderConfig {
  /** Minimum time (ms) to hold a letter before it's considered stable (default: 800ms) */
  dwellTime?: number;
  /** Time (ms) of no detection before considering it a word boundary (default: 2000ms) */
  idleTimeout?: number;
  /** Maximum time (ms) to wait for the same letter to reappear for intentional repeats (default: 1500ms) */
  repeatWindow?: number;
}

/**
 * Hook for building words from detected sign language letters
 *
 * Algorithm:
 * - Requires a letter to be held steady for `dwellTime` before adding it
 * - Prevents duplicate letters from hand staying still
 * - Allows intentional repeated letters if hand is removed and letter shown again
 * - Creates word boundaries after `idleTimeout` of no detection
 *
 * @example
 * const { currentWord, addLetter, clearWord, deleteLastLetter } = useWordBuilder();
 * // When letter detected: addLetter('A')
 * // Display: currentWord
 */
export function useWordBuilder(config: WordBuilderConfig = {}) {
  const { dwellTime = 800, idleTimeout = 2000, repeatWindow = 1500 } = config;

  const [currentWord, setCurrentWord] = useState("");
  const [letterBuffer, setLetterBuffer] = useState("");

  // Tracking state
  const lastLetterRef = useRef<string | null>(null);
  const lastDetectionTimeRef = useRef<number>(0);
  const letterStartTimeRef = useRef<number>(0);
  const letterAddedRef = useRef(false);
  const lastAddedLetterRef = useRef<string | null>(null);
  const lastAddedTimeRef = useRef<number>(0);
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Use refs for state that changes but shouldn't trigger re-renders in callback deps
  const currentWordRef = useRef(currentWord);
  const letterBufferRef = useRef(letterBuffer);

  currentWordRef.current = currentWord;
  letterBufferRef.current = letterBuffer;

  /**
   * Process a detected letter from the sign language model
   * @param letter - The detected letter (single character) or null if no hand detected
   */
  const addLetter = useCallback(
    (letter: string | null) => {
      const now = Date.now();

      // Clear idle timer since we have activity
      if (idleTimerRef.current) {
        clearTimeout(idleTimerRef.current);
        idleTimerRef.current = null;
      }

      // No hand detected
      if (!letter) {
        // Start idle timer for word boundary
        if (currentWordRef.current.length > 0 || letterBufferRef.current.length > 0) {
          idleTimerRef.current = setTimeout(() => {
            if (letterBufferRef.current.length > 0) {
              setCurrentWord((prev) => prev + letterBufferRef.current);
              setLetterBuffer("");
            }
          }, idleTimeout);
        }

        lastLetterRef.current = null;
        lastDetectionTimeRef.current = 0;
        letterStartTimeRef.current = 0;
        letterAddedRef.current = false;
        return;
      }

      lastDetectionTimeRef.current = now;

      // New letter detected
      if (letter !== lastLetterRef.current) {
        // Check if this is an intentional repeat of the last added letter
        const isIntentionalRepeat =
          letter === lastAddedLetterRef.current && now - lastAddedTimeRef.current < repeatWindow;

        if (!isIntentionalRepeat) {
          // Commit any buffered letter before starting new one
          if (letterBufferRef.current.length > 0 && letterAddedRef.current) {
            setCurrentWord((prev) => prev + letterBufferRef.current);
            setLetterBuffer("");
          }
        }

        lastLetterRef.current = letter;
        letterStartTimeRef.current = now;
        letterAddedRef.current = false;

        // Show letter in buffer immediately for visual feedback
        setLetterBuffer(letter);
      }

      // Letter held steady - check if we should add it
      if (letter === lastLetterRef.current && !letterAddedRef.current) {
        const holdDuration = now - letterStartTimeRef.current;

        if (holdDuration >= dwellTime) {
          // Check if this is a duplicate of the last added letter
          const isDuplicate =
            letter === lastAddedLetterRef.current && now - lastAddedTimeRef.current < repeatWindow;

          if (!isDuplicate) {
            // Add the letter to the word
            setCurrentWord((prev) => prev + letter);
            setLetterBuffer("");
            letterAddedRef.current = true;
            lastAddedLetterRef.current = letter;
            lastAddedTimeRef.current = now;
          } else {
            // This is likely just holding hand still after adding a letter
            setLetterBuffer("");
            letterAddedRef.current = true;
          }
        }
      }
    },
    [dwellTime, idleTimeout, repeatWindow]
  );

  /**
   * Clear the entire word
   */
  const clearWord = useCallback(() => {
    setCurrentWord("");
    setLetterBuffer("");
    lastLetterRef.current = null;
    lastDetectionTimeRef.current = 0;
    letterStartTimeRef.current = 0;
    letterAddedRef.current = false;
    lastAddedLetterRef.current = null;
    lastAddedTimeRef.current = 0;

    if (idleTimerRef.current) {
      clearTimeout(idleTimerRef.current);
      idleTimerRef.current = null;
    }
  }, []);

  /**
   * Delete the last letter from the word
   */
  const deleteLastLetter = useCallback(() => {
    setCurrentWord((prev) => prev.slice(0, -1));
    setLetterBuffer("");

    // Reset tracking to allow same letter to be added again
    lastAddedLetterRef.current = null;
    lastAddedTimeRef.current = 0;
  }, []);

  /**
   * Force commit any buffered letter to the word
   */
  const commitBuffer = useCallback(() => {
    if (letterBuffer.length > 0) {
      setCurrentWord((prev) => prev + letterBuffer);
      setLetterBuffer("");
      letterAddedRef.current = true;
      lastAddedLetterRef.current = letterBuffer;
      lastAddedTimeRef.current = Date.now();
    }
  }, [letterBuffer]);

  return {
    /** The current word being built */
    currentWord,
    /** Letter currently being detected (not yet committed) */
    letterBuffer,
    /** Add a detected letter (or null for no detection) */
    addLetter,
    /** Clear the entire word */
    clearWord,
    /** Delete the last letter from the word */
    deleteLastLetter,
    /** Force commit the buffer to the word */
    commitBuffer,
  };
}
