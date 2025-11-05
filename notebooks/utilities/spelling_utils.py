# spelling_utils.py
from __future__ import annotations
from typing import List, Optional, Tuple
import re
import time

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  
    _HAS_HF = True
except Exception:
    _HAS_HF = False


class WordAssembler:
    """
    Minimalistische assembler: voegt letters samen tot woorden en doet lichte postprocessing.
    Optioneel: neural polish met een NL spelling/grammar model (HF).
    """
    def __init__(self, hf_model_id: str | None = None, max_new_tokens: int = 64):
        self._tok = None
        self._mdl = None
        if hf_model_id and _HAS_HF:
            self._tok = AutoTokenizer.from_pretrained(hf_model_id)
            self._mdl = AutoModelForSeq2SeqLM.from_pretrained(hf_model_id)
        self._max_new = max_new_tokens

    @staticmethod
    def letters_to_words(chars: List[str]) -> str:
        # samenvouwen: dubbele identieke letters achter elkaar; spaties blijven
        buf, prev = [], None
        for c in chars:
            if c == " ":
                buf.append(" ")
                prev = None
                continue
            if c != prev:
                buf.append(c)
            prev = c
        text = "".join(buf)
        text = re.sub(r"\s+", " ", text).strip()

        # simpele naam-normalisatie:
        def _cap_token(tok: str) -> str:
            if tok.isalpha() and tok.upper() == tok and len(tok) >= 2:
                return tok.capitalize()
            return tok
        text = " ".join(_cap_token(t) for t in text.split(" "))
        return text

    def polish(self, text: str) -> str:
        if self._tok is None or self._mdl is None:
            return text
        x = self._tok(text, return_tensors="pt")
        y = self._mdl.generate(**x, max_new_tokens=self._max_new)
        return self._tok.decode(y[0], skip_special_tokens=True)


class TemporalSpeller:
    """
    Online decoder zonder LetterStream:
    - Neem frame-voor-frame top-1 label + confidence
    - Emiteer een letter pas als 'dwell' (aantal opeenvolgende acceptabele frames) gehaald is
    - Negeer frames < conf_min
    - Boundary als er 'idle_s' seconden geen hand werd gezien (extern door caller doorgegeven)
    - Anti-spam: min_word_len voor we een woord printen
    """
    def __init__(
        self,
        conf_min: float = 0.60,
        dwell_frames: int = 2,
        min_word_len: int = 2,
    ) -> None:
        self.conf_min = conf_min
        self.dwell_frames = dwell_frames
        self.min_word_len = min_word_len

        self._cur_label: Optional[str] = None
        self._run_len: int = 0
        self._run_accum_conf: float = 0.0

        self._emitted_chars: List[str] = []
        self._last_word_emit_ts: float = 0.0

    def consume(self, label: str, conf: float) -> Optional[str]:
        """
        Verwerk één frame. Retourneert de geëmiteerde letter (str) of None.
        """
        if conf < self.conf_min:
            # lage zekerheid: reset huidige run
            self._cur_label = None
            self._run_len = 0
            self._run_accum_conf = 0.0
            return None

        if self._cur_label is None or label != self._cur_label:
            # nieuwe run
            self._cur_label = label
            self._run_len = 1
            self._run_accum_conf = conf
            return None

        # zelfde label blijft lopen
        self._run_len += 1
        self._run_accum_conf += conf

        if self._run_len >= self.dwell_frames:
            # emit en reset run (hysterese: forceer nieuwe run voor volgende letter)
            emitted = self._cur_label
            self._cur_label = None
            self._run_len = 0
            self._run_accum_conf = 0.0
            self._emitted_chars.append(emitted)
            return emitted

        return None

    def force_boundary(self) -> Optional[str]:
        """
        Forceer woordgrens (bijv. op SPACE-toets). Retourneert het woord (gepolijst) of None.
        (Polish gebeurt extern via WordAssembler.)
        """
        if len([c for c in self._emitted_chars if c != " "]) >= self.min_word_len:
            word = "".join(self._emitted_chars).replace("  ", " ").strip()
            self._emitted_chars.clear()
            self._last_word_emit_ts = time.time()
            return word
        self._emitted_chars.clear()
        return None

    def boundary_if_idle(self, last_hand_seen_ts: float, idle_s: float) -> Optional[str]:
        """
        Check idle gap en emit boundary indien nodig. Retourneert woord of None.
        """
        now = time.time()
        if (now - last_hand_seen_ts) >= idle_s and (now - self._last_word_emit_ts) >= max(0.75, 0.9 * idle_s):
            return self.force_boundary()
        return None

    @property
    def live_buffer(self) -> str:
        return "".join(self._emitted_chars)

    def reset(self) -> None:
        self._cur_label = None
        self._run_len = 0
        self._run_accum_conf = 0.0
        self._emitted_chars.clear()
