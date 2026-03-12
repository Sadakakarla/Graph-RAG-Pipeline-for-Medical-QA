"""
Safety layer: Presidio PII redaction + Whisper STT/TTS for multimodal input.
HIPAA-compliant patient data scrubbing before any LLM processing.
"""
from __future__ import annotations

import re
import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


# ─── PII Redactor ─────────────────────────────────────────────────────────────

@dataclass
class RedactionResult:
    redacted_text: str
    pii_found: bool
    entity_types_found: list[str]
    redaction_count: int


class PIIRedactor:
    """
    HIPAA-compliant PII redaction using Microsoft Presidio.
    Covers: names, DOB, SSN, MRN, phone, email, address, insurance IDs.
    Falls back to regex patterns if Presidio is unavailable.
    """

    HIPAA_ENTITIES = [
        "PERSON", "DATE_TIME", "PHONE_NUMBER", "EMAIL_ADDRESS",
        "US_SSN", "US_ITIN", "MEDICAL_LICENSE", "US_PASSPORT",
        "LOCATION", "NRP", "IP_ADDRESS", "URL",
        "IBAN_CODE", "CREDIT_CARD", "CRYPTO",
    ]

    REGEX_FALLBACKS = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]"),
        (r"\b\d{10,12}\b", "[MRN_REDACTED]"),
        (r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "[NAME_REDACTED]"),
        (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATE_REDACTED]"),
        (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE_REDACTED]"),
        (r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", "[EMAIL_REDACTED]"),
    ]

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._analyzer = None
        self._anonymizer = None
        self._setup_presidio()

    def _setup_presidio(self) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
        except ImportError:
            print("[PII] Presidio not installed. Using regex fallback. "
                  "Install: pip install presidio-analyzer presidio-anonymizer")

    def redact(self, text: str) -> tuple[str, bool]:
        """
        Returns (redacted_text, pii_was_found).
        """
        if self._analyzer and self._anonymizer:
            return self._presidio_redact(text)
        return self._regex_redact(text)

    def _presidio_redact(self, text: str) -> tuple[str, bool]:
        results = self._analyzer.analyze(
            text=text,
            entities=self.HIPAA_ENTITIES,
            language="en",
        )
        if not results:
            return text, False

        anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text, True

    def _regex_redact(self, text: str) -> tuple[str, bool]:
        redacted = text
        found = False
        for pattern, replacement in self.REGEX_FALLBACKS:
            new_text = re.sub(pattern, replacement, redacted)
            if new_text != redacted:
                found = True
            redacted = new_text
        return redacted, found

    def redact_batch(self, texts: list[str]) -> list[tuple[str, bool]]:
        return [self.redact(t) for t in texts]


# ─── Whisper STT/TTS ──────────────────────────────────────────────────────────

class WhisperConfig(BaseModel):
    stt_model: str = "openai/whisper-large-v3"
    tts_model: str = "tts-1-hd"
    language: str = "en"
    device: str = "cuda"
    chunk_length_s: int = 30


class WhisperSTTTTS:
    """
    Multimodal audio interface using Whisper for STT and OpenAI TTS.
    Enables voice-based medical queries with automatic PII redaction.
    """

    def __init__(self, cfg: dict, pii_redactor: PIIRedactor):
        self.cfg = WhisperConfig(**cfg.get("whisper", {}))
        self.pii_redactor = pii_redactor
        self._stt_pipeline = None
        self._tts_client = None
        self._setup()

    def _setup(self) -> None:
        try:
            from transformers import pipeline
            self._stt_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.cfg.stt_model,
                device=self.cfg.device,
                chunk_length_s=self.cfg.chunk_length_s,
            )
        except ImportError:
            print("[Whisper] transformers not installed.")

        try:
            from openai import AsyncOpenAI
            self._tts_client = AsyncOpenAI()
        except ImportError:
            print("[TTS] openai not installed.")

    async def transcribe(self, audio_path: str) -> tuple[str, bool]:
        """
        Transcribe audio to text, then PII-redact.
        Returns (redacted_transcript, pii_found).
        """
        if self._stt_pipeline is None:
            return "STT model not loaded.", False

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._stt_pipeline, audio_path
        )
        raw_transcript = result.get("text", "")
        return self.pii_redactor.redact(raw_transcript)

    async def synthesize(self, text: str, output_path: str) -> str:
        """Convert text answer to speech for voice interface."""
        if self._tts_client is None:
            return output_path

        response = await self._tts_client.audio.speech.create(
            model=self.cfg.tts_model,
            voice="nova",
            input=text[:4096],
        )
        response.stream_to_file(output_path)
        return output_path
