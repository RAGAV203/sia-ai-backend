"""Speech to text: Gemini first, local Whisper as the safety net.

Three problems had to be solved together, because fixing any one alone does not
move the transcript quality much.

**1. The browser records a format Gemini will not accept.** ``MediaRecorder``
produces WebM/Opus (or MP4/AAC on Safari); the Gemini audio API takes WAV, MP3,
AIFF, AAC, OGG and FLAC. So every clip is decoded here first. PyAV does it in
memory — it already ships as a faster-whisper dependency, so this costs no new
package and no ffmpeg binary on the host.

**2. Phone-quality speech is quiet.** Laptop microphones at arm's length arrive
several dB below full scale, and both engines transcribe a normalized clip more
accurately than a faint one. Decoding to a float array means peak normalization
is two lines, and it applies to whichever engine runs.

**3. The vocabulary is the hard part, not the language.** The words this service
must get right are almost all proper nouns a general model has never seen:
"Shasun", "Shri Shankarlal Sundarbai", "Madley Road", "SAYE", "B.Com Corporate
Secretaryship". Whisper alone renders these as "Shashan"/"session", and a
misheard programme name retrieves the wrong page — an STT error becomes a
retrieval error becomes a wrong answer. Gemini takes the glossary as part of the
prompt, so the terms are biased in rather than corrected after the fact.

Whisper stays as the fallback and is still worth its weight: it keeps the mic
working when the API key is missing, the quota is exhausted, or the network is
down, and it does so without changing anything the caller sees.
"""

from __future__ import annotations

import io
import os
import re
import threading
import time
import unicodedata

import numpy as np

import languages
from kb import debug, gemini

# The words a general-purpose recognizer has no reason to know, fed to Gemini as
# a bias list. Keep it to terms that are actually *said* out loud and actually
# get mangled — a long list dilutes the signal and costs prompt tokens on every
# transcription.
GLOSSARY = (
    "Shasun, Shri Shankarlal Sundarbai Shasun Jain College for Women, Chennai, "
    "T. Nagar, Madley Road, NAAC, IQAC, NCC, NSS, SAYE, B.Com, B.Sc, B.B.A, M.A., "
    "M.Com, M.Sc, Corporate Secretaryship, Accounting and Finance, Honours, "
    "Visual Communication, Computer Science, Psychology, Journalism, "
    "University of Madras, Elementor, placement cell, autonomous, semester, "
    "eligibility, admission, undergraduate, postgraduate, alumnae"
)

TRANSCRIBE_PROMPT = (
    "Transcribe the speech in this audio exactly, as a single line of plain text.\n"
    "Context: the speaker is asking a question about an Indian women's college in "
    "Chennai, often in Indian-accented English.\n"
    f"Expect these proper nouns and spell them this way: {GLOSSARY}.\n"
    "Rules: output ONLY the transcript. No quotes, no speaker labels, no commentary, "
    "no translation. If there is no intelligible speech, output exactly: (no speech)"
)


def _prompt_for(language) -> str:
    """The transcription prompt for a given spoken language.

    Three instructions carry the weight for the non-English languages, and all
    three were measured failure modes rather than precautions:

    * **Show the script, do not merely name it.** This is the big one. Told
      "write the transcript in the native script of Tamil", Gemini returns
      romanized Tamil — measured **0%** of letters in Tamil script, on every
      clip tried. Given a concrete example of the target script and told plainly
      that romanization is wrong, the same model on the same audio returns
      **100%**. Naming the script is not an instruction a model can act on;
      seeing one is.

      This matters beyond looking right. A romanized transcript is echoed into
      the chat as the visitor's own message, where it reads as broken, and it is
      handed to a translator that has been told to expect Tamil.

    * **Forbid translating.** A model given Tamil audio and an English prompt
      tends to answer in the prompt's language rather than the audio's, silently
      turning transcription into translation — and the pipeline downstream would
      then translate it *again*.

    * **Keep the English glossary.** Code-switching is the norm here: a Tamil
      speaker asking about admissions still says "B.Com" and "placement cell" in
      English, and those are exactly the tokens that get mangled without a bias
      list.
    """
    if language.code == "en":
        return TRANSCRIBE_PROMPT

    name = language.english_name
    return (
        f"Transcribe this {name} audio into {name} text.\n\n"
        f"SCRIPT: The output MUST be written in {name}'s own script, like this: "
        f"{language.script_sample}\n"
        f"Romanized {name} written in the Latin alphabet is WRONG and unusable. "
        f"Never write {name} words using English letters.\n\n"
        f"The speaker may use English loanwords for academic terms (admission, "
        f"placement, B.Com). Write those in {name} script too, spelled as they are "
        f"pronounced, EXCEPT for the college's own name which stays as it sounds.\n"
        f"Context: a question about an Indian women's college in Chennai. "
        f"Proper nouns you may hear: {GLOSSARY}.\n\n"
        f"Output ONLY the transcript. Do not translate to English. No quotes, no "
        f"commentary. If there is no intelligible speech, output exactly: (no speech)"
    )


def native_script_ratio(text: str, script: str) -> float:
    """Fraction of letters written in `script`. 1.0 means fully native.

    Used only as a diagnostic. The prompt above reliably produces native script,
    but "reliably" is a measurement of today's model, and a future revision
    drifting back to romanized output would otherwise be invisible from the
    server side — the transcript would still be a plausible sentence, still
    answerable, and still wrong in the chat bubble.
    """
    prefix = {"tamil": "TAMIL", "devanagari": "DEVANAGARI"}.get(script)
    if not prefix:
        return 1.0  # Latin-script language: nothing to check
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    native = sum(1 for c in letters if unicodedata.name(c, "").startswith(prefix))
    return native / len(letters)

STT_BACKEND = os.getenv("STT_BACKEND", "gemini").strip().lower()

# Whether to keep the local Whisper fallback available at all.
#
# It is the single largest optional item in the process — measured resident:
# 155 MB for `base`, 111 MB for `tiny` — on top of a 323 MB steady state. On a
# 512 MB instance that is the difference between 188 MB of headroom and 34 MB,
# and 34 MB is not enough to survive a traffic spike without being OOM-killed.
#
# Note this does NOT disable audio decoding: PyAV is on the *primary* path,
# because Gemini will not accept the WebM/Opus the browser records. Only the
# transcription model is dropped, so a Gemini outage surfaces as an STT error
# the frontend degrades from, rather than as a dead container.
STT_FALLBACK = os.getenv("STT_FALLBACK", "1") not in ("0", "false", "False", "")

SAMPLE_RATE = 16000

# Below this the clip is silence, not speech. Peak normalization would otherwise
# amplify room tone into something both engines hallucinate words from.
SILENT_PEAK = 1e-3
TARGET_PEAK = 0.95
MAX_AUDIO_SECONDS = float(os.getenv("STT_MAX_SECONDS", "60"))

# Whisper config (fallback path).
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
WHISPER_LANG = os.getenv("WHISPER_LANG", "en")

# Whisper accepts a text prompt as decoder context, which biases it toward this
# vocabulary the same way the Gemini prompt does — the one lever it offers for
# proper nouns.
WHISPER_PROMPT = f"A question about {GLOSSARY}."

_whisper = None
# Created eagerly. Building the lock lazily would have been its own race: two
# threads can both observe it as None and end up guarding with different locks,
# which is exactly the case the lock exists to prevent — two copies of the model
# loaded at once on a host chosen for having little spare memory.
_whisper_lock = threading.Lock()


def get_whisper():
    global _whisper
    if _whisper is None:
        with _whisper_lock:
            if _whisper is None:
                from faster_whisper import WhisperModel

                _whisper = WhisperModel(
                    WHISPER_MODEL,
                    device="cpu",
                    compute_type=WHISPER_COMPUTE,
                    cpu_threads=int(os.getenv("CPU_THREADS", "4")),
                )
                print(f"[stt] faster-whisper '{WHISPER_MODEL}' ({WHISPER_COMPUTE})")
    return _whisper


# --- audio preparation --------------------------------------------------------

def decode(raw: bytes) -> np.ndarray:
    """Decode any browser-recorded container to 16 kHz mono float32.

    PyAV reads WebM/Opus, MP4/AAC, OGG and WAV, which covers every format
    ``MediaRecorder`` emits across browsers, so the client never has to negotiate
    one that both ends support.
    """
    from faster_whisper.audio import decode_audio

    return decode_audio(io.BytesIO(raw), sampling_rate=SAMPLE_RATE)


def normalize(samples: np.ndarray) -> tuple[np.ndarray, float]:
    """Peak-normalize, unless the clip is silent. Returns (audio, original peak)."""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak < SILENT_PEAK:
        return samples, peak
    return (samples * (TARGET_PEAK / peak)).astype(np.float32), peak


def to_wav(samples: np.ndarray) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# --- transcript cleanup -------------------------------------------------------

# An instructed model still occasionally frames its answer. These are stripped
# rather than treated as a failure, because the transcript itself is usually
# correct — it is only wearing a preamble.
_PREAMBLE = re.compile(
    r"^\s*(here is|here's|the )?\s*(the\s+)?"
    r"(transcript|transcription|audio says|speaker says|text)\s*(is)?\s*[:\-]\s*",
    re.I,
)
_NO_SPEECH = re.compile(r"^\(?\s*no speech\s*\)?\.?$", re.I)


def clean_transcript(text: str) -> str:
    text = " ".join((text or "").split())
    text = _PREAMBLE.sub("", text)
    text = text.strip().strip('"').strip("'").strip()
    if not text or _NO_SPEECH.match(text):
        return ""
    return text


# --- engines ------------------------------------------------------------------

def _via_gemini(wav: bytes, language) -> str:
    return clean_transcript(
        gemini.transcribe(wav, "audio/wav", prompt=_prompt_for(language))
    )


def _via_whisper(samples: np.ndarray, language) -> str:
    segments, info = get_whisper().transcribe(
        samples,
        # The registry's code, not WHISPER_LANG. The env var stays as the
        # deployment-wide default for a request that names no language, but a
        # request that does must win — Whisper transcribing Tamil audio under
        # `language="en"` does not fail, it invents plausible English.
        language=language.whisper,
        beam_size=5,          # 1 was chosen when this was the only engine and
                              # latency mattered most; as the fallback it should
                              # favour accuracy, and 5 is the usual sweet spot.
        vad_filter=True,      # drop silence -> fewer hallucinated words
        condition_on_previous_text=False,  # short clips: no context to carry
        initial_prompt=WHISPER_PROMPT,
        temperature=0.0,
    )
    parts = list(segments)
    if debug.ENABLED:
        debug.log(
            f"  whisper lang={getattr(info, 'language', '?')} "
            f"({getattr(info, 'language_probability', 0):.2f})"
        )
        for i, seg in enumerate(parts, 1):
            debug.log(f"  seg {i} [{seg.start:5.2f}-{seg.end:5.2f}s] {seg.text.strip()!r}")
    return clean_transcript(" ".join(seg.text.strip() for seg in parts))


# --- public API ---------------------------------------------------------------

def transcribe(raw: bytes, filename: str = "", lang: str = "") -> dict:
    """Transcribe a recorded clip. Blocking — always call off the event loop.

    Returns ``{"text", "engine", "duration_s", "lang"}``. ``text`` is empty when
    the clip carried no speech, which the caller reports differently from a
    failure. ``lang`` echoes the language actually used, so a client can tell a
    fallback from a mis-set picker.
    """
    language = languages.get(lang or WHISPER_LANG)
    started = time.perf_counter()
    debug.section(f"STT [{language.code}]")

    try:
        samples = decode(raw)
    except Exception as exc:  # noqa: BLE001 - a corrupt/empty upload
        raise ValueError(f"could not decode audio ({type(exc).__name__})") from exc

    duration = len(samples) / SAMPLE_RATE if samples.size else 0.0
    if duration > MAX_AUDIO_SECONDS:
        samples = samples[: int(MAX_AUDIO_SECONDS * SAMPLE_RATE)]
        duration = MAX_AUDIO_SECONDS

    samples, peak = normalize(samples)
    debug.log(
        f"upload {len(raw) / 1024:.0f} KB ({filename or '?'}) -> "
        f"{duration:.1f}s audio, peak {peak:.4f}"
    )

    # A clip that never left the noise floor has nothing to transcribe, and
    # sending it only buys a hallucinated sentence — both engines will invent
    # words from room tone given the chance.
    if peak < SILENT_PEAK:
        debug.log("silent clip — not transcribing")
        return {
            "text": "",
            "engine": "none",
            "duration_s": round(duration, 2),
            "lang": language.code,
        }

    engine = "whisper"
    text = ""
    if STT_BACKEND == "gemini" and gemini.available():
        try:
            text = _via_gemini(to_wav(samples), language)
            engine = "gemini"
        except Exception as exc:  # noqa: BLE001 - fall through to the local model
            if not STT_FALLBACK:
                raise
            print(f"[stt] gemini failed ({type(exc).__name__}: {exc}); using whisper")
            debug.log(f"gemini failed: {type(exc).__name__} — falling back")

    if engine != "gemini":
        if not STT_FALLBACK:
            raise RuntimeError("speech-to-text unavailable (STT_FALLBACK=0 and Gemini unreachable)")
        text = _via_whisper(samples, language)

    elapsed = (time.perf_counter() - started) * 1000
    debug.log(f"engine={engine} lang={language.code} in {elapsed:.0f} ms")

    # A transcript that came back romanized is still a usable sentence, so it
    # cannot be detected downstream — it is only wrong on screen and in the
    # translator's assumptions. Surface it here rather than nowhere.
    ratio = native_script_ratio(text, language.script)
    if text and ratio < 0.5:
        print(
            f"[stt] {language.code}: transcript is {(1 - ratio) * 100:.0f}% Latin — "
            f"the model romanized instead of using {language.script} script"
        )
        debug.log(f"native-script ratio {ratio:.2f} (expected ~1.0)")

    debug.block("TRANSCRIPT", text or "(empty)")
    return {
        "text": text,
        "engine": engine,
        "duration_s": round(duration, 2),
        "lang": language.code,
    }


def status() -> dict:
    return {
        "backend": STT_BACKEND,
        "gemini_ready": STT_BACKEND == "gemini" and gemini.available(),
        "fallback": WHISPER_MODEL if STT_FALLBACK else None,
        "fallback_loaded": _whisper is not None,
        "languages": languages.codes(),
    }
