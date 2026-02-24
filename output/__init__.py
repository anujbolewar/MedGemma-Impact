"""
output — Text-to-speech synthesis and emergency override broadcasting.

All patient-facing audio output is produced here. pyttsx3 is used for
offline TTS; emergency messages bypass the LLM entirely.
"""
