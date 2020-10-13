#SES DOSYASI TANIMA PROGRAMI

import speech_recognition as sr
import time
import os
r = sr.Recognizer()

file = sr.AudioFile('.wav')

with file as source:
    r.adjust_for_ambient_noise(source)
    audio = r.record(source)
    result = r.recognize_google(audio,language='tr')
print(result)
