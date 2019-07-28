import numpy as np
from pydub import AudioSegment
from google.cloud import texttospeech
from google.cloud import translate
import os
import sys

speech_speed=0.9


def translate_text(input,language):
    translate_client = translate.Client()
    text_original = input
    translation = translate_client.translate(
        text_original,
        target_language=language)
    text_translated=translation['translatedText'].encode('utf-8')
    return text_translated

def get_response(input,speech_speed,language):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.types.SynthesisInput(text=input)
    voice = texttospeech.types.VoiceSelectionParams(
        language_code=language,
        ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.types.AudioConfig(
        audio_encoding=texttospeech.enums.AudioEncoding.MP3,
        speaking_rate = speech_speed
    )
    response = client.synthesize_speech(synthesis_input, voice, audio_config)
    return response.audio_content

#data
data_to_translate=np.genfromtxt('data.txt', dtype=str, delimiter=';')
size=data_to_translate.shape[0]


#pause in speech
silence_duration = 1000
silence_segment = AudioSegment.silent(duration=silence_duration)  # duration in milliseconds
silence = silence_segment.raw_data


data_translated=np.array(translate_text(data_to_translate[0],'ru'))
for i in range(1,size):
    data_translated=np.append(data_translated,translate_text(data_to_translate[i],'ru'))


speech_to_translate=np.array(get_response(data_to_translate[0],speech_speed,'de-DE'))
speech_translated=np.array(get_response(data_translated[0],speech_speed,'ru-RU'))
for i in range(1, size):
    speech_to_translate = np.append(speech_to_translate,get_response(data_to_translate[i],speech_speed, 'de-DE'))
    speech_translated = np.append(speech_translated,get_response(data_translated[i],speech_speed, 'ru-RU'))



audio_out = AudioSegment(data=str(speech_to_translate[0]),sample_width=1,channels=1,frame_rate=44100)
audio_out = audio_out.append(silence_segment,crossfade=0)
audio_out = AudioSegment(data=str(speech_translated[0]),sample_width=1,channels=1,frame_rate=44100)
audio_out = audio_out.append(silence_segment,crossfade=0)

for i in range(1, data_to_translate.shape[0]):
    fname = 'output' + str(i) + '.mp3'
    temp_segment = AudioSegment(data=str(speech_to_translate[i]),sample_width=1,channels=1,frame_rate=44100)
    audio_out = audio_out.append(temp_segment,crossfade=0)
    audio_out = audio_out.append(silence_segment,crossfade=0)
    temp_segment = AudioSegment(data=str(speech_translated[i]),sample_width=1,channels=1,frame_rate=44100)
    audio_out = audio_out.append(temp_segment,crossfade=0)
    audio_out = audio_out.append(silence_segment,crossfade=0)


audio_out.export("mainout.mp3", format="mp3")

audio_out = AudioSegment(data=speech_to_translate[0],sample_width=1,channels=1,frame_rate=44100)
audio_out.export("mainout1.mp3", format="mp3")

with open('output.mp3', 'wb') as out:
    for i in range(0,size):
        out.write(speech_to_translate[i])
        out.write(speech_translated[i])

for i in range(0, data_to_translate.shape[0]):
    fname = 'output' + str(i) + '.mp3'
    if(os.path.isfile(fname)):
        os.remove(fname)

