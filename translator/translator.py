import numpy as np
from pydub import AudioSegment
from google.cloud import texttospeech
import os

silence_duration = 1000
silence_segment = AudioSegment.silent(duration=silence_duration)  # duration in milliseconds
silence = silence_segment.raw_data

data_to_translate = np.genfromtxt('data.txt', dtype=str, delimiter=';')

client = texttospeech.TextToSpeechClient()

def get_response(input,speed,language):
    synthesis_input = texttospeech.types.SynthesisInput(text=input)
    voice = texttospeech.types.VoiceSelectionParams(
        language_code=language,
        ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.types.AudioConfig(
        audio_encoding=texttospeech.enums.AudioEncoding.MP3,
        speaking_rate = speed
    )
    response = client.synthesize_speech(synthesis_input, voice, audio_config)
    return response


for i in range(0, data_to_translate.shape[0]):
    response=get_response(data_to_translate[i],0.8,'de-DE')
    out = open('output' + str(i) + '.mp3', 'wb')
    try:
        responses = np.append(responses,response.audio_content)
    except:
        responses = np.ndarray(shape=0,dtype='str')
        responses = np.append(responses,response.audio_content)

    out.write(response.audio_content)
    out.close()


out = open('1.mp3', 'wb')
for i in range(0, data_to_translate.shape[0]):
    out.write(responses[i])

# for i in range(0, data_to_translate.shape[0]):
#     fname = 'output' + str(i) + '.mp3'
#     temp_segment = AudioSegment.from_mp3(fname)
#     #temp_segment = AudioSegment.from_raw(str(responses[i]),sample_width=2,channels=2,frame_rate=44100)
#     try:
#         audio_out = audio_out.append(temp_segment)
#         audio_out = audio_out.append(silence_segment)
#     except:
#         audio_out = temp_segment
#         audio_out = audio_out.append(silence_segment)
#
#
#audio_out.export("mainout.mp3", format="mp3")

silence_segment.export("silence.mp3", format="mp3")
temp_segment = AudioSegment.from_mp3("silence.mp3")
rdata=str(temp_segment._data)

s = open('silence.mp3', 'rb').read()

for i in range(0, data_to_translate.shape[0]):
    fname = 'output' + str(i) + '.mp3'
    if(os.path.isfile(fname)):
        os.remove(fname)

print('Audio content written to file "output.mp3"')
