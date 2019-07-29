import numpy as np
from pydub import AudioSegment
from google.cloud import texttospeech
from google.cloud import translate
import os
import sys

speech_speed=0.9
silence_duration = 500
delimiter='\n'



pair=['ru','de']

dict={'ru':'ru-RU',
      'de':'de-DE'
      }


def translate_text(input):
    translate_client = translate.Client()
    text_original = input
    original_language = translate_client.detect_language(text_original)['language']
    #if(original_language!=pair[0]):
    #    target_language=pair[0]
    #else:
    #    target_language=pair[1]
    original_language=pair[0]
    target_language=pair[1]
    translation = translate_client.translate(
        text_original,
        target_language=target_language)
    text_translated=str(translation['translatedText'])#.encode('utf-8'))
    return text_translated,original_language,target_language

def get_speech(input, speech_speed, language):
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
    if(language!=-1):
        response = client.synthesize_speech(synthesis_input, voice, audio_config)
        return response.audio_content
    else:
        return 0

def compare_with_dict(input):
    for i in dict.keys():
        if(i==input):
            return dict[input]
    return -1

#data
fname='data.txt'
data_to_translate=np.loadtxt(fname,delimiter=delimiter,dtype=str,encoding='utf-8')
size=data_to_translate.shape[0]


#pause in speech
silence_segment = AudioSegment.silent(duration=silence_duration,frame_rate=44100)  # duration in milliseconds



[text,original_language,target_language]=translate_text(data_to_translate[0])
data_translated=np.array(text)
speech_to_translate=np.array(get_speech(data_to_translate[0], speech_speed,
                                        dict[original_language]))
speech_translated=np.array(get_speech(text, speech_speed, dict[target_language]))

for i in range(1,size):
    [text, original_language, target_language] = translate_text(data_to_translate[i])
    data_translated=np.append(data_translated,text)
    speech_to_translate = np.append(speech_to_translate, get_speech(data_to_translate[i], speech_speed, dict[original_language]))
    speech_translated = np.append(speech_translated, get_speech(text, speech_speed, dict[target_language]))




if(silence_duration>0):

    for i in range(0,size):
        fname1='out'+str(i*2)+'.mp3'
        fname2='out'+str(i*2+1)+'.mp3'
        with open(fname1, 'wb') as output:
            output.write(speech_to_translate[i])
        with open(fname2, 'wb') as output:
            output.write(speech_translated[i])

    audio_out=AudioSegment.empty()
    for i in range(0,size):
        fname1='out'+str(i*2)+'.mp3'
        fname2='out'+str(i*2+1)+'.mp3'
        audio_out=audio_out+AudioSegment.from_mp3(fname1)
        audio_out=audio_out+AudioSegment.from_mp3(fname2)
        audio_out=audio_out+silence_segment


    audio_out.export("output.mp3",format="mp3")


    for i in range(0, size):
        fname1='out'+str(i*2)+'.mp3'
        fname2='out'+str(i*2+1)+'.mp3'
        if(os.path.isfile(fname1)):
            os.remove(fname1)
        if(os.path.isfile(fname2)):
            os.remove(fname2)
else:
    fname = 'output.mp3'
    with open(fname, 'wb') as output:
        for i in range(0,size):
            output.write(speech_to_translate[i])
            output.write(speech_translated[i])

