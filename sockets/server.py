import socket
import time
import threading
import numpy as np
from pydub import AudioSegment
from google.cloud import texttospeech
from google.cloud import translate
import os


speech_speed=0.8
silence_duration = 0
delimiter='\n'



pair=['de','ru']

dict={'ru':'ru-RU',
      'de':'de-DE'
      }


class SockConnection:

    def __init__(self,_port,_log_fname,_code):
        self.is_opened=False
        self.log_fname=_log_fname
        self.code=_code
        self.port=_port
        try:
            self.sock = socket.socket()
        except:
            pass
        else:
            self.is_opened=True
        if(self.is_opened==True):
            try:
                self.sock.bind(('', self.port))
            except:
                pass
            try:
                self.sock.listen(1)
            except:
                pass
            self.mainThread()

    def translate_text(self,input):
        translate_client = translate.Client()
        text_original = input
        original_language = translate_client.detect_language(text_original)['language']
        # if(original_language!=pair[0]):
        #    target_language=pair[0]
        # else:
        #    target_language=pair[1]
        original_language = pair[0]
        target_language = pair[1]
        translation = translate_client.translate(
            text_original,
            target_language=target_language)
        text_translated = str(translation['translatedText'])  # .encode('utf-8'))
        return text_translated, original_language, target_language

    def get_speech(self,input, speech_speed, language):
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.types.SynthesisInput(text=input)
        voice = texttospeech.types.VoiceSelectionParams(
            language_code=language,
            ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.types.AudioConfig(
            audio_encoding=texttospeech.enums.AudioEncoding.MP3,
            speaking_rate=speech_speed
        )
        if (language != -1):
            response = client.synthesize_speech(synthesis_input, voice, audio_config)
            return response.audio_content
        else:
            return 0

    def compare_with_dict(self,input):
        for i in dict.keys():
            if (i == input):
                return dict[input]
        return -1


    def writeLog(self,data):
        file = open(self.log_fname, 'a+')
        file.write(data)
        file.close()

    # def connThread(self,conn,addr):
    #     _SIZE=32
    #     try:
    #         data = conn.recv(_SIZE)
    #     except:
    #         pass
    #     else:
    #         size=sys.getsizeof(data)
    #         pwd = data[0:6].decode()
    #         data = data[6:len(data)].decode()
    #         data_available=True
    #         while(data_available==True):
    #                 new_data=conn.recv(_SIZE)
    #                 data += new_data.decode()
    #                 if(len(new_data)<_SIZE):
    #                     data_available=False
    #         if (pwd == self.code):
    #             self.writeLog(" " + data)
    #             ans = str(addr) + str(data)
    #             ans = ans.encode()
    #             conn.send(ans.upper())
    #             conn.close()
    def connThread(self,conn,addr):
        _SIZE=32
        try:
            data = conn.recv(_SIZE)
        except:
            pass
        else:
            cmd=data[0:2].decode()
            language_original=None
            language_target=None
            mes_count=None
            sound_speed=None
            if(cmd=='tr'):
                language_original=data[3:5].decode()
                language_target=data[6:8].decode()
                mes_count=int(data[9:10].decode())
            if(cmd=='sp'):
                language_original=data[3:8].decode()
                sound_speed=float(data[9:13].decode())
                mes_count=int(data[14:15].decode())

            message=data[15:len(data)]
            for i in range(0,mes_count-1):
                message+=conn.recv(_SIZE)
            message=message.decode('utf-8')
            if(cmd=='tr'):
                pass
            if(cmd=='sp'):
                pass



    def mainThread(self):
        while True:
            try:
                conn,addr=self.sock.accept()
            except:
                pass
            else:
                self.writeLog("\n"+str(time.ctime())+" "+str(addr))
                print('connected:', addr)
                tt = threading.Thread(target=self.connThread, args=(conn, addr))
                tt.daemon = True
                tt.start()

            continue

z=SockConnection(9071,"log.txt","qwerty")