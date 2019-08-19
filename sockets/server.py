import socket
import time
import threading
import numpy as np
from google.cloud import texttospeech
from google.cloud import translate
import os


#формат посылок
#tr ru de xxxxx yyyyy
#sp de-DE 0.9 xxxxx yyyyy
# xxxxx - (размер посылки \ _SIZE)
# yyyyy - посылка для обработки

speech_speed=0.8
silence_duration = 0
delimiter='\n'



pair=['de','ru']

dict={'ru':'ru-RU',
      'de':'de-DE'
      }


class SockConnection:

    def __init__(self,_port,_log_fname):
        self.is_opened=False
        self.log_fname=_log_fname
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


    def getTranslate(self,input,originalLanguage,targetLanguage):
        translate_client = translate.Client()
        text_original = input
        original_language = originalLanguage
        target_language = targetLanguage
        translation = translate_client.translate(
            text_original,
            target_language=target_language)
        text_translated = str(translation['translatedText'])  # .encode('utf-8'))
        return text_translated


    def getSpeech(self,input, speech_speed, language):
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

    def writeFile(self,fname,data):
        with open(fname, 'wb') as output:
            output.write(data)

    def parseString(self,data,delimiter):
        prevIndex=0
        decodedData=data#.decode('utf-8')
        delimiter=delimiter.encode()[0]
        arrayCmd=None
        index=0
        for i in range(0,len(decodedData)):
            if(decodedData[i]==delimiter):
                if(index==0):
                    arrayCmd = np.array([decodedData[prevIndex:i].decode()], dtype=str)
                else:
                    arrayCmd = np.append(arrayCmd, decodedData[prevIndex:i].decode())

                prevIndex=i+1
                index=index+1
                if(index==4):
                    break
        data=data[prevIndex:len(data)]
        return arrayCmd,data


    def connThread(self,conn,addr):
        self.writeLog(str(time.ctime()) +" " + str(addr) + " connected\n"  )
        #print("New connection from ",addr)
        _SIZE=32
        try:
            data = conn.recv(_SIZE)
        except:
            pass
        else:
            cmd,data=self.parseString(data," ")

            for i in range(0,int(cmd[3])-1):
                data+=conn.recv(_SIZE)
            data=data.decode()

            self.writeLog(str(time.ctime()) + " received: "+data+"\n")
            self.writeLog(str(time.ctime()) + " cmd: "+cmd[0]+" "+cmd[1]+" "+cmd[2]+" "+cmd[3]+" "+"\n")
            #print('Data is received: ',data)

            language_original=None
            language_target=None
            sound_speed=None
            answer=None

            if(cmd[0]=='tr'):
                language_original=cmd[1]
                language_target=cmd[2]
                self.writeLog(str(time.ctime()) + " translation started"+ "\n")
                answer = self.getTranslate(data, language_original, language_target)
                self.writeLog(str(time.ctime()) + " translation finished"+ "\n")
                answer=answer.encode()
            if(cmd[0]=='sp'):
                language_original=cmd[1]
                sound_speed=float(cmd[2])
                self.writeLog(str(time.ctime()) + " speech generation started"+ "\n")
                answer = self.getSpeech(data, sound_speed, language_original)
                self.writeLog(str(time.ctime()) + " speech generation finished"+ "\n")
                #self.writeFile("speech_s.mp3",answer)

            count = int(0.99 + ((len(answer) + len("00000")) / _SIZE))

            self.writeLog(str(time.ctime()) + str(count) + " sent\n")
            #print("Sending ",count)

            data = str('%.5d' % count).encode()
            data+=" ".encode()
            data+=answer
            conn.send(data)
            #conn.close()


    def mainThread(self):
        while True:
            try:
                conn,addr=self.sock.accept()
            except:
                pass
            else:
                tt = threading.Thread(target=self.connThread, args=(conn, addr))
                tt.daemon = True
                tt.start()

            continue

z=SockConnection(9072,"log.txt")