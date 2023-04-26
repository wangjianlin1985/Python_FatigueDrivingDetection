from pygame import mixer
import time

a=mixer.init()
mixer.music.load('d:/pljs.mp3')
mixer.music.play(3,0.0)
time.sleep(30)
#mixer.music.stop()

