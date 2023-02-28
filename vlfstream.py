import os
cmd = 'vtvorbis -E 20 -dn 5.9.106.210,4401 | vtraw -ow > ./tmp/vlfex.wav'
os.system(cmd)
seg1 = 'ffmpeg -ss 0 -t 6 -i ./tmp/vlfex.wav ./tmp/out/out1.wav'
seg2 = 'ffmpeg -ss 4.5 -t 6 -i ./tmp/vlfex.wav ./tmp/out/out2.wav'
seg3 = 'ffmpeg -ss 9 -t 6 -i ./tmp/vlfex.wav ./tmp/out/out3.wav'
seg4 = 'ffmpeg -ss 14 -t 6 -i ./tmp/vlfex.wav ./tmp/out/out4.wav'

os.system(seg1)
os.system(seg2)
os.system(seg3)
os.system(seg4)
