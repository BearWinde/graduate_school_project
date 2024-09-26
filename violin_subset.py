import os

dict_train = os.listdir("D:/wenzo/wave-synth/violin/violin_100/audio")
dict_valid = os.listdir("D:/wenzo/wave-synth/violin/valid/audio")
dict_test = os.listdir("D:/wenzo/wave-synth/violin_test/voice_8/audio")

with open("D:/wenzo/wave-synth/violin_subset/keys_train.txt", "w+") as f:
    for key in dict_train:
        f.write(key + '/n')
with open("D:/wenzo/wave-synth/violin_subset/keys_valid.txt", "w+") as f:
    for key in dict_valid:
        f.write(key + '/n')
with open("D:/wenzo/wave-synth/violin_subset/keys_test.txt", "w+") as f:
    for key in dict_test:
        f.write(key + '\n')