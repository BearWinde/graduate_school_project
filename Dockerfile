FROM python:3.9

WORKDIR /wavesynth
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src /wavesynth


RUN pip install numpy
RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install librosa
RUN pip install crepe
RUN pip install tensorflow
RUN pip install PyYAML
RUN pip install tqdm
RUN pip install nnAudio
RUN pip install tensorboardX

CMD [ "python", "server.py" ]
