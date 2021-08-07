# Data generation

This data generation code is according to https://github.com/yluo42/TAC/tree/master/data. 

**Changed:** 
- [x] Finish config file generation and use [Wham Noise](http://wham.whisper.ai/).
- [x] Generate direct sound and beam signal, use py-rirgenerator instead of gpurir

## Additional Python packages
- [rir-generator](https://github.com/audiolabs/rir-generator) (beacause it can set rt60=0 to create anechoic sound)

## Run
1. For config creation
```bash
python audio_generation.py --task 0 --output-path "/home/ding/Documents/data/16kHz" --config-path "/home/ding/Documents/config" --libri-path="/home/ding/Documents/LibriSpeech" --noise-path="/home/ding/Documents/LibriSpeech/wham_noise"
```
2. For data generation
```bash
python audio_generation.py --task 1 --output-path "/home/ding/Documents/data/16kHz" --config-path "/home/ding/Documents/config" --libri-path="/home/ding/Documents/LibriSpeech" --noise-path="/home/ding/Documents/LibriSpeech/wham_noise"
```
