# Model

## Description:
The beat tracking model is derived from: <to be added (not in the system yet)>. It is a simple TCN network (inspired by the newest madmom implementation) trained
on 22050 Hz recordings and 50 fps (instead of the original 44100 Hz and 100 fps). It is faster, and you do not have to resample the audio recordings 
or the activation function to use it in the synctoolbox pipeline for enhanced synchronization.

