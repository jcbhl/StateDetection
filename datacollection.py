from pylsl import StreamInlet, resolve_stream
from playsound import playsound

import time
import numpy as np

STATES = ['TBD','TBD','TBD','TBD']

def main():
    print('Trying to connect to LSL stream...')
    streams = resolve_stream('type','EEG')
    inlet = StreamInlet(streams[0])
    print('LSL connection established.')
    print("Input the activity for data to be collected on: " )
    activity = str(input())

    time_to_collect = 300
    samples_to_collect = time_to_collect * 5
    # Pulling 4 channels of FFT data from the board occurs at 25 Hz. I'm gonna start off with trying to pull 5 full measurements, so 20 total individual calls to pull_sample.
    # This gets us down to 5 vectors created per second.
    # We should end up with a vector of shape (20, 60) assuming I cut it off at 60 Hz again.

    print('Beginning data collection in 2 seconds.')
    time.sleep(10)
    samples_collected = 0
    sampleLst = []
    while samples_collected < samples_to_collect:
        chan_list = []
        for i in range(20):
            sample,timestamp = inlet.pull_sample()
            chan_list.append(sample[:60]) # Cut off data at 60 Hz
        sampleLst.append(chan_list)
        samples_collected+=1
    playsound('assets/eventually.mp3')

    sampleLst = np.asarray(sampleLst)
    print('Shape of dataset is ' + str(sampleLst.shape))
    print('Saving data...')
    np.save('data/' + activity + '-' + str(time.time()) + '.npy',sampleLst)
    print('Data saved.')


if __name__ == '__main__':
    main()