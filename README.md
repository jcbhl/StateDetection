# StateDetection
Using Tensorflow and OpenBCI EEG data to infer the user's state in realtime.

# Tutorial
Tutorial on how to build your own should be coming soon!

# Demo
See [here](https://drive.google.com/file/d/1OPYp-l1kXYGiru-VdRlFliqr6hPodywm/view?usp=sharing) for a demo of the program. For the first half of the video, I'm playing a video game on another computer, with the black box representing the model's prediction of my state. What's interesting here is that when the model switches over to predicting "reading", my character was dead in the game, signifying that my brain activity was substantially different while actively playing versus waiting to respawn! After about halfway through the video, I stop playing the video game and begin reading, and the model switches over to that state quite quickly. I didn't want to demo the meditation feature as it typically took me around 20-30 seconds to start producing enough delta waves for the model to recognize, which would've been a little boring for the viewer to sit through.

# Technical details
This project has a similar architecture to the left vs right project that I did, but now we have a multiclass classification problem instead of a binary one. Additionally, I made the decision to use vectors of shape (20,60) in the model input, so we take 5 measurements for each channel, 4 since I'm using a Ganglion. Because we have more data to work with, I can use a proper 2-layer convolution setup, which results in better feature extraction. Compared to the right vs left project, this model had much less data to work with (only 5 min of data for each category) and achieved substantially better results on both the training set and in real practice, likely because of this architectural difference. Also, there's probably a bigger difference in the FFT profiles of these 3 activities compared to the difference between left thinking vs right thinking. For the visualizer, I use a simple aggregate of the last 2 seconds of predictions as a method of noise reduction.

# Credits
Thanks to Prof. Williams for the hardware and getting me started with openBCI! 
