This is some of the code I wrote for the analysis of object detection models made for detecting debris in the ocean floor as a part of my Bachelor's Project in BSc Artificial Intelligence in University of Groningen. A part of the results can be seen in the DebrisDetectorsRes.png file. The pre-processing method implementations were taken from users 'wangyanckxx' and 'xahidbuffon' (on github). The abstract of my thesis is given below:



Tons of debris end up in the seas and oceans each year, a lot of which sink to the seafloor. This paper is an attempt to make it easier to rid our seafloors of debris and aiding the SeaClear Project. The goal is to reduce the murkiness and improve the colour balance of underwater footage through various image pre-processing methods, namely; CLAHE (Contrast Limited Adaptive Histogram
Equalisation), UCM (Unsupervised Colour Correction
Method), IBLA (Underwater Image Restoration Based on
Image Blurriness and Light Absorption) and funieGAN (Fast Underwater Image Enhancement for Improved Visual Perception), in order to make object detection easier and more consistent. The YOLOv8 model was chosen to be trained with these, as it is shown to be one of the highest performing object detection models both in terms of detection rate and processing speed. The images resulting from each pre-processing method were used to create a corresponding model for that method. These models are assessed based on operational speed (FPS), mAP, accuracy, precision, recall and F1 score, and compared to a model trained with the original (non-pre-processed) images. The combined predictions of all models was also assessed in order to see what the best results are that can be achieved. The UCM and the combination of the models achieved higher overall mAP and F1 scores than the original model, although their processing speeds render them inefficient for real-time use. CLAHE and funieGAN models can be considered if specific objects are being targeted.


