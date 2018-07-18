# Semantic_VSUM

This project uses deep learning to evaluate different deep learning architectures for the task of Video Summarization using SumMe dataset proposed by Gigly et. al. 2014. Keras is the main framework used for this project.

Files:
	Deep_Description.ipynb: Python notebook that computes intermediate activations and classification labels for known archictectures of DCN (VGG16, XCeption, densenet, etc.). Also, computes the GloVe the mean activation for each video frame, with respect to the labels detected by DCN architectures.
	S-VSUM.ipynb: Python notebook that uses DCN and GloVe activations for create a video summary of an input video from SumMe dataset. Then, calculate f-score with respect to human score using code provided by Gygli in SumMe dataset.

Folders:
	Utils: 
	video_dataset: SumMe dataset as provided by Gygli et. al. 2014.
	pretrained-glove: glove.6B.100d embedding matrix.
	DCN_outputs: frame-by-frame intermediate and labels activations for each DCN architecture evaluated.
	DLM_outputs: frame-by-frame GloVe activation using labels detected by DCN architectures.

