# Video Summarization by Visual and Categorical Diversity

We propose a video summarization method based on visual and categorical diversity by transfer learning. Our method extracts visual and categorical features from a pre-trained deep convolutional network (DCN) and a pre-trained word embedding matrix. Using visual and categorical information we obtain video diversity, which it is used as an importance score to select segments from the input video that best describes it. Our method also allows to perform queries during the search process, in this way personalizing the resulting video summaries according to the particular intended purposes. The performance of the method is evaluated using different pre-trained DCN models in order to select the architecture with the best throughput. We then compare it with other state-of-art proposals in video summarization using a data-driven approach with the public dataset SumMe, which contains annotated videos with per-fragment importance. The results show that our method outperforms other proposals in most of the examples. As an additional advantage our method requires a simple and direct implementation that does not require a training stage.

__Files__:
- demo_vsum_diversity.ipynb: Notebook with proposed method

__Folders__:
- Utils: Utility scripts to performs different tasks.
- video_dataset
	- SumMe: SumMe dataset content must be added here.
- pretrained-glove: glove.6B.100d must be added from: https://nlp.stanford.edu/projects/glove/.
