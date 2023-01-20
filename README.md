# AAIT Homework 2

### System requirements 

* Python 3.8
* CUDA 11.6
* PyTorch 1.13

### Install notes
0. Clone repository
`git clone git@github.com:petremihaivalentin/aait-hw2.git`
1. Get dataset from [here](https://download.wetransfer.com/eugv/c38ba5efc5e19f51bac8c921e53ffe9120230120195841/0a9c2e5627a182c2a933d55296c626b4a606b275/hw2.zip?token=eyJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE2NzQyNDQ4MTMsImV4cCI6MTY3NDI0NTQxMywidW5pcXVlIjoiYzM4YmE1ZWZjNWUxOWY1MWJhYzhjOTIxZTUzZmZlOTEyMDIzMDEyMDE5NTg0MSIsImZpbGVuYW1lIjoiaHcyLnppcCIsIndheWJpbGxfdXJsIjoiaHR0cDovL3N0b3JtLWludGVybmFsLnNlcnZpY2UuZXUtd2VzdC0xLndldHJhbnNmZXIubmV0L2FwaS93YXliaWxscz9zaWduZWRfd2F5YmlsbF9pZD1leUpmY21GcGJITWlPbnNpYldWemMyRm5aU0k2SWtKQmFITkxkMmxRZUhaYVNFRm5RVDBpTENKbGVIQWlPaUl5TURJekxUQXhMVEl3VkRJd09qRXdPakV6TGpBd01Gb2lMQ0p3ZFhJaU9pSjNZWGxpYVd4c1gybGtJbjE5LS0yZmM0MmE3OTczNTAxMDlkOGUzNTFkYTY5ODA1OWM4NTcxMzA0Mjg3M2I1ZGY2NGZlZmYwMjcyOTI2NzAxZjY5IiwiZmluZ2VycHJpbnQiOiIwYTljMmU1NjI3YTE4MmMyYTkzM2Q1NTI5NmM2MjZiNGE2MDZiMjc1IiwiY2FsbGJhY2siOiJ7XCJmb3JtZGF0YVwiOntcImFjdGlvblwiOlwiaHR0cDovL2Zyb250ZW5kLnNlcnZpY2UuZXUtd2VzdC0xLndldHJhbnNmZXIubmV0L3dlYmhvb2tzL2JhY2tlbmRcIn0sXCJmb3JtXCI6e1widHJhbnNmZXJfaWRcIjpcImMzOGJhNWVmYzVlMTlmNTFiYWM4YzkyMWU1M2ZmZTkxMjAyMzAxMjAxOTU4NDFcIixcImRvd25sb2FkX2lkXCI6MTc2NjE3ODU3OTh9fSJ9.mxliwxPzBJfLqN9wY9AK4c2z4KFaywl2honFwyqcj_k&cf=y) and weights from [here]()

2. Place dataset in './' and weights in './weights/'

3. Install python modules
`python -m pip install requirements.txt`

### Usage

Simply use `python driver.py --[model] --[weights] --[datadir]` to get an evaluation on the test dataset for Task 1

Available models:

| Model argument | Description |
| ----------- | ----------- |
| resnet50 | [ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) |
| mobilenet_v2 | [MobileNetV2](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html#torchvision.models.mobilenet_v2) | 

Available weights:
| Weights argument | Description | % Acc on Leaderboard |
| ----------- | ----------- | ----------- |
| resnet50_blank | Best ResNet50 with no additional funky stuff | 43.32 |
| resnet50_noisy | Best ResNet50 with some additional training on unlabeled | 13.20 |
| resnet50_shuffle | Best ResNet50 trained with [ShuffleFold]() | **43.40** |
| mobilenet_v2_blank | Best MobileNetV2 with no additional funky stuff | 31.10 |

### Notes

#### Task 1

I spent almost a week getting < 1% accuracy on Task 1. I found out the problem was that I wasn't normalizing the data with the correct values (I found some on some GitHub Repo) and I also found out that applying an Affine transformation (thanks to the same GitHub Repo) would make things much better.

Then I went on to try different models. I tried all ResNets from torch, but I found ResNet50 to be the best when it comes to training time/ performance ratio. ResNet101 performed well, but was training really slow.

I also tried MobileNet V2 and trained really fast but didn't perform very well.

Finally, as I scrolled through the ML Leaderboard, I saw someone used KFold. So I thought "of course, I totally forgot about folding. I might as well try as I only have 4 hours until the deadline". So I tried ShuffleFold on the ResNet50 training and it improved the result.

#### Task 2

I tried an idea:
1. PCA on all labeled data
2. KMeans clustering on PCA'd data. Get 100 clusters.
3. Find dominant label in every cluster.
4. Assign dominant label to samples, instead of original label

This failed miserably as there was virtually no dominant label in the clusters. I have attached the code for this try.