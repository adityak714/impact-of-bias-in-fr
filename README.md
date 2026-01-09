# Impact of Bias in Face Recognition

<p align='center'>
<img width="42%" height="auto" alt="Component 7(1)" style="margin: auto;" src="https://github.com/user-attachments/assets/c142b2f8-b2e8-46a2-8b55-c1a0956219ca" /> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
<img width="35%" height="auto" alt="Screenshot 2025-09-15 at 17 16 17 1(1)" style="margin: auto;" src="https://github.com/user-attachments/assets/04c66246-d11c-4fab-a068-35c0cd7e68da" />
<p/>

> Credit: Combined images from some of currently used facial recognition toolkits like [DeepFace](https://github.com/serengil/deepface).

### Contributors:
- Aditya Khadkikar, M.Sc. Data Science student, Uppsala University
- Mark Smithson Rivas, M.Sc. Data Science student, Uppsala University

### Supervisor:
- Anders Hast, Professor in Department of Information Technology, Uppsala University

<hr/>

## **Aim of Research** 
Nowadays, different facial detection and recognition models are being used, but their robustness to subject faces of different races, and ethnicities remains a continued area of research. Additionally, there are several factors in the captured frame of the subjects, such as poor lighting, presence of accessories (glasses, headwear), or the head tilt/rotation, that can affect classification, or proper detection of facial features. 

In this project, the work has been on evaluating the performance of different facial detection backends for race, and gender classification on the Labelled Faces in the Wild (LFW) dataset. More specifically, I take a subset from the LFW dataset, given the name as the 'GT20' subset, which contains images of individuals that have at least 20 images to their name. 62 different individuals are filtered out from the original dataset, and 20 images were taken for consistency, having 1,240 images total. We use the Deepface toolkit for running the experimentation for this project. We try different facial detector backends; `dlib, RetinaFace, MTCNN`. The recognizer model, or the model converting the images to a high-dimensional embedding format, and for optionally verifying to another image, is kept to `Facenet512`. 

There are 7 parts outlining the whole conducted process:

## 1. Estimate the Headpose of the Image Analysed
- For the project, the headpose was calculated using facial landmarks that the detector observes on a given subject's image. Important landmarks are right-eye, left-eye, right- and left-boundaries of the mouth, and the nose (5 in total, x2 coordinates in x and y). RetinaFace and MTCNN have a defined output schema, giving those landmarks, as well as bounding boxes, which can be drawn to show. 

<p align='center'>
<img width="679" height="942" alt="Component 2" src="https://github.com/user-attachments/assets/60270f60-a9d7-4a19-b82a-7a27676248d4" />
</p>

- `dlib` estimates the headpose and facial landmarks differently than `retinaface` and `mtcnn`. It marks 68 landmark points, highlighting not only the 5 landmarks we look for (lips, eyes and nose), but also eyebrows and the jawline. However, we do the headpose calculation by picking out the index of the nose midpoint, each eye's midpoint etc. instead of all 68 points. In estimating the headpose, we observe 3 attributes; the **roll**, **yaw** and **pitch**.

<p align='center'>
<img width="400" height="auto" alt="face" src="https://github.com/user-attachments/assets/ed8ae939-468b-4ffc-8e6c-70614a16ac71" />
</p>

> Credit: M. Luzardo, M. Karppa, J. Laaksonen, T. Jantunen, "Head pose estimation for sign language video," in J.-K. Kamarainen and M. Koskela (eds.), Image Analysis. Springer, Lecture Notes in Computer Science, Vol. 7944, pp. 349–360, 2013.

These are stored in a CSV file `data/all-backends_per-image-headposes-lfw.csv`, and the code is present in `headposes_<..>_per_image-lfw.ipynb`.

## 2. Get Race and Gender Classifications
- Using the same collection of detector backends we tried in the headpose estimation (`dlib, mtcnn, retinaface`), by utilizing the toolkit [DeepFace](https://github.com/serengil/deepface), we run methods to get predictions of the race and gender of the subject and their images. We store them in a CSV file `data/all-backends_per-image-labels-lfw.csv`.

## 3. Observe Deviating Classifications from the Mode Gender and Mode Race
- In getting the race and gender classifications for each image, we now group all of those images by person. The classifications will be stored as `{'classified_race_1': 20}` (the race was classified as `classified_race_1` (could be one or more from 'white', 'middle eastern', 'asian' and several others)), or `{'male': 18, 'female': 2}` (in a subject, e.g. 20 images of 'Tom Ridge'). For this, see `labels_grouping_per_person-lfw.ipynb`. 

These are stored in a CSV file `data/all-backends_per-person-labels-lfw.csv`.

<p align='center'>
<img width="1683" height="1795" alt="Component 3" src="https://github.com/user-attachments/assets/b162b535-c53f-462e-b462-73fa9e894208" />
</p>

## 4. Observe Representation with Low-Rank Approximation from Embedding w.r.t. `Facenet512`
- The FR model chosen, and kept constant in the experimentation on the LFW dataset, is the `Facenet512`, from the several others offered on DeepFace. Each image was converted to its embedding form (a high-dimensional array `[]`), and later, the embeddings were grouped by individual. Then, using t-SNE as the method for approximation, the embeddings were converted to a lower 2-d dimension for plotting. This is done for visualization of which images might have deviated from the other images in their representation, by inspecting the outlying points from a cluster (if exists).

Combined code for stages 2, 3, 4 is done in the `labelling+embedding_per_image-lfw.ipynb` notebook. The detailed plot images are in `data/tsne-embeddings_per-person-plots.zip`, labelled by which image of the specific person.

The raw Facenet512 embeddings for each person's images, and the approximated 2D t-SNE embeddings are stored in `tsne_embeddings_per_image-lfw.pickle`.

<p align='center'>
<img width="779" height="1197" alt="Component 4" src="https://github.com/user-attachments/assets/1c854970-3028-4194-a679-1bdaad3d6c70" />
</p>

## 5. Inter- and Intra-Similarity Density Estimation (using Gaussian KDE) and Compute Bhattacharyya Coefficient
- The use of the obtained embeddings is followed in getting an estimation of the model's inter and intra-similarity. Inter-similarity is the correct verification of 2 compared images of distinct subjects, and intra-similarity is the ability to verify 2 images of the same subject.

- Verification is done with different defined distance methods, which can be `cosine, euclidean, euclidean_l2` and more. It is chosen for this investigation to use the `cosine` distance (provided as an argument in `find_distance()` in the submodule of DeepFace `deepface.modules.verification`). 

- From this, we also compute a value known as the Bhattacharyya distance, which is a way to compute the distance between two probability distributions, in our case, being the intra-similarity distribution, and the inter-similarity distribution. Equivalently, the Bhattacharyya Coefficient (BC) is the overlap of the two distributions, which we compute using a Monte-Carlo based approach. Using a similar approach, we get the 95% confidence interval of the estimated coefficient value, using bootstrapping, to have upper and lower bounds of the error in the value. 

```py
def BD_mc(p, q, n=100):
  points = p.resample(n)
  p_pdf = p.pdf(points)
  q_pdf = q.pdf(points)
  return (np.sqrt(p_pdf * q_pdf)/p_pdf).mean()
```

```py
def bootstrap_CI(bd_s, B=1000):
    """B: number of bootstrapping trials"""
    bootstrapped_bd = np.zeros(B)
    boot_length = len(bd_s)
    # For each bootstrap trial
    # Randomly pick a value from the arr of Bhattacharyya values with repl.
    for b in range(B):
        Ci_star = np.random.choice(bd_s, size = boot_length, replace = True)
        bootstrapped_bd[b] = np.mean(Ci_star)

    # Compute percentile confidence intervals
    percentile = np.quantile(bootstrapped_bd, [0.025, 0.975])
    return percentile
```

<p align='center'>
  <img width="930" height="423" alt="Screenshot 2026-01-09 at 23 35 35 1" src="https://github.com/user-attachments/assets/1ceccfe9-ea03-495f-88d8-951d37981356" />
</p>

Above is an example graph. This is done in the `intra-inter-simil_bc-lfw.ipynb` notebook. 

## 6. Link BC value with performance of the FR backends
- We provide a comparison of which detectors from `dlib, mtcnn, retinaface` had a greater BC value, when evaluated from the `Facenet512` embedding + verification model, and with the `cosine` distance. 

## 7. Observe change in BC value with removal of images with Medium- or Very Sharply Turned Headposes
- If there are 20 images of an individual, if we remove 2 images from each individual's collection, which have a drastically high roll/yaw/pitch (e.g. above 60 degrees in either category), we also aim to observe if it makes a difference in the Bhattacharyya coefficient or not. 

<hr/>

## **To Keep in Mind:**
- Some models could handle multiple facial detection, but some can not. In earlier stages of trying DeepFace, when using the basic detector `opencv`, up to 30 images had ambiguous classifications. Manual labelling was required here, if more than one face was present.
- The collection of runs, and notebooks, and data generation and analysis were run using Google Colab.
- The code is written in Python. If certain libraries are not loading, make sure to install them using `pip`. Additionally, make sure to save things during running the code, like the following, to not have to compute them again:
   - id's of individuals when filtering those that have at least 20 images
   - embeddings of the images

## References
- S. Serengil and A. Ozpinar, "A Benchmark of Facial Recognition Pipelines and Co-Usability Performances of Modules", Journal of Information Technologies, vol. 17, no. 2, pp. 95-107, 2024. https://github.com/serengil/deepface
- https://sefiks.com/2020/11/20/facial-landmarks-for-face-recognition-with-dlib/
- https://github.com/ipazc/mtcnn
- https://github.com/serengil/retinaface
- https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

## Datasets
- Labelled Faces in the Wild (LFW) [HuggingFace](https://huggingface.co/datasets/bitmind/lfw)
- FairFace (https://arxiv.org/abs/1908.04913) | [HuggingFace](https://huggingface.co/datasets/ryanramos/fairface)
