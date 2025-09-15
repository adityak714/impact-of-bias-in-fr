# Impact of Bias in Face Recognition

## Contributors:
- Aditya Khadkikar, M.Sc. Data Science student, Uppsala University
- Mark Smithson Rivas, M.Sc. Data Science student, Uppsala University

## Supervisor:
- Anders Hast, Professor in Department of Information Technology, Uppsala University

<hr/>

## **Aim of Research** 
- How does cosine similarity (**inter- and intra-**) vary with social markers/classes? (age, ethnicity, and gender)?

Extended to a broader question:

- How is the facial detection (D), alignment (A) and recognition (R) quality affected based on social markers (age, ethnicity, gender)

(if time permits) 
- What are representation translation methods between different DAR triplets
- Which combinations aim to provide least bias for FR in diverse gender, age and ethnicity groups?

## Important links
- https://github.com/mk-minchul/CVLface
- https://github.com/deepinsight/insightface
- https://github.com/serengil/deepface
- https://github.com/tomas-gajarsky/facetorch

## Useful papers (giving some datasets)
- FairFace (https://arxiv.org/abs/1908.04913) | [HuggingFace](https://huggingface.co/datasets/ryanramos/fairface)
- VGGFace2 (https://github.com/ox-vgg/vgg_face2)

## Ideas Discussed Until Now:
**Preprocessing Step (Understanding the data):**
- Remove if there is more than one face in an image (some pipelines can handle it, but some can not).
- In case certain faces are not perfectly front-facing, or are slightly rotated, find ways to homogenize. Or make it into a new class "Orientation".
- Verify cosine similarity between different poses, and between images of the same person.
- Photos of the same people should not be off in the cos similarity by a certain threshold (can this be investigated)

> Is cosine similarity the same between the varying poses? What is the cutoff at which after reaching a certain cos similarity, a different pose image of the same person can be distinguished?

- The above gained information can be used as a controlled variable in the experimentation process, otherwise reflect in your research why it could not fully be controlled / could be controlled. 

**Stage 1 of the Research Process:**
- Inter-similarity and Intra-similarity (Bhattacharya coefficient - dot product between the curves)
   - Complicated dataset: more overlap (thus faces of the same person are also confused to be photos of other people)

**Try to do this on the FairFace dataset.**

**Stage 2 of the Research Process**:
- Conduct the above inter- and intra-similarity investigation, now for comparing between multiple classes: 
   - Age, Ethnicity, and Gender
- Observe the bias (can be done with the metrics previously provided, along with MAE, MAP and so on.)

DO this for one computer-vision facial recognition library *first* (e.g. CVLFace library), **and compare with potentially other FR pipelines and how well they perform**. 
- Collect **metrics**: Mean Absolute Error, Mean Average Precision, ... 
- Do more trials!
