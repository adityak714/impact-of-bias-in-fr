# Impact of Bias in Face Recognition

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img width="42%" height="auto" alt="Component 7(1)" style="margin: auto;" src="https://github.com/user-attachments/assets/c142b2f8-b2e8-46a2-8b55-c1a0956219ca" /> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
<img width="35%" height="auto" alt="Screenshot 2025-09-15 at 17 16 17 1(1)" style="margin: auto;" src="https://github.com/user-attachments/assets/04c66246-d11c-4fab-a068-35c0cd7e68da" />

> Credit: Combined images from some of currently used facial recognition toolkits [CVLFace](https://github.com/mk-minchul/CVLface) and [DeepFace](https://github.com/serengil/deepface).

### Contributors:
- Aditya Khadkikar, M.Sc. Data Science student, Uppsala University
- Mark Smithson Rivas, M.Sc. Data Science student, Uppsala University

### Supervisor:
- Anders Hast, Professor in Department of Information Technology, Uppsala University

<hr/>

## **Aim of Research** 
How does cosine similarity (**inter- and intra-**) vary with social markers/classes? (age, ethnicity, and gender)? Extended to a broader question, how is the facial detection (D), alignment (A) and recognition (R) quality affected based on social markers (age, ethnicity, gender). If time permits, additional areas are exploring representation translation methods between different DAR triplets, and which combinations aim to provide least bias for FR in diverse gender, age and ethnicity groups.

**Research Process**

<p align="center" width="100%">
   <img align="center" width="50%" height="auto" alt="project-in-ds(1)" src="https://github.com/user-attachments/assets/1f710423-43b9-4933-8167-253b2516c816" /> 
</p>

## Important links
- https://github.com/mk-minchul/CVLface
- https://github.com/deepinsight/insightface
- https://github.com/serengil/deepface

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

<hr/>
