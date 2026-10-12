# **Adversarial Robustness of AI-Generated Image Detectors in the Real World** 

Sina Mavali<sup>1</sup> , Jonas Ricker<sup>2</sup> , David Pape<sup>1</sup> , Asja Fischer<sup>2</sup> , and Lea Schönherr<sup>1</sup> 

> 1 CISPA Helmholtz Center for Information Security, Saarbrücken, Germany `{sina.mavali,david.pape,schoenherr}@cispa.de` 

> 2 Ruhr University Bochum, Bochum, Germany 

```
{jonas.ricker,asja.fischer}@rub.de
```

**Abstract.** The rapid advancement of generative artificial intelligence (GenAI) is accompanied by a concerning rise in its misuse. In particular, generated images are increasingly used to create visually convincing false narratives. To counter this threat, a range of methods has been proposed to distinguish AI-generated images from authentic ones. Yet their adversarial robustness is rarely assessed, and when it is, evaluations are often restricted to controlled rather than real-world settings. In this work, we demonstrate that current state-of-the-art detectors are highly vulnerable to adversarial examples. Most importantly, our experiments show that attacks remain effective even in realistic settings, where images are degraded throughout their lifecycle, _e.g_ ., common post-processing applied during uploads to social media platforms. We also demonstrate the real-world impact of such vulnerabilities by successfully attacking a commercial detection tool. Moreover, we investigate a defense based on robust CLIP features and show that it can improve robustness against adversarial perturbations. While our results suggest that countermeasures can partially mitigate the risk of adversarial examples, we call upon model developers to make greater efforts toward the robust detection of AI-generated images. The source code for this work is available at `https://github.com/dormant-neurons/aigi-break` . 

## **1 Introduction** 

The creation of synthetic visual media is no longer a compelling vision, but has become part of everyday life and a vital business. In seconds, images with any desired content can be created using state-of-the-art generative models [32,40,3]. Recently, the quality of synthetic images has reached a level at which humans cannot distinguish them from real media [18]. Thus, although GenAI can support us in productive and creative tasks, the potential for misuse poses unprecedented challenges to our digital society, for example, by enabling the creation of fake identities on social media platforms [37]. In addition, AI-generated disinformation can discredit individuals, manipulate public opinion, and erode trust in democratic processes [38,13]. While model providers implement safety guardrails to prevent misuse, users often find loopholes. For instance, journalists 



2 S. Mavali, J. Ricker, D. Pape, A. Fischer, and L. Schönherr 

were able to create conspiracy imagery related to the 9/11 attacks and the assassination of John F. Kennedy using Google’s Gemini [20]. Additionally, recent law-enforcement reports show that synthetic media is already being misused in many downstream abuse scenarios. For instance, the FBI attributes nearly $893 million in reported losses in 2025 to AI-related complaints and notes that scammers use fake social profiles, voice clones, forged identity documents, and realistic videos of public figures or loved ones [15]. 

Since AI-generated images (AIGIs) are often widely shared on various social media platforms before being removed [11], they pose a serious risk to global political discourse and public trust. Due to this growing threat, detecting AIGIs is an active research topic. Proactive methods such as watermarking [17,16] and cryptographically signed metadata in image files [6] pose significant challenges: These methods require all parties involved to agree on the same standard and cooperate, while adversaries could use custom generative models or simply remove metadata or watermarks before distributing harmful images [27,24,35]. Another approach, which does not require the cooperation of model providers, is to passively detect synthetic images based on forensic artifacts. The majority of detectors utilize standard convolutional neural networks (CNNs) [43,19] or pre-trained foundation models [34,5] to distinguish between generated and real images. While these detectors often achieve very high accuracy, they are typically evaluated only in a _clean_ setting, without considering an attacker’s attempts to bypass detection. However, as previous work suggests, adversarial examples can dramatically reduce the performance of AIGI detectors [4,1]. Yet this prior work leaves several practical questions open. Recent work [14] focuses on constructing adversarial datasets, but does not analyze the quality-attack trade-off that determines whether perturbations remain stealthy. More critically, existing evaluations largely operate in idealized settings, ignoring the image degradations (e.g., compression, resizing) that occur when images are shared on social media [4]. Furthermore, prior work has not validated these vulnerabilities against commercial detection systems. Finally, while adversarial defenses exist for other vision tasks, it remains unexplored whether they are useful for AIGI detectors. 

In this work, we address these limitations and evaluate AIGI detectors under conditions they face in the real world. In particular, we study whether adversarial examples remain effective after common social media platform degradations such as resizing, JPEG compression, and other post-processing. Through extensive experiments with four detection methods, three AIGI datasets, and five attack algorithms, we demonstrate that state-of-the-art detectors can be successfully deceived with minimal impact on visual quality and without any knowledge of the detector’s internals. Critically, most attacks remain effective if images are degraded, as typically happens when they are uploaded to a social media platform. In combination with the general performance loss caused by image post-processing, our results show that detectors are effectively unusable under realistic, adverse conditions. To validate our findings, we confirm the effectiveness of our attacks on a popular proprietary AIGI detector, HIVE. While our results underline the vulnerability of current AIGI detectors, we also demonstrate that 



Adversarial Robustness of AI-Generated Image Detectors 3 

their robustness can be significantly improved by leveraging robustly pre-trained feature extractors, a practical defense approach not previously explored for AIGI detection. 

We summarize our findings as follows: 

- **State-of-the-art AIGI detectors are vulnerable to adversarial examples.** For instance, the Diverse Input Attack can render detectors unusable, reducing the average AUROC from 87.5% on clean inputs to approximately 0.0%. Critically, the attacker does not need internal knowledge of the detector ( _i.e_ ., a black-box setting). 

- **Attacks remain effective even if images are altered in social media uploads** involving compression, resizing, etc. Under the strongest attack, detectors still collapse after high degradation, with average AUROC dropping to 28.2%; degradation alone can already reduce clean average AUROC from 87.5% to 66.6%. 

- **Adversarial examples are not only a theoretical risk, but can be used against real-world commercial detection tools** such as HIVE. In our case study, HIVE’s average AUROC drops from 98.9% on clean images to 76.6% under attack; after responsible disclosure, the HIVE team confirmed the validity of our results. 

- Despite these vulnerabilities, **we demonstrate that adversarial robustness can be achieved without training detectors from scratch.** By replacing standard feature extractors with robustly pre-trained backbones (RobustCLIP), AIGI detectors gain significant protection against strong attacks ( _e.g_ ., AUROC increasing from 0.0 to 75.7 under the Diverse Input attack). 

Based on our findings, we urge researchers and model developers to consider adversarial robustness when evaluating their detectors and call for further research on the development of robust AIGI misuse prevention. 

## **2 Related Work** 

**Detection of AI-Generated Images.** Due to the potentially harmful consequences of photorealistic synthetic images, various detection methods have been proposed. Among them, the most common approach for detecting generated images is to learn discriminative features directly from the data. Training CNNs, such as ResNet-50 [21], on real and generated images has been shown to be surprisingly effective [43,19,7]. However, developing detectors that can adapt to unseen generative models is an ongoing challenge. Recently, several works [34,8,5] have shown that using large vision foundation models (e.g., CLIP [36]) can help train such universal detectors. These methods train a relatively simple classifier on top of the features extracted by the pre-trained model, resulting in good generalization capabilities and higher robustness to perturbations. A possible explanation is that by using the pre-trained feature extractor, the detector does 



4 S. Mavali, J. Ricker, D. Pape, A. Fischer, and L. Schönherr 

not overfit on specific artifacts of a certain generative model, but learns to identify more general properties of synthetic images [34]. In our work, we focus on AIGI classifiers due to their generalizability and their inherent vulnerability to adversarial examples. We refer to the recent work of Tariang _et al_ . [41] for a more detailed overview. 

**Adversarial Robustness of AIGI Detectors.** Pioneering work by Carlini & Farid [4] demonstrated that forensic classifiers were fundamentally vulnerable to both white-box and black-box attacks. Following this, a significant thread of research pursued customized attacks tailored specifically to the AIGI detection task. For instance, some methods generated high-quality adversarial images by optimizing within the latent manifold of a specific generator, such as StyleGAN [28] or by adversarially modifying image attributes ( _e.g_ ., exposure, noise, and blur) to minimize statistical differences between real and fake images [23]. Other approaches focused on removing generative artifacts through trace removal or retouching methods [29]. Abdullah _et al_ . [1] propose a semantic-based attack that leverages a fine-tuned foundation model to craft adversarial samples by manipulating image content. 

All of these specialized attacks often suffer from critical limitations: Many are tightly coupled to specific GANs, rendering them less relevant for SOTA diffusion-based detectors, while others rely on impractical assumptions, such as requiring an attacker to possess deep knowledge of the image generation pipeline. We argue that the pursuit of novel attacks, while technically interesting, has obscured a more fundamental vulnerability. Our findings demonstrate that simpler, classical adversarial methods are not only sufficient but are often more effective, as shown in our attack comparison in Section 5. There have also been efforts to create adversarial datasets for AI-generated images. More recently, Eddoubi _et al_ . [14] proposed a large dataset of adversarial examples for AIGI detectors. However, their work has two main limitations. First, the adversarial examples are generated with a large noise budget, resulting in poor perceptual quality that compromises their stealthiness. Furthermore, no trade-off between attack success and quality is explored. Second, it does not investigate whether these examples remain effective after realistic post-processing such as compression and resizing. Our work addresses these limitations by evaluating attacks under realistic degradations and validating against a commercial system. 

## **3 Threat Model** 

The general objective of the attacker is to create an adversarial example that bypasses AIGI detection tools, for example, in a social media platform’s moderation system or an individual user’s verification tool. For this, we consider a realistic black-box scenario where the attacker has no access to the target model’s gradients or parameters. Unlike white-box settings that assume full transparency, this threat model mirrors the knowledge gap an adversary faces when targeting proprietary social media detectors or commercial APIs. In this setting, the attacker must rely on surrogate models to craft adversarial examples, exploiting the 



Adversarial Robustness of AI-Generated Image Detectors 5 

transferability of adversarial perturbations. This approach tests the generalization of attacks across different architectures (e.g., transferring from a CNN-based surrogate to a transformer-based target) and training data. 

Furthermore, our threat model accounts for the constraints that real-world deployment might impose on an attacker. Images uploaded to the internet inevitably undergo post-processing steps such as compression and resizing. Additionally, an attacker may deliberately degrade an image before uploading it (e.g., by blurring or adding noise) to hide forensic artifacts and increase the success chance of the attack. Consequently, we evaluate attack success not just on adversarial examples, but under various levels of degradation. 

Finally, we anticipate an adaptive defender who may employ defense mechanisms to harden their detectors. By evaluating attacks against both standard and robustified models, we provide a comprehensive view of the current security landscape for AIGI detection. 

## **4 Methodology** 

In this work, we systematically evaluate the performance of state-of-the-art detection methods under realistic attack scenarios. To this end, we have curated a diverse collection of datasets that includes images generated by various generative models alongside a wide range of real images. We further utilize a series of attacks and degradations to assess the effectiveness of current detection methods. 

### **4.1 Datasets** 

Our analysis employs three different datasets: Synthbuster [2], Chameleon [46], and images generated by GPT-4o [47]. These datasets collectively offer a broad spectrum of synthetic and real images, enabling a comprehensive evaluation of detection methods. _Synthbuster_ provides a controlled evaluation environment by pairing 1,000 uncompressed real photographs from RAISE-1k [9] with synthetic counterparts generated by nine different diffusion models. This design mitigates dataset biases and allows for a granular analysis of detector performance across various generative architectures. To test detectors in a more challenging setting, we use the _Chameleon_ dataset. It contains high-resolution synthetic images that have passed a human perception “Turing Test,” making them perceptually indistinguishable from their real counterparts, which are sourced from online photo repositories. As the _GPT-4o_ dataset lacks a paired set of real images, we employ the real images from the Synthbuster dataset (RAISE-1k) for evaluation. 

### **4.2 Detectors** 

For our evaluation, we consider four state-of-the-art AIGI image detectors that have demonstrated high accuracy and robust generalization in detecting diffusion model (DM)-generated images: Corvi [19], UnivFD [34], and DRCT [5] (both 



6 S. Mavali, J. Ricker, D. Pape, A. Fischer, and L. Schönherr 

Conv-B and CLIP versions). These detectors were chosen to span a diverse range of concepts, architectures, and training paradigms. 

**Corvi.** This detector leverages a ResNet-50 [21] and relies on heavy data augmentation during training. The notable improvement is the removal of the initial downsampling layer, which preserves high-frequency information at the cost of a higher number of trainable parameters. 

**UnivFD.** Ojha _et al_ . [34] address universal fake image detection by leveraging a large vision foundation model, specifically CLIP-ViT-L/14 [12], as a feature extractor. A single linear layer is then trained on top of these pre-extracted features, while the CLIP weights remain frozen. This decoupled training approach improves the detector’s generalization. 

**DRCT.** Diffusion reconstruction contrastive training [5] is a training paradigm that enhances existing AIGI image detectors by applying a contrastive loss to difficult image pairs. These pairs consist of a real image and its reconstruction, created by passing it through a diffusion model’s forward-and-backward process. This method, which enhances generalization, is supported by both ConvNeXt and CLIP backbones. 

For DRCT-CLIP and DRCT-ConvB we use publicly available checkpoints trained on the SD-1.4 train subset from GenImage [49]. Since no matching checkpoints are available for UnivFD and Corvi, we retrain both models on the same dataset, following the authors’ instructions. 

### **4.3 Attacks** 

The objective of an attack against AIGI detection is to generate a perturbed image _x_ adv that remains nearly indistinguishable from the original generated image _x_ yet is classified by the detector as real if it is AI-generated and vice versa. The perturbed image _x_ adv is called an adversarial example. The task of finding such adversarial examples can be formulated as a constrained optimization problem: 



where _y_ represents the ground-truth label, _∥·∥p_ denotes the _ℓp_ -norm, and _L_ ( _·_ ) is typically the cross-entropy loss. The parameter _ϵ_ defines the maximum allowed distortion between _x_ and _x_ adv. Attacks can generally be grouped into whitebox and black-box methods. Under the white-box scenario, the attacker has complete knowledge of the detector’s parameters and architecture, allowing the direct use of gradient-based techniques. In contrast, black-box attacks restrict the adversary to using a surrogate model to approximate its gradients or querying the target model. 

We evaluate our model under a realistic black-box threat model, generating adversarial examples with attacks known for high transferability. These are grouped into two categories: transfer-based and query-based attacks. Transferbased attacks use a surrogate model to craft adversarial examples offline, which 



Adversarial Robustness of AI-Generated Image Detectors 7 

is more efficient and stealthy. The success of these attacks depends on the similarity between the surrogate(s) and the target models. In contrast, query-based attacks interact directly with the target model, using its output to guide the attack. We use five representative methods, split into four transfer-based and one query-based attack: 

- **Projected Gradient Descent (PGD)** [30]: An iterative attack that makes small, imperceptible changes to an image by following the gradient of the loss function. We compute adversarial examples on a surrogate model and use their transferability to attack the target model. 

- **Ensemble-based** [42]: Extends PGD by averaging the gradients from multiple surrogate models to improve transferability. 

- **Diverse Input Attack** [44]: Enhances the ensemble attack by applying random transformations, like resizing, to the input at each iteration. 

- **Universal Perturbations** [33]: Finds a single, image-agnostic perturbation designed to fool the detector on a wide range of inputs, which we use in an ensemble mode. 

- **Query-Efficient Attack** [26]: Estimates the gradient direction by sending a few queries with structured noise to the target model and observing the outcomes. 

### **4.4 Degradations** 

To complement the adversarial threat model, we also study quality degradations that naturally arise on social media platforms. Xu _et al_ . [45] showed that the degradation induced by real-world social media platforms ( _e.g_ ., Facebook, Twitter) can be simulated using a combination of common transformations. While testing on live platforms would be the ideal case, it is generally not feasible due to Terms of Service violations regarding API limits and the upload of AIgenerated content. Furthermore, such testing lacks scientific reproducibility due to the possible changes in platform compression. Therefore, we rely on simulated degradation protocols as seen in previous research [43,31]. To assess detector performance under both clean and adversarial conditions in real-world scenarios, we adopt the degradation simulation introduced by Xu _et al_ ., which corresponds to typical and severe processing scenarios: 

- **Medium Degradation** : Corresponds to the common degradation levels observed on social media platforms. Images are downsampled to 75% of their original size and JPEG compression is applied with a quality factor of 50. 

- **High Degradation** : Represents severe image quality loss, simulating scenarios with aggressive compression or multiple retransmissions. We achieve this by combining four types of degradation: Gaussian blurring (kernel size 5 _×_ 5, _µ_ = 1, _σ_ = 1), downsampling to 50% of the original size, heavy JPEG compression with a quality factor of 20, and the addition of Gaussian noise ( _µ_ = 1, _σ_ = 0 _._ 1). 



8 S. Mavali, J. Ricker, D. Pape, A. Fischer, and L. Schönherr 

**Terminology.** To clarify our experimental conditions, we define four image categories along two axes: adversarial perturbation and post-processing degradation. **Clean** images are unperturbed and without degradation. **Degraded** images are unperturbed but have undergone post-processing such as compression or resizing. **Adversarial** images contain adversarial perturbations but no degradation. Finally, **degraded adversarial** images are adversarial examples that have also been degraded, representing the realistic scenario where an attacker uploads a perturbed image to social media. 

### **4.5 Metrics** 

To evaluate detector performance and attack effectiveness, we use three complementary metrics. **AUROC** (Area Under the ROC Curve) serves as our primary metric since it is threshold-independent and enables fair comparison across detectors. For practical interpretability, we also report **Accuracy** (using a fixed decision threshold of 0.5) and **TPR@5%FPR** (true positive rate when false positive rate is fixed at 5%), representing an operating point where false alarms are minimized. We exclude Attack Success Rate (ASR) as it cannot distinguish between a powerful attack and a detector with weaker clean performance. Lower AUROC, Accuracy, and TPR@5%FPR indicate weaker detector performance; for attacked samples, this corresponds to a more successful attack. 

To quantify the impact of attacks relative to clean performance, we report the **AUROC Drop Ratio** , defined as: 



This metric normalizes the reduction in performance by the detector’s initial performance, which allows a fair comparison across different detectors and degradation levels with varying initial performance. We report absolute AUROC in tables to show actual detection performance, while using AUROC Drop Ratio in visualizations to highlight relative attack effectiveness. 

## **5 Robustness Analysis** 

In this section, we present a comprehensive analysis of the adversarial robustness of four state-of-the-art detectors. We perform a robustness analysis comprising five attacks in a black-box setting and investigate the effectiveness and quality of the resulting adversarial examples. Next, we examine the impact of real-world image degradations ( _e.g_ ., social media compression) on attack performance, followed by a case study targeting a closed-source online detection tool. Finally, we demonstrate that deploying a simple defense mechanism can significantly enhance the robustness of the detectors against the strongest attack. 



Adversarial Robustness of AI-Generated Image Detectors 9 

### **5.1 Baseline Attack Evaluation** 

We first examine the adversarial robustness of our AI-generated image detectors by testing their performance on adversarial examples created using five attacks in the black-box setting. For this, we start with evaluating adversarial examples that have not been subjected to any degradation. 

**Setup.** We use the _ℓ∞_ norm with _ϵ_ = 8 _/_ 255 for all attacks and a step size of 2 _/_ 255 for all transfer-based attacks. We run projected gradient descent (PGD), ensemble-based, and diverse input attack for 10 steps, while the universal perturbation attack requires 50 steps due to the harder optimization problem. We validated this choice by comparing attack effectiveness across 5, 10, and 20 steps on the Synthbuster dataset (see Table 6 in the Appendix). The results show negligible differences in AUROC (less than 1 percentage point), indicating that 10 steps is sufficient for convergence. For the query-efficient attack, we use a maximum of 500 queries and 50 samples per draw for the gradient estimation, alongside a noise scale _σ_ = 0 _._ 005 and a minimum step size of _ϵ ·_ 0 _._ 01. We evaluate attacks on a balanced test set of 1,500 images: 500 images per dataset (Synthbuster, Chameleon, GPT-4o), with each subset containing 250 real and 250 AI-generated images. Adversarial perturbations are applied to both classes to flip the detector’s prediction (i.e., making AI-generated images appear real and vice versa). For Synthbuster, we limit our analysis to images generated by Stable Diffusion 1.4 (SD-1.4), as detectors achieve high performance on this subset in the clean setting. 

**Attack Comparison.** Table 1 presents our robustness analysis averaged across all datasets, reporting AUROC alongside Accuracy and TPR@5%FPR. We evaluate the attacks and their transferability across different degradation levels, where “None” refers to attacks without degradation. “LTO” refers to “LeaveTarget-Out”: For each target detector, the ensemble attack is optimized on all other detectors and then evaluated on the held-out target. Standard **PGD** attacks, when transferred between models, are largely ineffective, though exceptions occur when the source and target models are well aligned (e.g., PGD attacks from DRCT-CLIP significantly impact UnivFD, dropping the AUROC from 91.4 to 18.0). However, attacks from a different source, such as DRCT-ConvB, are less effective (AUROC drops to 80.5). We analyze PGD transferability in Figure 1, highlighting strong transfers between CLIP-based detectors (UnivFD and DRCT-CLIP) and weaker transfers from CNN-based sources (DRCT-ConvB, Corvi) to CLIP-based targets. While Corvi shows considerable robustness to transferred PGD attacks, white-box analyses (diagonal of Figure 1) confirm that this does not imply inherent immunity when PGD is directly optimized on the target. For a comprehensive breakdown of these results across each dataset, we refer to Table 5 in the Appendix. 

To counter the limited transferability of individual PGD, **Ensemble-based Attacks** , specifically a Leave-Target-Out (LTO) ensemble, prove more effective. Adversarial examples crafted against an ensemble of all detectors except the target reduce the average AUROC across detectors and datasets to 14.0 (down from 48.8 for PGD), though Corvi remains more resilient (AUROC 50.3). The 



10 S. Mavali, J. Ricker, D. Pape, A. Fischer, and L. Schönherr 

Table 1: **Detectors’ Robustness.** AUROC, Accuracy, and TPR@5%FPR averaged across all datasets for clean (no attack) and different attacks with varying degradation levels (None, Medium: 75% downsampling and JPEG compression at quality 50, and High: Gaussian blur (5 _×_ 5, _µ_ = 1, _σ_ = 1), 50% downsampling, JPEG compression at quality 20, and Gaussian noise ( _µ_ = 1, _σ_ = 0 _._ 1)). “LTO” means Leave-Target-Out: attacks are optimized on all detectors except the evaluated target. 

|**Attack**|**Target**|**Source**|**A**|**URO**|**C**|**A**|**ccurac**|**y**|**TPR**|**@5%**|**FPR**|
|---|---|---|---|---|---|---|---|---|---|---|---|
||||**None **|**Med **|**High**|**None **|**Med **|**High**|**None **|**Med **|**High**|
||UnivFD|–|91.4|89.5|89.7|80.9|79.2|78.5|65.9|63.1|62.4|
||DRCT-CLIP|–|87.7|73.2|73.8|76.5|63.8|64.2|61.1|38.9|39.5|
|**Clean**|DRCT-ConvB|–|85.6|80.4|46.8|75.1|70.3|52.1|51.6|43.2|12.8|
||Corvi|–|85.4|77.4|55.9|70.7|64.3|53.8|60.1|48.7|24.3|
||Average|–|87.5|80.1|66.6|75.8|69.4|62.2|59.7|48.5|34.8|
|||DRCT-CLIP|18.0|61.5|64.0|29.8|58.1|55.1|0.1|16.7|13.5|
||UnivFD|DRCT-ConvB|80.5|85.1|68.3|71.5|72.6|57.9|39.2|50.5|20.9|
|||Corvi|77.5|81.8|70.3|66.2|73.5|59.1|36.1|44.9|24.7|
|||UnivFD|7.5|38.5|59.1|30.4|49.0|50.6|0.0|4.5|11.7|
||DRCT-CLIP|DRCT-ConvB|62.3|68.5|62.5|52.8|50.4|51.3|14.7|21.2|14.1|
|**PGD**||Corvi|63.6|71.7|63.4|53.0|50.6|51.3|18.8|26.7|14.1|
|||UnivFD|58.5|53.6|47.0|50.5|49.9|48.1|14.0|9.3|4.7|
||DRCT-ConvB|DRCT-CLIP|16.4|45.1|46.9|44.5|47.7|48.6|0.8|4.5|3.5|
|||Corvi|0.7|5.5|42.8|7.6|14.4|47.0|0.0|0.0|3.5|
|||UnivFD|63.3|54.0|49.2|58.2|50.2|50.0|29.9|16.3|6.5|
||Corvi|DRCT-CLIP|60.9|57.9|50.9|56.9|50.1|50.0|27.2|16.4|8.1|
|||DRCT-ConvB|76.4|67.5|53.6|60.3|50.0|50.0|38.0|24.1|8.3|
||Average|–|48.8|57.6|56.5|48.5|51.4|51.6|18.2|19.6|11.1|
||UnivFD|LTO|0.0|51.5|63.7|0.6|52.5|54.7|0.0|8.1|14.3|
||DRCT-CLIP|LTO|0.0|8.5|52.5|0.0|45.9|49.1|0.0|0.0|9.6|
|**Ensemble**|DRCT-ConvB|LTO|0.5|4.2|41.1|3.7|10.5|43.8|0.0|0.0|2.5|
||Corvi|LTO|0.0|1.2|42.7|0.1|41.5|50.0|0.0|0.0|5.6|
||Average|–|0.1|16.4|50.0|1.1|37.6|49.4|0.0|2.0|8.0|
||UnivFD|LTO|0.1|6.0|47.9|3.1|19.4|50.3|0.0|0.0|6.1|
||DRCT-CLIP|LTO|0.0|0.0|29.5|0.0|12.5|44.6|0.0|0.0|1.3|
|**Diverse**|DRCT-ConvB|LTO|0.0|0.1|12.8|1.3|0.2|21.4|0.0|0.0|0.0|
||Corvi|LTO|0.0|0.0|22.5|0.1|8.6|49.8|0.0|0.0|0.8|
||Average|–|0.0|1.5|28.2|1.1|10.2|41.5|0.0|0.0|2.1|
||UnivFD|LTO|28.6|79.4|65.3|35.6|69.1|57.6|0.8|34.8|16.3|
||DRCT-CLIP|LTO|0.0|62.1|60.9|0.5|50.1|50.5|0.0|12.5|14.4|
|**Universal**|DRCT-ConvB|LTO|0.1|12.5|41.7|2.4|24.6|43.8|0.0|0.0|2.3|
||Corvi|LTO|1.5|9.9|43.3|12.1|45.2|50.1|0.0|0.0|4.9|
||Average|–|7.6|41.0|52.8|12.7|47.2|50.5|0.2|11.8|9.5|
||UnivFD|–|66.2|76.6|68.7|59.1|67.6|57.2|21.3|36.8|18.1|
||DRCT-CLIP|–|71.1|68.6|62.8|55.1|50.7|51.4|28.0|23.2|13.9|
|**Query**|DRCT-ConvB|–|71.3|58.2|48.9|59.1|52.6|48.7|24.5|10.1|4.9|
||Corvi|–|86.4|59.4|47.9|64.5|50.1|50.0|53.6|17.3|6.7|
||Average|–|73.8|65.7|57.1|59.5|55.2|51.8|31.9|21.9|10.9|





Adversarial Robustness of AI-Generated Image Detectors 

11 



<!-- Start of picture text -->
ViT CNN<br>UnivFD 99.5% 88.8% 31.2% 19.6% 60.6%<br>DRCT-CLIP 72.7% 99.8% 76.0% 23.9% 68.4%<br>DRCT-ConvB 9.1% 23.7% 78.7% 14.3% 31.0%<br>Corvi 10.4% 20.7% 99.2% 100.0% 56.6%<br>Average 48.0% 58.3% 71.3% 39.4% 54.2%<br>Target Detector<br>UnivFDDRCT-CLIPDRCT-ConvB Corvi Average<br>ViT<br>CNN<br>Source Detector<br><!-- End of picture text -->

Fig. 1: **Transferability Analysis (PGD,** _ϵ_ **=8/255).** Each cell shows the **AUROC Drop Ratio (%)** . Darker colors indicate more effective attacks. Detectors are grouped by architecture: ViT-based (UnivFD, DRCT-CLIP) and CNN-based (DRCT-ConvB, Corvi). Diagonal cells show white-box attacks; off-diagonal cells show black-box transferability. The **Average** row/column summarizes overall effectiveness per detector. 

**Diverse Input Attack** , an enhancement of this ensemble approach, is even more successful, reducing the average AUROC to a mere 2.0 and significantly degrading Corvi’s performance (AUROC 6.0). 

Alternatively, an attacker might employ a **Universal Perturbation Attack** , seeking a single perturbation applicable to any image class (real/fake). While universal perturbations are less effective than ensemble or diverse imagespecific attacks, they still pose a significant threat as they can be added to an image on the fly, without any computation, and can be shared easily. They achieve an average AUROC of 36.8, outperforming PGD transfer attacks (48.8 AUROC), a trend observed across all the datasets. In contrast, a **Query-efficient Attack** is one of the least effective in our setup. We report an average AUROC of 64.9. We attribute this lower performance to the restricted query budget and sampling strategy. 

Overall, our findings show that PGD attacks have limited transferability. This is because they heavily rely on the specific gradients of the source model. Ensemble-based attacks significantly improve adversarial effectiveness, especially when combined with input diversity. This discourages overfitting to any single 



12 S. Mavali, J. Ricker, D. Pape, A. Fischer, and L. Schönherr 



<!-- Start of picture text -->
Original Adversarial Med. Degradation High Degradation<br><!-- End of picture text -->

Fig. 2: **Visual effect of various degradation levels on adversarial examples.** Example showing an adversarial example (Diverse Input attack, _ϵ_ = 8 _/_ 255) and its appearance after medium and high degradation (degraded adversarial). Robust attacks aim to remain effective after such transformations. 

model and makes the attack robust to common image transformations, ensuring the perturbation is not reliant on specific pixel locations. Although universal perturbations are only computed once, they offer a modest improvement over standard PGD and are far easier to deploy, making them a scalable attack. In contrast, query-efficient attacks remain the least effective in our evaluation and appear less promising under current constraints. 

**Finding 1.** Standard PGD attacks struggle to transfer across different architectures and models. Ensembling detectors, particularly when augmented with input diversity, produces highly effective and transferable adversarial examples. 

### **5.2 Attacks in Real-world Scenarios** 

While the previous analysis demonstrates attack effectiveness without degradation, real-world scenarios introduce additional complexities. We consider attacks in a setting, where adversarial examples, spread via social media or messaging apps, frequently undergo transformations such as resizing and compression, as shown by Xu _et al_ . [45]. Our analysis assumes that an attacker aims to evade detection across various degradation levels (none, medium, high) encountered in an image’s digital life cycle, rather than focusing on less common, specific edits. To illustrate the visual impact of these settings, Figure 2 provides a qualitative comparison between the original image, the adversarial examples, and their degraded counterparts. To ensure a fair comparison and measure the effectiveness of the attack without bias, we first report detector performance in the clean 



Adversarial Robustness of AI-Generated Image Detectors 13 



<!-- Start of picture text -->
1.0 Attack Type 1.0 0.99 0.96 0.98 0.95 0.96 0.93<br>Diverse 0.88<br>Ensemble<br>0.8 Universal 0.8 0.77<br>PGD<br>Query 0.62<br>0.6 0.6 0.57 0.54<br>Attack Type 0.47<br>Diverse<br>0.4 0.4 Ensemble<br>Universal<br>PGD<br>0.2 0.2 QueryEpsilon4/255 0.13 0.11 0.10<br>8/255<br>0.0 0.0 16/255<br>None Medium High 0 20 40 60 80 100<br>Degradation Level Quality Score<br>(a) (b)<br>AUROC Drop Ratio  AUROC Drop Ratio<br><!-- End of picture text -->

Fig. 3: **Impact of Degradation on Attack Effectiveness and Quality.** (a) Impact of degradation severity on attack effectiveness, quantified by the AUROC Drop Ratio. Results are averaged over all tested detectors and datasets. (b) Comparison of three _ϵ_ for tested attacks. Image quality decreases notably with larger _ϵ_ (lower quality score), while AUROC remains relatively stable, suggesting that minimal perturbations suffice for effective attacks with high image quality. 

setting over these degradations in Table 1. The results show that most detectors are generally robust to these transformations, and their performance is slightly affected. CLIP-based detectors (UnivFD and DRCT-CLIP) exhibit greater robustness, potentially due to their reliance on high-level features. In contrast, CNN-based detectors (DRCT-ConvB and Corvi), which depend more on lowlevel features, are consequently more affected by these (often low-level) transformations, even though such common transformations are part of the training set for all detectors. 

Subsequently, we evaluate the impact of image degradations on the effectiveness of adversarial examples. The overall impact on detector performance is detailed in the right two columns of Table 1 and further illustrated in Figure 3a. 

In general, most attacks become less effective after transformation, although they can still pose a substantial threat. The **Diverse Input attack** shows greater robustness to degradation compared to the **Ensemble attack** , as evidenced by a smaller performance drop from the “None” to “Medium” degradation levels. We hypothesize that this is due to the incorporation of resizing into the adversarial example generation process, which helps the attack generalize better under transformation. In contrast, the **Universal attack** is the least robust, suffering from a significant performance decline when comparing no degradation to medium degradation (as showcased by the drop in Figure 3a). This is expected, as universal perturbations are generic and intended to work across an entire dataset; even minor alterations can make them ineffective. 

In summary, while transformations reduce attack effectiveness (as indicated by the AUROC drop ratio), even severe degradations that significantly alter the image do not completely neutralize the adversarial noise. It is also noteworthy 



14 S. Mavali, J. Ricker, D. Pape, A. Fischer, and L. Schönherr 

Table 2: **Quantitative evaluation of various attack methods.** The table compares five attacks using image quality metrics and the AUROC drop ratio as an attack-effectiveness measure. Adversarial examples are generated with _ϵ_ = 8 _/_ 255 in the no-degradation black-box setting. The arrows ( _↑ / ↓_ ) indicate the <u>preferable</u> direction for each metric. 

|**Metric** (Range)|**PGD**|**Ensemble**|**Diverse**|**Universal**|**Query**|
|---|---|---|---|---|---|
|Mean Pert. _↓_[0, _∞_)|0.077|0.076|0.108|0.020|0.058|
|SSIM _↑_[0, 1]|0.917|0.921|0.871|0.875|0.943|
|PSNR _↑_[0, _∞_)|33.97|34.07|31.63|32.48|49.52|
|LPIPS _↓_[0, _∞_)|0.093|0.092|0.161|0.156|0.053|
|DISTS _↓_[0, _∞_)|0.127|0.122|0.165|0.140|0.079|
|Quality Score _↑_[0, 100]|58.12|59.27|40.73|55.14|81.82|
|AUROC Drop Ratio _↑_[0, 100]|54.2|95.0|98.0|76.4|11.4|



that medium degradation is the setting closest to common social-media processing, following the degradation model of Xu _et al_ . [45], while high degradation captures more extreme cases such as aggressive compression or repeated postprocessing. 

**Finding 2.** While degradations on adversarial examples reduce attack effectiveness, they do not neutralize them. Diverse Input attacks are most robust to these transformations, while Universal attacks are most fragile. 

### **5.3 Adversarial Example Quality Analysis** 

Analyzing the quality of adversarial examples is important to ensure they are not only effective in evading detection but also maintain high visual fidelity. **Setup.** To quantitatively evaluate the quality of adversarial examples, we employ five widely-used metrics: Mean Perturbation, Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), Learned Perceptual Image Patch Similarity (LPIPS) [48], and Deep Image Structure and Texture Similarity (DISTS) [10], selected to capture different aspects of image quality, including perceptual similarity and structural integrity. These metrics are computed between adversarial examples and their original counterparts without degradation, as constant degradation applied to both would not alter their relative quality. For a unified and holistic measure, we compute a normalized average of these five metrics with similar weights, presented as “Quality” (0-100, with 100 being the highest similarity) in Table 2. 

**Quality-Attack Success Trade-off.** The results in Table 2 reveal a strong correlation between an attack’s effectiveness, measured by AUROC drop ratio, and the visual quality of the generated adversarial examples. For instance, the Ensemble attack and Universal attack demonstrate quality on par with the PGD 



Adversarial Robustness of AI-Generated Image Detectors 15 

attack, yet their effectiveness scores are substantially higher (0.95 for Ensemble and 0.76 for Universal, compared to 0.54 for PGD), indicating greater impact while preserving similar visual fidelity. Conversely, the Diverse Input attack, which has the highest score of 0.98, exhibits the lowest quality. To facilitate a fair comparison, we plot this success metric against the Quality Score for three distinct _ϵ_ values (4/255, 8/255, and 16/255) in Figure 3b. This analysis shows a significant decrease in quality with increasing _ϵ_ , while the corresponding gain in attack efficacy is not as pronounced for methods like PGD, Ensemble, Diverse Input, and Query-efficient. This demonstrates that an attacker can achieve high impact by carefully selecting the attack method and an appropriate _ϵ_ while maintaining excellent visual fidelity. Samples of these adversarial example images are shown in Figure 4 in the Appendix. 

**Finding 3.** The trade-off between image quality and attack success is not linear. Increasing perturbation magnitude ( _ϵ_ ) significantly degrades quality without proportionally increasing attack success. This shows that the optimization technique matters more than the perturbation budget. 

### **5.4 Case Study – Commercial Detector** 

Our analysis identifies the Diverse Input attack as highly effective, achieving a significant AUROC drop ratio while producing good quality images with an appropriate epsilon value. To assess its real-world applicability, we conducted a case study on HIVE [22], a widely-used closed-source online AIGI detector. Using HIVE’s free web interface, we tested the Diverse Input attack (using an ensemble of all four detectors) on a subset of 200 images (100 from each of the Synthbuster and Chameleon datasets, representing 20% of our evaluation set). This sample size was constrained by the interface’s rate limits. The results are presented in Table 3. Notably, in the clean setting, commercial detectors like HIVE significantly outperform their academic counterparts, with HIVE achieving near-perfect AUROC on both datasets. However, when attacked with _ϵ_ = 8 _/_ 255, HIVE’s AUROC dropped from 100.0 to 73.8 on Synthbuster and from 97.8 to 79.4 on Chameleon dataset. 

To conduct a more in-depth analysis, we used the 0.9 threshold suggested by HIVE and measured the attack performance on real and fake classes separately. A key observation is the divergent impact of adversarial perturbations across datasets. On Chameleon, the attack primarily induces false positives. Comparing perturbation magnitudes reveals that at _ϵ_ = 4 _/_ 255, fake detection accuracy drops to 72.0%, but at _ϵ_ = 8 _/_ 255, it recovers to 82.0%, while real accuracy continues to degrade (from 82.0% to 69.4%). This suggests that for the high-resolution images in Chameleon, high-magnitude adversarial noise acts as an additive artifact that mimics synthetic traits. Upon disclosure, the HIVE team confirmed the validity of our results. 



16 S. Mavali, J. Ricker, D. Pape, A. Fischer, and L. Schönherr 

Table 3: **Case Study on HIVE.** Performance under attacks showing AUROC, Accuracy, and TPR@5%FPR <u>(%)</u> under different <u>perturbation</u> bounds. 

|**Metric**|**Dataset**||**Atta**|**cked**|
|---|---|---|---|---|
|||**Clean**|_ϵ_=**4**_/_**255**|_ϵ_=**8**_/_**255**|
|**AUROC**|Synthbuster<br>Chameleon|100_._0<br>97_._8|90_._3<br>83_._3|73_._8<br>79_._4|
||Average|98_._9|86_._8|76_._6|
||Synthbuster|100_._0|82_._0|65_._0|
|**Accuracy**|Chameleon|91_._0|77_._0|75_._7|
||Average|95_._5|79_._5|70_._4|
||Synthbuster|100_._0|72_._0|30_._0|
|**TPR@5%FPR**|Chameleon|98_._0|52_._0|30_._0|
||Average|99_._0|62_._0|30_._0|



### **5.5 Towards Defending AIGI Detectors** 

Given the demonstrated vulnerabilities, we explore the potential of leveraging robustly pre-trained feature extractors as a defense mechanism. It is important to emphasize that our objective here is _not_ to present this as the _definitive_ or _optimal_ defense. Rather, our aim is to provide a preliminary showcase: to investigate whether established defense paradigms can offer a degree of robustness in this rapidly evolving landscape and to stimulate further research into tailored AIGI adversarial robustness. 

**Defense Method.** We adapt an existing defense based on robust feature extractors to AIGI detection. Our approach builds on RobustCLIP [39], a version of CLIP fine-tuned with unsupervised adversarial training to produce perturbationinvariant feature representations. This choice is motivated by two observations: (1) CLIP-based detectors (UnivFD, DRCT-CLIP) achieve strong detection and generalization performance, and (2) robust pre-trained backbones can transfer adversarial robustness to downstream tasks [25]. 

**Architecture and Training.** We replace the standard CLIP-ViT-L/14 backbone in UnivFD with the RobustCLIP backbone. The architecture remains otherwise identical: a frozen backbone followed by a single linear classification layer. We train only the linear head using the same training data (GenImage SD-1.4 subset) and hyperparameters as the original UnivFD model. The key difference from the baseline is the use of the adversarially-trained backbone, which provides robustness without requiring adversarial training of the detector itself. 

We evaluate two variants: **UnivFD-R2** and **UnivFD-R4** , which use RobustCLIP backbones trained with _ϵ_ = 2 _/_ 255 and _ϵ_ = 4 _/_ 255, respectively. Higher _ϵ_ during backbone training generally yields stronger robustness but may reduce clean performance due to the robustness-accuracy trade-off. 



Adversarial Robustness of AI-Generated Image Detectors 17 

Table 4: **Defense Performance.** AUROC, Accuracy, and TPR@5%FPR for robust models (UnivFD-R2, UnivFD-R4) compared to baseline UnivFD, averaged across Synthbuster, Chameleon, and GPT-4o. 

|**Attack**|**Target**|**A**|**URO**|**C**|**A**|**ccura**|**cy**|**TPR**|**@5%**|**FPR**|
|---|---|---|---|---|---|---|---|---|---|---|
|||**None **|**Med **|**High**|**None **|**Med **|**High**|**None **|**Med **|**High**|
||UnivFD|91.4|89.5|89.7|80.9|77.7|82.7|65.9|60.9|66.4|
|**Clean**|UnivFD-R2|88.3|85.8|84.1|82.7|77.8|77.1|62.4|53.9|48.1|
||UnivFD-R4|84.4|82.6|81.1|77.7|76.3|74.9|54.4|51.1|45.3|
||UnivFD|0.1|34.4|58.6|2.4|41.5|53.9|0.0|3.7|11.6|
|**PGD**|UnivFD-R2|72.3|78.3|74.1|66.0|72.0|66.3|31.1|38.5|28.4|
||UnivFD-R4|76.5|78.1|74.9|69.9|70.2|67.3|33.3|39.2|25.7|
||UnivFD|0.0|51.5|63.7|0.6|52.5|54.7|0.0|8.1|14.3|
|**Ensemble**|UnivFD-R2|75.8|81.3|75.2|68.6|74.8|67.7|36.7|45.2|32.3|
||UnivFD-R4|78.1|79.3|75.3|70.5|71.4|67.3|37.9|39.2|26.0|
||UnivFD|0.1|6.0|47.9|3.1|19.4|50.3|0.0|0.0|6.1|
|**Diverse**|UnivFD-R2|60.5|75.7|72.8|56.9|68.3|65.2|16.9|38.7|30.3|
||UnivFD-R4|72.2|77.2|74.5|65.9|69.2|66.9|29.7|35.9|25.5|
||UnivFD|28.6|79.4|65.3|35.6|69.1|57.6|0.8|34.8|16.3|
|**Universal**|UnivFD-R2|77.3|82.3|75.4|68.3|72.9|67.1|36.1|45.6|31.1|
||UnivFD-R4|78.1|79.8|75.4|70.3|71.0|67.5|36.4|39.3|27.2|
||UnivFD|66.2|76.6|68.7|59.1|67.6|57.2|21.3|36.8|18.1|
|**Query**|UnivFD-R2|76.5|81.2|75.6|69.7|74.3|67.9|38.4|45.9|30.9|
||UnivFD-R4|78.2|79.4|75.6|70.8|71.9|67.6|37.5|40.3|28.0|



**Evaluating the Defended Models.** The defended models are evaluated against the same attacks and degradations used in previous experiments with _ϵ_ = 8 _/_ 255. Table 4 reports AUROC averaged across all datasets on defended and reference models. The defended models demonstrate improved robustness against all attacks: the AUROC score of the undefended model drops to nearly 0 for our strongest attack (Diverse Input) in the clean setting, while the defended models achieve AUROC scores of 68 _._ 9 (UnivFD-R2) and 75 _._ 7 (UnivFD-R4). This enhanced robustness comes with a slight decrease in clean AUROC, from 91.4 for the undefended UnivFD to 88.3 (R2) and 84.4 (R4). Similar improvements are observed for attacks combined with various levels of degradation; however, we do not nearly reach the same performance as in the clean case. The defended models achieve AUROC of 78.8 (R2) and 78.3 (R4), compared to 6.7 for the undefended UnivFD. Although UnivFD-R4 demonstrates a higher degree of robustness, detector deployers must consider the detector’s susceptibility to attacks and the inherent trade-off between robustness and clean performance. 

**Finding 4.** Replacing standard backbones with robustly pre-trained feature extractors can improve robustness against strong attacks, but comes with a 3.1–7.0 point drop in clean AUROC. 



18 S. Mavali, J. Ricker, D. Pape, A. Fischer, and L. Schönherr 

## **6 Discussion** 

Our results show that state-of-the-art AIGI detectors are highly sensitive to realistic degradations and adversarial perturbations, even when images remain visually convincing to humans. In this section, we discuss what these findings imply for how AIGI detection should be defined, evaluated, and deployed in practice, as well as the limitations of our research. 

**Detection vs. Generation Arms Race.** Preventing misuse of AIGI is an arms race between AI generation techniques and detection mechanisms. As soon as a new detection method is developed, generative techniques are devised to circumvent these detectors, propagating a continuous cycle of advancements on both sides. This dynamic nature makes developing persistent solutions for AIgenerated content detection challenging. In the future, methods must take into account that detection cannot be perfect, but we must also study the limitations of detection methods. Our work offers one step toward defining and evaluating realistic threat models and measuring their robustness against attacks. **Context-dependent Detection.** AI-generated content is becoming an integral part of everyday life, with its usage expected to rise significantly. As a result, auditing every AI-generated image may become impractical. Instead, the focus should shift toward detecting harmful AI-generated content to identify and mitigate associated risks. For instance, in social media, AI-generated images are more likely to be used in disinformation and scams. Therefore, detection methods should concentrate on the most critical scenarios and consider characteristics unique to these contexts. 

**Robustness-Accuracy Trade-off.** Enhancing forensic classifier robustness often reduces accuracy on clean data. Deploying these systems requires weighing this trade-off based on the likelihood of attacks. Balancing robustness and accuracy demands careful, context-aware evaluation. In some cases, utility may outweigh full AI-content detection (e.g., less interactive platforms), while others require catching every instance (e.g., news sites). Thus, classification needs must be accurately assessed to align performance with specific use cases. **Limitations.** Our study has several limitations. First, we focus on a subset of generators, datasets, and post-processing pipelines, so our evaluation should be viewed as covering representative but not exhaustive scenarios. Second, our commercial case study on HIVE is small-scale due to the web interface limitation. Although it illustrates a concrete instance of fragility in real-world systems, it does not provide a full characterization of real-world risk across detection platforms. Finally, our defense analysis is scoped to a limited set of detectors, so future work should compare a broader range of robustness techniques under the same threat models. 

## **7 Conclusion** 

Our evaluation reveals that state-of-the-art AIGI detectors remain highly vulnerable to adversarial attacks. Our findings indicate that adversarial examples 



Adversarial Robustness of AI-Generated Image Detectors 

19 

transfer across different detectors, posing a significant threat to forensic detection mechanisms. Moreover, real-world degradations, such as those induced by social media post-processing, do not render well-chosen attacks ineffective. 

Our work is one step toward defining and evaluating realistic threat models, and we argue that future research must accurately assess classification needs to align detector performance with specific, critical use cases. Our results on using robust features show that while the robustness of detectors can be improved, they do not achieve the same performance as in clean settings. We argue that to prevent the misuse of GenAI, the field requires clear definitions, better risk assessments, and alternative solutions that consider the context of misuse. 

## **8 Ethical Considerations** 

This research is dual-use, as our attack methods could be misused to evade detectors. We practiced responsible disclosure by informing HIVE before publishing. We believe technical solutions alone are insufficient; a broader strategy integrating technology with education and policy is essential. 

## **9 Use of Generative AI** 

GitHub Copilot assisted in coding and debugging experimental scripts. LLMs helped with polishing the manuscript for clarity. All outputs were independently verified by the authors. The research used existing third-party datasets of AIgenerated images. 

## **Acknowledgments** 

This work was supported by the German Federal Ministry of Education and Research under the grants AIgenCY (16KIS2012) and SisWiss (16KIS2330). Moreover, it was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy - EXC 2092 CASA - 390781972, the LCIS center VW-Vorab-2025, ZN4704 11-76251-2055, as well as the Daimler and Benz Foundation under the grant Ladenburger Kolleg, Project KonCheck. 

## **References** 

1. Abdullah, S.M., Cheruvu, A., Kanchi, S., Chung, T., Gao, P., Jadliwala, M., Viswanath, B.: An analysis of recent advances in deepfake image detection in an evolving threat landscape. In: IEEE Symposium on Security and Privacy (S&P) (2024) 

2. Bammey, Q.: Synthbuster: Towards detection of diffusion model generated images. IEEE Open Journal of Signal Processing (2023) 

3. Black Forest Labs: FLUX (2024), `https://github.com/black-forest-labs/flux` 



- 20 S. Mavali, J. Ricker, D. Pape, A. Fischer, and L. Schönherr 

4. Carlini, N., Farid, H.: Evading deepfake-image detectors with white- and black-box attacks. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) (2020) 

5. Chen, B., Zeng, J., Yang, J., Yang, R.: DRCT: Diffusion reconstruction contrastive training towards universal detection of diffusion generated images. In: International Conference on Machine Learning (ICML) (2024) 

6. Coalition for Content Provenance and Authenticity (C2PA): C2PA technical specification (2024), `https://c2pa.org/specifications/specifications/2.0/ specs/C2PA_Specification.html` 

7. Corvi, R., Cozzolino, D., Zingarini, G., Poggi, G., Nagano, K., Verdoliva, L.: On the detection of synthetic images generated by diffusion models. In: IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (2023) 

8. Cozzolino, D., Poggi, G., Corvi, R., Nießner, M., Verdoliva, L.: Raising the bar of AI-generated image detection with CLIP. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops (2024) 

9. Dang-Nguyen, D.T., Pasquini, C., Conotter, V., Boato, G.: RAISE: A raw images dataset for digital image forensics. In: ACM Multimedia Systems Conference (2015) 

10. Ding, K., Ma, K., Wang, S., Simoncelli, E.P.: Image quality assessment: Unifying structure and texture similarity. IEEE Transactions on Pattern Analysis and Machine Intelligence (2022) 

11. DiResta, R., Goldstein, J.A.: How spammers and scammers leverage AI-generated images on Facebook for audience growth. Harvard Kennedy School Misinformation Review (2024) 

12. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (ICLR) (2020) 

13. Drolsbach, C., Pröllochs, N.: Characterizing AI-generated misinformation on social media. arXiv preprint arXiv:2505.10266 (2025) 

14. Eddoubi, H., Ricker, J., Cocchi, F., Baraldi, L., Sotgiu, A., Pintor, M., Cornia, M., Baraldi, L., Fischer, A., Cucchiara, R., Biggio, B.: RAID: A dataset for testing the adversarial robustness of AI-generated image detectors. arXiv preprint arXiv:2506.03988 (2025) 

15. Federal Bureau of Investigation (FBI): Cryptocurrency and AI scams bilk americans of billions (2026), `https://www.fbi.gov/news/press-releases/ cryptocurrency-and-ai-scams-bilk-americans-of-billions` 

16. Fei, J., Xia, Z., Tondi, B., Barni, M.: Supervised GAN watermarking for intellectual property protection. In: IEEE International Workshop on Information Forensics and Security (WIFS) (2022) 

17. Fernandez, P., Couairon, G., Jégou, H., Douze, M., Furon, T.: The stable signature: Rooting watermarks in latent diffusion models. In: IEEE/CVF International Conference on Computer Vision (ICCV) (2023) 

18. Frank, J., Herbert, F., Ricker, J., Schönherr, L., Eisenhofer, T., Fischer, A., Dürmuth, M., Holz, T.: A representative study on human detection of artificially generated media across countries. In: IEEE Symposium on Security and Privacy (S&P) (2024) 

19. Gragnaniello, D., Cozzolino, D., Marra, F., Poggi, G., Verdoliva, L.: Are GAN generated images easy to detect? a critical analysis of the state-of-the-art. In: IEEE International Conference on Multimedia and Expo (ICME) (2021) 



- Adversarial Robustness of AI-Generated Image Detectors 21 

- 20. Hart, R., Ricker, T.: Google’s Nano Banana Pro generates excellent conspiracy fuel (2025), `https://www.theverge.com/report/826003/ googles-nano-banana-pro-generates-excellent-conspiracy-fuel` 

- 21. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2016) 

- 22. HIVE: HIVE moderation (2025), `https://hivemoderation.com/` 23. Hou, Y., Guo, Q., Huang, Y., Xie, X., Ma, L., Zhao, J.: Evading deepfake detectors via adversarial statistical consistency. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023) 

24. Hu, Y., Jiang, Z., Guo, M., Gong, N.Z.: A transfer attack to image watermarks. In: International Conference on Learning Representations (ICLR) (2025) 

25. Hua, A., Gu, J., Xue, Z., Carlini, N., Wong, E., Qin, Y.: Initialization matters for adversarial transfer learning. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024) 

26. Ilyas, A., Engstrom, L., Athalye, A., Lin, J.: Black-box adversarial attacks with limited queries and information. In: International Conference on Machine Learning (ICML) (2018) 

27. Jiang, Z., Zhang, J., Gong, N.Z.: Evading watermark based detection of AIgenerated content. In: ACM SIGSAC Conference on Computer and Communications Security (CCS) (2023) 

28. Li, D., Wang, W., Fan, H., Dong, J.: Exploring adversarial fake images on face manifold. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021) 

29. Liu, C., Chen, H., Zhu, T., Zhang, J., Zhou, W.: Making deepfakes more spurious: Evading deep face forgery detection via trace removal attack. IEEE Transactions on Dependable and Secure Computing (2023) 

30. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., Vladu, A.: Towards deep learning models resistant to adversarial attacks. In: International Conference on Learning Representations (ICLR) (2018) 

31. Marra, F., Gragnaniello, D., Cozzolino, D., Verdoliva, L.: Detection of GANgenerated fake images over social networks. In: IEEE Conference on Multimedia Information Processing and Retrieval (MIPR) (2018) 

32. Midjourney: Midjourney (2024), `https://www.midjourney.com/home` 

33. Moosavi-Dezfooli, S.M., Fawzi, A., Fawzi, O., Frossard, P.: Universal adversarial perturbations. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2017) 

34. Ojha, U., Li, Y., Lee, Y.J.: Towards universal fake image detectors that generalize across generative models. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023) 

35. Park, S., Vilesov, A., Zhang, J., Khalili, H., Tian, Y., Kadambi, A., Sehatbakhsh, N.: Chimera: Creating digitally signed fake photos by fooling image recapture and deepfake detectors. In: USENIX Security Symposium (2025) 

36. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning (ICML) (2021) 

37. Ricker, J., Assenmacher, D., Holz, T., Fischer, A., Quiring, E.: AI-generated faces in the real world: A large-scale case study of Twitter profile images. In: International Symposium on Research in Attacks, Intrusions and Defenses (RAID) (2024) 



- 22 S. Mavali, J. Ricker, D. Pape, A. Fischer, and L. Schönherr 

38. Ryan-Mosley, T.: How generative AI is boosting the spread of disinformation and propaganda (2023), `https://www.technologyreview.com/2023/10/04/1080801/ generative-ai-boosting-disinformation-and-propaganda-freedom-house/` 

39. Schlarmann, C., Singh, N.D., Croce, F., Hein, M.: Robust CLIP: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models. In: International Conference on Machine Learning (ICML) (2024) 

40. Stability AI: Stable Diffusion (2022), `https://stability.ai/stable-image` 

41. Tariang, D., Corvi, R., Cozzolino, D., Poggi, G., Nagano, K., Verdoliva, L.: Synthetic image verification in the era of generative artificial intelligence: What works and what isn’t there yet. IEEE Security and Privacy (2024) 

42. Tramèr, F., Kurakin, A., Papernot, N., Goodfellow, I., Boneh, D., McDaniel, P.: Ensemble adversarial training: Attacks and defenses. In: International Conference on Learning Representations (ICLR) (2018) 

43. Wang, S.Y., Wang, O., Zhang, R., Owens, A., Efros, A.A.: CNN-generated images are surprisingly easy to spot... for now. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2020) 

44. Xie, C., Zhang, Z., Zhou, Y., Bai, S., Wang, J., Ren, Z., Yuille, A.L.: Improving transferability of adversarial examples with input diversity. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2019) 

45. Xu, H., Wang, Y., Wang, Z., Ba, Z., Liu, W., Jin, L., Weng, H., Wei, T., Ren, K.: ProFake: Detecting deepfakes in the wild against quality degradation with progressive quality-adaptive learning. In: ACM SIGSAC Conference on Computer and Communications Security (CCS) (2024) 

46. Yan, S., Li, O., Cai, J., Hao, Y., Jiang, X., Hu, Y., Xie, W.: A sanity check for AI-generated image detection. In: International Conference on Learning Representations (ICLR) (2025) 

47. Yan, Z., Ye, J., Li, W., Huang, Z., Yuan, S., He, X., Lin, K., He, J., He, C., Yuan, L.: GPT-ImgEval: A comprehensive benchmark for diagnosing GPT-4o in image generation. arXiv preprint arXiv:2504.02782 (2025) 

48. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2018) 

49. Zhu, M., Chen, H., Yan, Q., Huang, X., Lin, G., Li, W., Tu, Z., Hu, H., Hu, J., Wang, Y.: GenImage: A million-scale benchmark for detecting AI-generated image. Advances in Neural Information Processing Systems (NeurIPS) (2023) 



Adversarial Robustness of AI-Generated Image Detectors 23 

## **Appendix** 

Table 5: **Detectors’ Robustness.** AUROC, Accuracy, and TPR@5%FPR averaged for each datasets for clean (no attack) and different attacks with varying degradation levels (None, Medium: 75% downsampling and JPEG compression at quality 50, and High: Gaussian blur (5 _×_ 5, _µ_ = 1, _σ_ = 1), 50% downsampling, JPEG compression at quality 20, and Gaussian noise ( _µ_ = 1, _σ_ = 0 _._ 1)). “LTO” means Leave-Target-Out: attacks are optimized on all detectors except the evaluated target. 

|||||**Syn**|**thbuster**|||**Ch**|**ameleon**|||**G**|**PT-4o**||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**Attack**|**Target**|**Source**|**Clean**||**Attacked**||**Clean**||**Attacked**||**Clean**||**Attacked**||
|||||**No Deg.**|**Med Deg.**|**High Deg.**||**No Deg.**|**Med Deg.**|**High Deg.**||**No Deg.**|**Med Deg.**|**High Deg.**|
|||DRCT-CLIP|95_._9|15_._9|57_._2|62_._2|79_._2|21_._8|52_._2|63_._9|99_._2|16_._4|75_._2|66_._0|
||UnivFD|DRCT-ConvB|95_._9|76_._8|81_._9|67_._1|79_._2|81_._4|79_._3|68_._7|99_._2|83_._4|94_._1|69_._1|
|||Corvi|95_._9|76_._2|80_._3|71_._3|79_._2|73_._7|72_._6|68_._8|99_._2|82_._6|92_._5|70_._7|
|||UnivFD|97_._3|8_._7|45_._4|60_._8|70_._7|8_._8|43_._6|70_._4|95_._0|5_._0|26_._5|46_._0|
||DRCT-CLIP|DRCT-ConvB|97_._3|64_._6|73_._3|62_._2|70_._7|70_._2|77_._3|75_._8|95_._0|52_._0|55_._0|49_._4|
|||Corvi|97_._3|72_._8|78_._1|64_._9|70_._7|67_._8|73_._9|74_._7|95_._0|50_._3|63_._0|50_._5|
|PGD||UnivFD|98_._8|75_._6|61_._0|53_._2|69_._6|33_._3|32_._5|41_._1|88_._5|66_._6|67_._3|46_._6|
||DRCT-ConvB|DRCT-CLIP|98_._8|24_._8|50_._2|55_._2|69_._6|5_._6|28_._2|40_._2|88_._5|18_._8|56_._9|45_._3|
|||Corvi|98_._8|1_._4|6_._9|48_._0|69_._6|0_._0|2_._4|40_._0|88_._5|0_._7|7_._2|40_._5|
|||UnivFD|100_._0|95_._5|82_._4|57_._2|63_._0|27_._3|25_._3|33_._2|93_._3|67_._1|54_._3|57_._3|
||Corvi|DRCT-CLIP|100_._0|94_._0|83_._9|58_._0|63_._0|26_._2|29_._9|37_._4|93_._3|62_._5|59_._9|57_._3|
|||DRCT-ConvB|100_._0|96_._3|86_._6|62_._6|63_._0|49_._8|42_._3|38_._5|93_._3|83_._0|73_._7|59_._6|
||Average|–|98_._0|58_._6|65_._6|60_._2|70_._6|38_._8|46_._6|54_._4|94_._0|49_._0|60_._5|54_._9|
||UnivFD|LTO|95_._9|0_._0|45_._6|64_._7|79_._2|0_._0|43_._4|61_._8|99_._2|0_._0|65_._6|64_._5|
|Enmbl|DRCT-CLIP|LTO|97_._3|0_._9|50_._5|62_._6|70_._7|0_._9|54_._6|72_._0|95_._0|0_._1|31_._5|46_._7|
|see|DRCT-ConvB|LTO|98_._8|8_._1|26_._8|50_._2|69_._6|0_._9|6_._8|36_._4|88_._5|6_._3|29_._5|44_._8|
||Corvi|LTO|100_._0|84_._6|76_._3|54_._1|63_._0|19_._3|28_._2|34_._9|93_._3|47_._0|53_._0|54_._8|
||Average|–|98_._0|23_._4|49_._8|57_._9|70_._6|5_._3|33_._3|51_._3|94_._0|13_._4|44_._9|52_._7|
||UnivFD|LTO|95_._9|0_._0|6_._4|51_._7|79_._2|0_._1|3_._7|43_._9|99_._2|0_._1|8_._0|48_._2|
|Di|DRCT-CLIP|LTO|97_._3|0_._6|7_._6|52_._9|70_._7|0_._6|9_._3|59_._6|95_._0|0_._1|4_._0|37_._3|
|verse|DRCT-ConvB|LTO|98_._8|1_._5|7_._6|43_._0|69_._6|0_._1|2_._0|28_._3|88_._5|2_._7|12_._8|39_._7|
||Corvi|LTO|100_._0|14_._4|23_._4|43_._7|63_._0|0_._4|4_._8|24_._8|93_._3|3_._1|14_._8|49_._1|
||Average|–|98_._0|4_._1|11_._3|47_._8|70_._6|0_._3|5_._0|39_._2|94_._0|1_._5|9_._9|43_._6|
||UnivFD|LTO|95_._9|20_._8|74_._0|63_._6|79_._2|31_._5|76_._8|69_._3|99_._2|33_._7|87_._2|62_._9|
||DRCT-CLIP|LTO|97_._3|56_._9|64_._8|61_._6|70_._7|49_._9|74_._6|74_._3|95_._0|39_._0|57_._4|47_._6|
|Universal|DRCT-ConvB|LTO|98_._8|23_._9|24_._4|51_._4|69_._6|5_._8|18_._3|38_._6|88_._5|9_._3|26_._6|46_._4|
||Corvi|LTO|100_._0|85_._8|65_._2|53_._4|63_._0|22_._3|34_._6|37_._6|93_._3|62_._0|60_._5|54_._7|
||Average|–|98_._0|46_._9|57_._1|57_._5|70_._6|27_._4|51_._1|55_._0|94_._0|36_._0|57_._9|52_._9|
||UnivFD|–|95_._9|64_._1|74_._7|66_._9|79_._2|60_._4|64_._2|69_._0|99_._2|74_._1|91_._0|70_._3|
||DRCT-CLIP|–|97_._3|79_._0|75_._9|64_._1|70_._7|61_._0|72_._1|75_._7|95_._0|72_._9|60_._5|48_._1|
|Query|DRCT-ConvB|–|98_._8|77_._2|63_._1|54_._8|69_._6|33_._5|35_._2|42_._3|88_._5|63_._4|64_._8|47_._9|
||Corvi|–|100_._0|98_._5|86_._0|51_._9|63_._0|24_._6|31_._4|37_._2|93_._3|70_._4|55_._0|57_._0|
||Average|–|98_._0|79_._7|74_._9|59_._4|70_._6|44_._9|50_._7|56_._1|94_._0|70_._2|67_._8|55_._8|





24 S. Mavali, J. Ricker, D. Pape, A. Fischer, and L. Schönherr 

Table 6: PGD step convergence analysis. Post-attack AUROC (%) averaged over three target detectors in black-box setting ( _ϵ_ = 8 _/_ 255, _α_ = 2 _/_ 255). Lower values indicate more effective attacks. The negligible differences across step counts confirm that ten steps is sufficient for convergence. 

|**Source**|**5 steps **|**10 steps **|**20 steps**|
|---|---|---|---|
|Corvi|83.4|84.3|83.7|
|DRCT-CLIP|49.8|49.1|48.5|
|DRCT-ConvB|91.2|91.6|91.8|
|UnivFD|99.7|99.2|99.1|
|**Average**|**81.0**|**81.0**|**80.8**|





<!-- Start of picture text -->
Original PGD Ensemble Diverse Universal Query<br><!-- End of picture text -->

Fig. 4: Adversarial examples generated for target UnivFD with _ϵ_ = 8 _/_ 255. Images are from the Chameleon dataset. 

