# Code Repository for AIA

## Brief Intro

- We introduce the **Acoustic Interference Attack (AIA)**, a novel jailbreak paradigm that bypasses the safety alignment of Large Audio Language Models (LALMs) using benign, instruction-neutral audio. By leveraging intrinsic Acoustic Latent Semantics (ALS) rather than embedding malicious payloads, AIA achieves *SOTA universal attack success rates without requiring instance-specific optimization*.

- Please see our project page [here](https://flaai.github.io/AIA_page) for more detailed information.

## Code Structure

- ```hierarchical_acoustic_arsenal```: The whole process to capture the preliminary arsenal with sampled ALS patterns, corresponding to *Figure 2* of our paper.

- ```exploration```: The whole process to build the final arsenal with effective ALS, corresponding to *Figure 3* of our paper.

- ```attack_practice```: The main attack loop of the proposed AIA method, corresponding to *Figure 4* of our paper.
