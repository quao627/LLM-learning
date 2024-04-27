# SPIN: Self-Play Fine-Tuning

UCLA-AGI has proven that large language models, even weaker large language models, can improve themselves with data only produced by original model. The question they answer in their paper is:

"Can we empower a weak LLM to improve itself without acquiring additional human annotated data?"

- Self-Play: Eg: AlphaGo Zero
- Basic Setting:
    - Input: LLM, **Supervised Dataset**, regularization parameter: λ
    - Output: A fine tuned LLM which output more closely to the input dataset.
    
    # Method
    
    - Iterative method:
        - We have LLM_0, LLM_1, LLM_2,...,LLM_n
        - The parameter of LLM_t is **p_t**
        - For LLM_i, assume we have a **function based on the parameter** **of LLM_t** that can predict how likely an output is from training data. f_t (input, output). The parameter for f_t is from p_t
        - Now, given LLM_t, we want f_(t+1) from LLM_(t+1) can tell if an output is from training data (the bigger, the better) **AND** LLM_(t+1) cannot be too far away from LLM_t (control by λ)
        - Magically! Setting f_i (input, output) as
            - We can accomplish two optimization task at one time:
                - Maximize the ability to tell the difference between real data and output of LLM_t
                - Move LLM_t towards better LLM under the dataset while consider regularization function λ

![Screenshot 2024-04-27 at 10.18.25 AM.png](SPIN%20Self-Play%20Fine-Tuning%2044465575538e4ed6b5e89ac78443e2a9/Screenshot_2024-04-27_at_10.18.25_AM.png)

**Experiment**

1. Model and Datasets. 

In this study, we adopt zephyr-7b-sft-full as our base model. This model derives from the pre-trained Mistral-7B and has been further fine-tuned on the SFT dataset Ultrachat200k1 by HuggingFace. Ultrachat200k represents a high-quality 200k subset of the larger UltraChat corpus, which comprises approximately 1.4M dialogues produced using OpenAI’s Turbo APIs. From UltraChat200k, We randomly sample 50k prompts and use the base model to generate the synthetic responses. In multiple iterations, we leverage the synthetic data from the most recent iteration and add to the newly generated synthetic data, therefore resulting in a synthetic dataset size of 50k at iteration 0 and 100k at iteration 1, 2 and 3. At each iteration, we train our model for 2 epochs.

1. zephyr-7b-sft-full
- from Mistral-7B,further fine-tuned on the SFT dataset Ultrachat200k (1 epoch)
- 3 iterations, each iteration 2 epochs

**performance**

![Screenshot 2024-04-27 at 10.19.14 AM.png](SPIN%20Self-Play%20Fine-Tuning%2044465575538e4ed6b5e89ac78443e2a9/Screenshot_2024-04-27_at_10.19.14_AM.png)