# Untitled

Mainly from **[LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives?utm_source=profile&utm_medium=reader2)**

# Reinforcement learning

Reinforcement learning is a [machine learning](https://www.techtarget.com/searchenterpriseai/definition/machine-learning-ML) training method based on rewarding desired behaviors and punishing undesired ones. In general, a reinforcement learning agent -- the entity being trained -- is able to perceive and interpret its environment, take actions and learn through trial and error.

# Supervised fine tuning cons:

models are incentivized to place probability mass on all human demonstrations, including those that are low-quality; and distributional shift during sampling can degrade performance

# RLHF goal

The essential goal here is to make a conventional large language model (GPT-3 in our case) align with human principles or preferences. This makes our LLMs less toxic, more truthful, and less biased.

# Steps:

<p float="left">
  <img src="https://lh7-us.googleusercontent.com/l1jDFQpeR6iJ6_iPvxW7Nu12R_GxXZJxaEhJr-P1zavdGdRIroMWfCzz6OwAgzhaUXPLk6TAeohXamraHcEDsrEUEgIE0HF_hYdmU1I4MW6GcgoSaf0rhUDMSNbrZciBaUZhv6Ub3-Pg_AgJkHfO7Q" width="85%" />
</p>


1. Pretraining a language model (LM),
    1. fine tune with preference data set or supervised learning
    2. supervised fine tuning uses a smaller dataset
2. gathering data and training a reward model, and
    1. These LMs for reward modeling can be both another fine-tuned LM or a LM trained from scratch on the preference data.
        1. fine-tuned LM’s output layer (the next-token classification layer) is substituted with a regression layer, which features a single output node.
    2. sample a set of prompts from a predefined dataset
    3. pass through the initial language model to generate new text.
    4. for each prompt, we generate four to nine responses from the finetuned LLM created in the prior step. An individual then ranks these responses based on their preference.
    5. Human annotators are used to rank the generated text outputs from the LM
        1. There are multiple methods for ranking the text. One method that has been successful is to have users compare generated text from two language models conditioned on the same prompt. By comparing model outputs in head-to-head matchups, an **[Elo](https://en.wikipedia.org/wiki/Elo_rating_system)** system can be used to generate a ranking of the models and outputs relative to each-other. These different methods of ranking are normalized into a scalar reward signal for training.
    6. An interesting artifact of this process is that the successful RLHF systems to date have used reward language models with varying sizes relative to the text generation (e.g. OpenAI 175B LM, 6B reward model, Anthropic used LM and reward models from 10B to 52B, DeepMind uses 70B Chinchilla models for both LM and reward).
    7. An intuition would be that these preference models need to have similar capacity to understand the text given to them as a model would need in order to generate said text.
3. fine-tuning the LM with reinforcement learning.
    1. fine-tuning some or all of the parameters of a **copy of the initial LM** with a policy-gradient RL algorithm, Proximal Policy Optimization (PPO)
    2. Some parameters of the LM are frozen

# InstructGPT stats

- Pretrain: 100B - 5T tokens
- Supervised Finetuning: 1k - 50k instruction response pairs
- RLHF > 50k examples

# RLHF in Llama 2

<p float="left">
  <img src="https://lh7-us.googleusercontent.com/BYu0JyjAr-jw3N2Xlkr86fPK9yNV0bcLFupruYvNtmfvnQ2vsO5iqVTV0hzq5DooCemlf-qTI-yGfilE9C7DD6UBx0h6vU5WQTMy9GOPZF_aYAMC0gmKScP6mLy6DZoJ5Db_OLVfeK9b0BU-jxNRyg" width="85%" />
</p>


- 2 reward model
    - Helpfulness
    - Harmlessness
- Different human ranking method
    - InstructGPT ask human labelers to rank 4 responses at a time
    - LLama 2 only presents 2 responses for ranking but an additional "margin" label (ranging from "significantly better" to "negligibly better") is gathered
- Different ranking loss function to train reward model
    - InstructGPT loss
        
<p float="left">
  <img src="https://lh7-us.googleusercontent.com/qc7-9VVAho-zLwhcKkK921U-MGYfv4WSKb1CpZgsWN06OLUwu09ZzTWL5gWb7LZkR4rjjXU9iyANvwg_unOAjwTDcrBV7yEq7PxXQ0LiMTR9DfL3CsG-ADhsvOs0dMffXeAAQy-sadQhHzL4T5x9ew" width="85%" />
</p>

        
    - Llama2 loss
        
<p float="left">
  <img src="https://lh7-us.googleusercontent.com/sJ-hH4JRP2Vj74McEfHVK2c-gyOULFtHmUXgRVF_rxlOaoMJCIUb9TEvT9RNKCCW2WwdXJqxaLOgsS7Sa6rZkIk4mwE69W6y8yPFQkU0y0PrMxvYNV5KEsFTB4APJjsdWvWp62IdbzIo0kux65JNkA" width="85%" />
</p>


        
    - Llama 2 added the the margin “m(r)” as a discrete function of the preference rating as follows:
    - returning a higher margin via “m(r)” will make the difference between the reward of the preferred and rejected responses smaller, resulting in a larger loss, which in turn results in larger gradients, and consequently model changes, during the policy gradient update.
- 2 RLHF stages
    - Rejection sampling
        - K outputs are drawn, and the one with the highest reward is chosen for the gradient update during the optimization step
    - PPO stage

# Proximal Policy Optimization Algorithms

<p float="left">
  <img src="https://lh7-us.googleusercontent.com/wX_XXniLq33c8SuNDngLK30cf9bQXYKuIKkhcTmJJrTLoA6Yu4Wd0V3GJ9qx-QHNWrsFYgrCrVC_3ccpEixOU06-4fJ85ZnzjLS2mnbT6edrcoN-xUYoVYJ6om23FzH95hL_ErbozPICCu8tt7dySw" width="85%" />
</p>

<p float="left">
  <img src="https://lh7-us.googleusercontent.com/Szr2DVOe-LgEkeAuOWNlgdA1a4WByBrNjtMc8peBowZV7SwjJl2RMcmeI-pkIwzuNpwo_oCB4_2xjUASuXBp-C1JZBnphXP8ioyBggF2d78_Ie3AyMPl8roha29WKkmB5RATFm24t6WRbVHNKCDVVw" width="85%" />
</p>


1. Given a prompt, *x*, from the dataset, the text *y* is generated by the current iteration of the fine-tuned policy.
2. Concatenated with the original prompt, that text is passed to the preference model, which returns a scalar notion of “preferability”, *rθ*.
3. In addition, per-token probability distributions from the RL policy are compared to the ones from the initial model to compute a penalty on the difference between them.
    1. In multiple papers from OpenAI, Anthropic, and DeepMind, this penalty has been designed as a scaled version of the Kullback–Leibler **[(KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)** between these sequences of distributions over tokens, Rkl.
    2. The KL divergence term penalizes the RL policy from moving substantially away from the initial pretrained model with each training batch, which can be useful to make sure the model outputs reasonably coherent text snippets.
    3. Without this penalty the optimization can start to generate text that is gibberish but fools the reward model to give a high reward.
    4. In practice, the KL divergence is approximated via sampling from both distributions (explained by John Schulman **[here](http://joschu.net/blog/kl-approx.html)**). The final reward sent to the RL update rule is *r*=*rθ*−*λ*Rkl.
        1. This KL term serves two purposes. First, it acts as an entropy bonus, encouraging the policy to explore and deterring it from collapsing to a single mode.
        2. Second, it ensures the policy doesn’t learn to produce outputs that are too different from those that the reward model has seen during training.
4. the **update rule** is the parameter update from PPO that maximizes the reward metrics in the current batch of data (PPO is on-policy, which means the parameters are only updated with the current batch of prompt-generation pairs). PPO is a trust region optimization algorithm that uses constraints on the gradient to ensure the update step does not destabilize the learning process.

## Results for RLHF with PPO:

- better than supervised learning in summarizing reddit posts and news articles
- over optimize the reward model hurt the true preference on llm output
- doubling the training data amount leads to a ~1.1% increase in the reward model validation set accuracy, whereas doubling the model size leads to a ~1.8% increase
- our reward models are sensitive to small but semantically important details in the summary.
- our learned reward models consistently outperform other metrics such as ROUGE, summary length, amount of copying from the post, and log probability under our baseline supervised models.

<p float="left">
  <img src="https://lh7-us.googleusercontent.com/Szr2DVOe-LgEkeAuOWNlgdA1a4WByBrNjtMc8peBowZV7SwjJl2RMcmeI-pkIwzuNpwo_oCB4_2xjUASuXBp-C1JZBnphXP8ioyBggF2d78_Ie3AyMPl8roha29WKkmB5RATFm24t6WRbVHNKCDVVw" width="85%" />
</p>

[https://lh7-us.googleusercontent.com/xgppLpR9JDz4H2BxgsJaKq4_htB5AobPIsGvhPnPgr8q_oWZvC2RvFAtfCiGjdamKf8j1be9YsqYnWqqxuHJoV9hFH3D_k9XdutGj0p_TnO8w5R-RTDvLPu49gZsubLC2tg1FYy9doGbwsIZanZ5JQ](https://lh7-us.googleusercontent.com/xgppLpR9JDz4H2BxgsJaKq4_htB5AobPIsGvhPnPgr8q_oWZvC2RvFAtfCiGjdamKf8j1be9YsqYnWqqxuHJoV9hFH3D_k9XdutGj0p_TnO8w5R-RTDvLPu49gZsubLC2tg1FYy9doGbwsIZanZ5JQ)

# Direct Preference Optimization:

**RLHF cons**

- More complex
- High computational cost

In this paper, we show how to directly optimize a language model to adhere to human preferences, without explicit reward modeling or reinforcement learning.

**Given a dataset of human preferences over model responses, DPO can therefore optimize a policy using a simple binary cross entropy objective, producing the optimal policy to an implicit reward function fit to the preference data.**

our key insight is to leverage an analytical mapping from reward functions to optimal policies, which enables us to transform a loss function over reward functions into a loss function over policies. This change-of-variables approach avoids fitting an explicit, standalone reward model, while still optimizing under existing models of human preferences, such as the Bradley-Terry model. In essence, the policy network represents both the language model and the (implicit) reward.

<p float="left">
  <img src="hhttps://lh7-us.googleusercontent.com/W0aqCJp3zI8s_39CMCem6mhzC1E1sGUz6GU-TgnBbT_yEv75y0sWe870HTh9e4nPm9pN9vtkYtvLrSSdbysl1N38CaKSBpNlU34pFC_zFhQctXozxI7ET86WBax-NgjSbJ4Rc6c3YZG2zRDCg_FYXg" width="85%" />
</p>


<p float="left">
  <img src="https://lh7-us.googleusercontent.com/aTkCQp46SNx4jZTPNhEgXPzP2A94bTAi9neXoTvaJYuSghHJtS9fhMiwJa63mV81jOtkPAnqr-HDKvpliHUcVNJOhmPgLlibZM5HSo7IVYPWmzhDzUIxn3ICJ-wvfHK87MEBR7GeQq2CgR_Znrb0zQ" width="85%" />
</p>


<p float="left">
  <img src="https://lh7-us.googleusercontent.com/SORnicnjirknemkmyQQTJWP1oBJ9ZBBtJX-cnRfSqk9DcloO0EK0p4nPUKAFkChFFyJ9DU-mMj33cqwbnlGp618No-MHeFE9GJNnxaeo4WzdpM_Qj40E_NzAWX5X-vD-mrmbn2ADWaBP4bMql1-iDA" width="85%" />
</p>


<p float="left">
  <img src="https://lh7-us.googleusercontent.com/I6z2wLpxNtKGjAnY-gCGmMBnotzbPH2C6_brCWXwv2O1wQVSMD91gtrpOTq3Upm2EjwCtjGn3HW1AyJmqbklMH3u-pj-NhQaVnwdxu_xI3guxDIvksGz9b5r_44bdErFUnHnJ4y5AOPVdFuIUTJfZA" width="85%" />
</p>


**Experiments and Results**

- DPO converges to its best performance relatively quickly.
- DPO policies can generalize similarly well to PPO policies, even though DPO does not use the additional unlabeled Reddit TL;DR prompts that PPO uses.
- These experiments are judged by GPT-4 e.g. GPT-4 decides if a completion is better than human written summaries
- And another experiment, We find that with both prompts, GPT-4 tends to agree with humans about as often as humans agree with each other, suggesting that GPT-4 is a reasonable proxy for human evaluations

# RLAIF

## Constitutional AI: Harmlessness from AI Feedback (Dec 2022, [https://arxiv.org/abs/2212.08073](https://arxiv.org/abs/2212.08073))

<p float="left">
  <img src="https://lh7-us.googleusercontent.com/PJjNkqXeO3EhhFWs7P_blC1Ee9bcRpwm1rXqOYywBYHVv1brgiNQC2vZXpb1u7emJmpPesqx2b19q4UdYiQTDP0bU6NBYbTwMeEUFQcfhlaZxwdLvXLvp9gh9K3Roa4Mb6WCF0OAyLaNCAk3eJFP-w" width="85%" />
</p>


<p float="left">
  <img src="https://lh7-us.googleusercontent.com/T20i1U7PiNbG_r15idPhINkyz2u56RYBKFaw6I2qridFFB_uLmFA_8_2e37Ld1sC5o3pX2XFAJfPDF7JPX_m5gtYUztVcQclVx5TIvqyxJc6O413WE71aaV6lyE0DPLUwxYntjxQiFyvQK6neZxkPQ" width="85%" />
</p>


## **RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback** (Sep 2023, [https://arxiv.org/abs/2309.00267](https://arxiv.org/abs/2309.00267))

The main contributions of this work are as follows:

1. We demonstrate that RLAIF achieves comparable or superior performance to RLHF on the tasks of summarization, helpful dialogue generation, and harmless dialogue generation.
2. We show that RLAIF can improve upon a SFT policy even when the LLM labeler is the same size as the policy.
3. We find that directly prompting the LLM for reward scores during RL can outperform the canonical setup where a reward model is trained on LLM preferences.
4. We compare various techniques for generating AI labels and identify optimal settings for RLAIF practitioners.
    1. use chain of thoughts prompting and few shot prompting

<p float="left">
  <img src="https://lh7-us.googleusercontent.com/1LvFfVO7PDGUsWsLQ2IWaWrm81FJFyABwBPbEBmQ7YXtDnFNvOAzlKVRIkPC2icl4uX4e50R1KKBysAccTWHq27F-yS8a0JI6r1sn52DDXN-eXUqJowYZfNX00_pCMGvt0q6Pq0Jj1TDsYQlZHloCw" width="85%" />
</p>


<p float="left">
  <img src="https://lh7-us.googleusercontent.com/jaJpKl2Z6Ih4J02eVEyryFxtTan6jbX7F7GoOmmVuNibDyL4MLALTtN_aOsAeKMxUF7rcQ9cYeyusHEpiJliuhLrrfzclOebpmN0sIHDjfIXremWaK35FnPf28HZcZdyjdxNgZW91jKJpxsPoPq0mw" width="85%" />
</p>


2 approaches:

1. Distilled RLAIF: produces soft labels (e.g. [0.6, 0.4]), and train a reward model on it
2. Direct RLAIF: ask LLM model to rate from 1 - 10

Evaluation:

- AI Labeler Alignment measures the accuracy of AI-labeled preferences with respect to human preferences.
- Win Rate evaluates the end-to-end quality of two policies by measuring how often one policy is preferred by human annotators over another.
- Harmless Rate measures the percentage of responses that are considered harmless by human evaluators

Results

- RLAIF achieves performance gains on par with or better than RLHF on all three tasks
- One natural question that arises is whether there is value in combining human and AI feedback. We experimented with combining both types of feedback but did not see an improvement beyond using human feedback alone.
- RLAIF can yield improvements even when the AI labeler model is the same size (in terms number of params ) as the policy LLM.
    - We note that the AI labeler and initial policy are not the exact same model.
- Direct RLAIF performs better than Distilled RLAIF
    - One hypothesis for the improved quality is that bypassing the distillation from AI preferences into a RM enables information to flow directly from the off-the-shelf LLM to the policy.

<p float="left">
  <img src="https://lh7-us.googleusercontent.com/RIZBRKAs4RhESxE9To-SG8F-JuicvGf2Z-g2OwikecosjqWL3fU2DAGOdDOrGI8k5Wtocb-gWOtbCpqjYdOZHnE5Rz5l3_xI60EY09UYjZM_CZ8iTgKuK2Nb06mKgXCeIcNtL7YB5YDf6u2AIXMqHw" width="85%" />
</p>


- We observe that eliciting chain-of-thought reasoning generally improves AI labeler alignment, while the impacts of preamble specificity and in-context learning vary across tasks
- We also conduct experiments with selfconsistency (Wang et al., 2022b), where multiple chain-of-thought rationales are sampled with temperature T > 0. The preference distributions generated by the LLM are averaged together to arrive at the final preference label. We find that **selfconsistency** strictly degrades AI labeler alignment

<p float="left">
  <img src="https://lh7-us.googleusercontent.com/GFYhzI5N0zKdy7RrkzUXnLQ_kxRQ3kKjX44YmGutyNbjoCgmbXQtRzPdRfOY2EAUgZbL51zvi8GmWoK0sNOD33A-S5dwG8Ag-LN8lxVVldcLGGCNf6ZDTfsdOlFhsqVqJCcPViGndAtccBprqXMrhw" width="85%" />
</p>


- Results show that the policy trained with more aligned AI labels achieves a significantly higher win rate.
- larger ai labeler model size leads to better ai labeler alignment and produce even higher quality preference labels.
    - Since the AI labeler is only used to generate preference examples once and is not called during RL, using an even larger AI labeler is not necessarily prohibitively expensive.

# Other Optimization options

## The Wisdom of Hindsight Makes Language Models Better Instruction Followers

<p float="left">
  <img src="https://lh7-us.googleusercontent.com/s3C97Uq4i2tJI_MYUwsDPJSstvvfLCGVSwOluvX20UMGkjyXWYBuc5hs_d8L2t0uu7AqMnpmUJ82le-fgElhFuzI8Hn_l9Uu9UVs5FrYwfzTldIIh72N8pp3MWXAbm-YwQOQEpgXhBm760O2Xl6_Ag" width="85%" />
</p>


**Contrastive Preference Learning: Learning from Human Feedback without RL** (Oct 2023, [https://arxiv.org/abs/2310.13639](https://arxiv.org/abs/2310.13639))

Similar to DPO but used in robotics environment

**(5) Reinforced Self-Training (ReST) for Language Modeling** (Aug 2023, [https://arxiv.org/abs/2308.08998](https://arxiv.org/abs/2308.08998))

<p float="left">
  <img src="https://lh7-us.googleusercontent.com/2UNCcW0sYgqqo3or8qLAFwjaYdKK8ytdeGj1TgVVL05Kj8sjV5NK3Ek04uUCz4dQ2nreNpzZiPe_FOkQQdGSzik30-09v-TtNtINGZWDBk5HXgVkY-_j6g8PpB0AaSARI7tQ_qnxQeX9urk3TSXghA" width="85%" />
</p>


# References

- **[An Introduction to Training LLMs Using Reinforcement Learning From Human Feedback (RLHF)](https://wandb.ai/ayush-thakur/Intro-RLAIF/reports/An-Introduction-to-Training-LLMs-Using-Reinforcement-Learning-From-Human-Feedback-RLHF---VmlldzozMzYyNjcy)**
- **[Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)**
- **[LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives?utm_source=profile&utm_medium=reader2)**
- **Learning to summarize from human feedback** [https://arxiv.org/pdf/2009.01325.pdf](https://arxiv.org/pdf/2009.01325.pdf)
- Proximal Policy Optimization Algorithms [https://arxiv.org/pdf/1707.06347.pdf](https://arxiv.org/pdf/1707.06347.pdf)
- [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/pdf/2309.00267.pdf)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf)