# Project Proposal: LLM-Based Structurally Focused Topic Modeling

## 1. Research Question and Hypothesis

We propose to investigate whether large language models (LLMs) can improve topic modeling by extracting structured information from text before applying topic models. The central research question is: Can LLM-driven structured extraction enhance Latent Dirichlet Allocation (LDA) topic modeling of natural language data, yielding more meaningful or insightful topics than conventional approaches?

We hypothesize that incorporating structure will reveal latent themes tied to the data's organization. For example, in mental health discussions, we expect that an LLM can identify cognitive distortions in posts, and that running LDA on these structured outputs will produce clearer thematic groupings (e.g., types of negative thinking or coping themes) than running LDA on raw text.

## 2. Literature Review

**Key Study:**
[Talbot et al. (2023)](https://github.com/Hramir/educational_concept_librarian) demonstrated the combination of LLMs with topic modeling by analyzing educational YouTube videos. They used GPT to extract structured knowledge graphs of concepts from video transcripts, then applied LDA to identify themes—revealing that popular videos tended to introduce core concepts early and reinforce them with supporting ideas.

**Related Work:**

- **Wang et al. (2023):** Introduced _PromptTopic_, where LLMs generate fine-grained topics at the sentence level.
- **Singh et al. (2023):** Developed a framework to detect cognitive distortions in patient-therapist conversations.
- **Traditional LDA:** Often used in mental health text analysis but typically requires careful preprocessing for interpretable results.

## 3. Dataset Selection

We will use mental health discussions (e.g., posts from Reddit communities like r/depression or r/Anxiety, or therapy conversation transcripts). This domain naturally contains structural elements (e.g., cognitive distortion patterns, therapy dialogue structure) that align well with our methodology. The structured schema (e.g., issues, distortions, emotions, coping strategies) allows us to test hypotheses such as whether certain cognitive distortion patterns co-occur with specific discussion topics.

## 4. LLM and Topic Modeling Methodology

**Chosen LLM:**
We will use [OpenAI GPT-4](https://openai.com/research/gpt-4) for structured extraction because of its strong language understanding and instruction following. Alternatives include GPT-3.5 Turbo (more cost-effective) or open-source models like [LLaMA-2](https://ai.meta.com/llama/) if budget constraints require.

**Topic Modeling Technique:**
We will use **Latent Dirichlet Allocation (LDA)** because:

- It was used in the key study, facilitating comparative analysis.
- It is interpretable and well-supported by existing implementations.
- It has been applied to mental health data in previous research, providing benchmarks.

## 5. Project Plan and Experiments

### 5.1 Structured Data Extraction with LLM

- **Objective:** Transform unstructured posts into a semi-structured format.
- **Method:** Design prompt templates for GPT-4 to extract:
  - Main issue or concern
  - Cognitive distortions (if any)
  - Emotion or tone expressed
  - Coping strategies mentioned

### 5.2 Topic Modeling with LDA

- **Preprocessing:** Concatenate key fields into a single text per post and perform standard text cleaning (e.g., lowercasing, stopword removal).
- **Modeling:** Run LDA with different numbers of topics (K), using coherence analysis to select the optimal K.
- **Expected Output:** Topics that reflect themes like “social anxiety and fear of judgment” or “health anxiety and physical symptoms.”

### 5.3 Baseline and Comparative Experiments

- **Baseline:** Run LDA on raw text.
- **Comparisons:** Evaluate LDA on LLM-structured text versus raw text and simpler preprocessing methods.
- **Ablation:** Assess which structured fields contribute most to topic quality.

### 5.4 Process Flowchart

**Pipeline:** Raw text → LLM structured extraction → Preprocessed structured summaries → LDA topic modeling → Topic interpretation and evaluation.

### 5.5 Implementation Details

1. **Initial Testing:** Refine LLM prompts on a small subset.
2. **Scaling:** Apply extraction to the full dataset.
3. **LDA Runs:** Execute LDA with various topic counts.
4. **Analysis:** Interpret topics in the context of our hypothesis.

## 6. Evaluation Metrics and Benchmarks

- **Topic Coherence:** Use metrics (e.g., Cv, UCI) to assess the semantic relatedness of topic words.
- **Topic Diversity:** Measure the uniqueness of top words across topics.
- **Baseline Comparison:** Compare LDA on raw text vs. LLM-structured text.
- **Human Evaluation:** Qualitative assessment of topic interpretability.
- **Hypothesis Testing:** Verify if posts with specific cognitive distortions cluster into distinct topics.
- **Performance:** Monitor LLM extraction consistency and alignment with external labels where available.

## 7. Computational Feasibility

- **Resources:**
  - **Academic Node:** 8 CPU cores, 32GB RAM, 24GB GPU VRAM.
  - **MacBook M3 Max:** 16 cores, ample unified memory, 40 GPU cores.
- **Data Processing & LLM Inference:**
  - GPT-4 API shifts heavy computation off-site; parallel API calls will be managed on the academic node.
  - Local model alternatives can be used with quantization techniques to fit available GPU VRAM.
- **LDA Training:**
  - Efficient with 32GB RAM and multi-core CPUs; estimated runtime is manageable (e.g., ~1 hour for 1000 posts with GPT-4, minutes for LDA).

## 8. Potential Challenges and Mitigation Strategies

- **LLM Output Consistency:**
  - Use refined, structured prompts (e.g., JSON format), set temperature=0, and implement output validation.
- **LLM Biases and Domain Knowledge:**
  - Supply clear definitions in prompts and manually review samples; adjust categories as needed.
- **Dataset Quality and Noise:**
  - Filter irrelevant or very short posts; instruct the LLM to perform relevance checks.
- **Choosing Number of Topics:**
  - Use coherence score cross-validation and manual verification to determine optimal K.
- **Computational/Rate Limits:**
  - Batch process data, cache results, and implement resumable extraction.
- **Ethical and Privacy Considerations:**
  - Use only public data, strip identifiers, and instruct the LLM to avoid sensitive info.
- **Evaluation Subjectivity:**
  - Conduct blind evaluations and use predefined quantitative metrics.
- **Overfitting to Structure:**
  - Design a comprehensive schema and compare with alternative extraction approaches.
