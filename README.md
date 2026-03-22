# Natural Language Satisfiability: Exploring the Problem Distribution and Evaluating Transformer-based Language Models (Polish Adaptation)

Efforts to apply transformer-based language models (TLMs) to the problem of reasoning in natural language have enjoyed ever-increasing success in recent years. The most fundamental task in this area to which nearly all others can be reduced is that of determining satisfiability. However, from a logical point of view, satisfiability problems vary along various dimensions, which may affect TLMs' ability to learn how to solve them. 

The problem instances of satisfiability in natural language can belong to different computational complexity classes depending on the language fragment in which they are expressed. Although prior research has explored the problem of natural language satisfiability, the above-mentioned point has not been discussed adequately. Hence, we investigate how problem instances from varying computational complexity classes and having different grammatical constructs impact TLMs' ability to learn rules of inference. Furthermore, to faithfully evaluate TLMs, we conduct an empirical study to explore the distribution of satisfiability problems.

### 🇵🇱 Polish Language Adaptation

**This specific project is a Polish language adaptation of the original natural language satisfiability framework.** Translating logical syllogisms and generated formulas from English to Polish introduces significant linguistic challenges that this project successfully resolves. Unlike English, Polish is a highly inflected language with complex grammatical rules. Key features implemented in this adaptation include:
* **Dynamic Inflection (Fleksja):** The sentence generation engine utilizes a custom lexicon dictionary to correctly decline nouns and verbs. It automatically applies the appropriate grammatical cases (e.g., Nominative for subjects, Instrumental after the verb "to be", Accusative for direct objects).
* **Double Negation Handling:** The templates and logic translation have been entirely rewritten to support native Polish double negation structures (e.g., translating "No X is Y" to the grammatically correct "Żaden X **nie** jest Y") without breaking the underlying Z3 solver logic.
* **Accurate Agreement:** Proper morphological agreement between quantifiers, nouns, and verbs depending on the sentence context (singular/plural, affirmative/negative).

### Data Construction

We construct data by sampling from the `phase-change region` (the region around which the probability of satisfiability is around 0.5). The updated pipeline dynamically inflects the Polish vocabulary during this generation process. Refer to the visualization of how the probability of satisfiability varies with the number of unary and binary predicates in the original study.

To generate a dataset in Polish, run the following command:

```bash
python data_construction.py \
    --fragment <FRAGMENT> \
    --max_a <maximum number of unary predicates> \
    --min_a <minimum number of unary predicates> \
    --max_b <maximum number of binary predicates> \
    --min_b <minimum number of binary predicates> \
    --sampling_file <link to the sampling file> \
    --max_ab <max prob(sat)> \
    --min_ab <min prob(sat)> \
    --time_out <time out for the sat solver> \
    --prob <probability for complex fragment compared to simple ones> \
    --num_datapoints 4000 \
    --output_file "data-construction/fragment_pl.csv"
```


### Acknowledgments & Credits

The core logical framework, data sampling methodology, and baseline code utilized in this project were originally developed by iTharindu. 

This repository builds upon their foundational work by introducing a rigorous linguistic adaptation for the Polish language. We highly encourage exploring their original paper and repository: 
https://github.com/iTharindu/nl-sat.git