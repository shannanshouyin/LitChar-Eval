# Study protocol

The study used the public PersoNet release at commit 34745ded1ce168dcfb5ac7e5ca06fe364d16ff04. The following details accompany the main article.

We used PersoNet [5], with the public release fixed at repository commit 34745ded1ce168dcfb5ac7e5ca06fe364d16ff04. A development pilot on 50 source groups produced 600 planned responses across two models and six prompts to check collection and scoring. The reported estimates use the test split. After inspecting the main results, we recorded the matched stability and deletion protocols before collecting their respective follow-ups. Figure 1 summarizes the sequence.

The test file contained 3,624 records from ten books. We excluded 86 with duplicate options after case and whitespace normalization and one with empty history. The remaining 3,537 records formed 2,530 groups defined by book, character, and target passage. We selected one record per sampled group by deterministic hash ranking. Allocation favored equal book representation within the available group counts, yielding 600 groups with 23–72 per book (Appendix A).

Each input contained the character name, five candidate traits, and passages immediately before, at, and after the annotated location. Three conditions also included the released history passage, a short excerpt of earlier context. We removed anchor markup, normalized whitespace, and retained the corpus’s unknown-token marker. An exact overlap of at least 12 whitespace tokens between the end of history and the start of local context was trimmed from history. A 1,200-word history cap was specified but inactive for the selected inputs.

The original record key can encode the reference trait. We therefore constructed API inputs from an explicit list of permitted fields: character, options, and passages. Book titles, source keys, record indices, and reference labels remained outside the messages. Options were shuffled deterministically within each group, with the same order across prompt conditions. We used the models for inference only; neither was trained, fine-tuned, or selected using PersoNet test outcomes.

Table 1 lists the six conditions. A and D requested a choice only; B and E requested analysis followed by a choice. C and F additionally requested up to two exact quotations, each no longer than 25 whitespace-separated words and linked to a passage identifier. Analysis and quotations together were limited to 120 words. An empty quotation list was allowed when the model found no suitable quotation. C minus B and F minus E compare the quotation request at fixed context and analysis settings.

GLM-4.6 was accessed through Zhipu’s official BigModel application programming interface (API), with identifier glm-4.6 and base URL https://open.bigmodel.cn/api/paas/v4/ [21]. DeepSeek flash was accessed through https://api.deepseek.com with identifier deepseek-flash. Official documentation checked on 23 October 2026 mapped this alias to DeepSeek-V4.1-Flash-1017 [22]. We retained returned model identifiers and UTC timestamps.

Requests used temperature 0, top_p 1, a maximum of 1,024 output tokens, JavaScript Object Notation (JSON)-object response mode, and disabled provider thinking mode. Each request ran independently, without external search or tools. Main collection ran on 22 October 2026 from 06:05 to 12:01 UTC; matched stability collection ran on 23 October from 01:55 to 02:51 UTC; deletion collection ran on 23 October from 03:31 to 03:53 UTC. The client timezone was Asia/Shanghai, and BigModel used its mainland-China endpoint. The providers did not expose exact serving datacenters or a dated GLM weight snapshot.

Requests were interleaved within each phase, with up to two simultaneous requests per provider. Transient failures were retried once. Malformed responses, refusals, and truncated outputs were retained under the scoring rules. Authentication or billing failures halted collection. The logs preserve source and input hashes, request bodies, and terminal outcomes.

A valid response satisfied the JSON schema and contained one permitted option letter, which we mapped back to its trait before comparing judgments. Strict reference agreement was one for a valid answer matching the public label and zero otherwise, including malformed or unavailable outputs. Prompt-length and quotation adherence were recorded separately from structural validity.

A quotation was traceable when it matched a continuous substring of its declared passage after normalizing whitespace and straight and curly quotation marks. Case and other punctuation were preserved. A response was traceable if it contained at least one such quotation of at most 25 words. We cross-tabulated this measure of textual occurrence with reference agreement.

For paired comparisons, we averaged item differences within each book and then weighted the ten book effects equally. The main primary family comprised C minus B and F minus E for each model. We report unadjusted 95% confidence intervals (CIs) based on the t distribution over book effects and two-sided p-values from exact enumeration of sign flips, with Holm adjustment across the four comparisons [23,24]. The sign-flip test assumes independent, symmetric book effects. Intervals and adjusted p-values use different constructions and may give different threshold decisions.

The ten books represent seven authors, including multiple works by Dostoyevsky and Hugo. We examined leave-one-book and leave-one-author estimates. For the follow-up primary effects, an exploratory analysis averaged book effects within each author and then weighted the seven authors equally, using the same interval and sign-flip calculations. This sensitivity analysis changes both the target average and the assumed independent unit.

A fixed hash ranking selected 12 groups per book from the main sample, giving 120 groups independently of their response outcomes. Each model received E and F in three fresh versions: an anchor, an identical repeat, and a cyclic option rotation. Nonzero shifts of one to four positions were balanced within books. The passages and candidate traits stayed fixed. All 1,440 responses were collected in the same follow-up period.

For each valid triple, we recorded whether the repeat and rotation selected a different trait from the anchor. The primary effect was rotation-induced change minus repeat-induced change in F, averaged equally across books. The two model effects formed a Holm-adjusted family. E and the F-minus-E contrast were secondary comparisons. We reported valid-triple coverage and calculated worst-case bounds over all 120 groups by allowing each missing paired difference to range from minus one to plus one. An additional operational measure counted invalid outputs as failures to reproduce the anchor.

We also examined three fixed selection rules: a traceable anchor; three valid answers choosing the same trait; and both together. For each rule, we report coverage, agreement among retained anchors, and expected agreement under random retention with the same number selected per book. This descriptive analysis follows the coverage–performance trade-off used in selective classification [25]. Stability-based selection requires three responses per case.

The deletion experiment used F and the same 120 groups, with a fresh complete-text anchor for each model. An eligible anchor was structurally valid and contained one or two quotations, all at most 25 words and uniquely located in their declared passages. We removed every whitespace token intersecting a quoted span, merging adjacent or overlapping spans. For each merged span, a control of equal token length was selected by fixed hash ranking from the same passage. Controls excluded cited tokens and previously selected control spans. Anchors lacking a suitable control were excluded. Eligibility was independent of reference-label agreement.

Edited inputs retained the character, options, remaining passages, and prompt. Tokens were omitted without masking markers, and both edited versions were sent to the model that had produced the anchor. We compared changes from that anchor after cited and control deletion. The protocol specified common valid triples, equal-book aggregation, and planned Holm adjustment across the two model contrasts, with all ten books required for primary inference. We also report eligibility, missing-output bounds, and author-level sensitivity. Agreement with the intact-text reference after deletion is descriptive, since editing can alter the evidence for the trait.

We illustrate the latter pattern with the lexicographically first group identifier for each model among F rotation-change pairs sharing a traceable quotation.

Seventy-three GLM anchors and 39 DeepSeek anchors met the deletion rules, or 60.8% and 32.5% of 120 anchors per model. Of 128 excluded anchors, 83 lacked a disjoint same-passage control of the required length, 34 had an invalid quotation, eight had absent or ambiguous matches, and three were structurally invalid. Median removal sizes were 23 words for GLM and 29 for DeepSeek. GLM supplied 70 valid triples across ten books: its book-averaged change rates were 36.1% after cited deletion and 11.5% after control deletion, a difference of 24.7 percentage points (95% CI 12.9 to 36.4; nominal, unadjusted p = 0.0078). DeepSeek supplied 39 valid triples across nine books, with rates of 41.8% and 14.2%, a descriptive difference of 27.6 points. No DeepSeek anchor from The Idiot qualified, leaving the prescribed ten-book inference and the two-model Holm family incomplete (Table 5; Figure 4).

At the item level, GLM changed its trait in 27 of 70 cited-deletion responses and eight control responses; DeepSeek changed in 13 of 39 and seven, respectively. The resulting micro-averaged differences were 27.1 and 15.4 points. DeepSeek’s larger book-average estimate reflects the equal weighting of books with sparse eligible samples. Worst-case assignments of GLM’s three missing triples gave an eligible-sample bound of 20.5–27.6 points. Exploratory equal-author differences were 24.8 points for GLM and 33.7 for DeepSeek, with Holm-adjusted p-values of 0.0625 for both.

## Software
The collection and analysis environment used Python 3.13.13, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, and Matplotlib 3.11.2. The full recorded environment is in requirements-original.txt.

## References

Reference numbers follow the accompanying article.

5. Yu, M.; Li, J.; Yao, S.; Pang, W.; Zhou, X.; Xiao, Z.; Meng, F.; Zhou, J. Personality understanding of fictional characters during book reading. In Proceedings of the Annual Meeting of the Association for Computational Linguistics. 2023; pp. 14784–14802. https://doi.org/10.18653/v1/2023.acl-long.826

21. Zhipu AI. GLM-4.6 model and API documentation. Available online: https://docs.bigmodel.cn/cn/guide/models/text/glm-4.6 (accessed on 23 October 2026).

22. DeepSeek. Models and pricing. Available online: https://api-docs.deepseek.com/zh-cn/quick_start/pricing/ (accessed on 23 October 2026).

23. Dror, R.; Baumer, G.; Shlomov, S.; Reichart, R. The hitchhiker’s guide to testing statistical significance in natural language processing. In Proceedings of the Annual Meeting of the Association for Computational Linguistics. 2018; pp. 1383–1392. https://doi.org/10.18653/v1/P18-1128

24. Holm, S. A simple sequentially rejective multiple test procedure. Scand. J. Stat. 1979, 6, 65–70. https://www.jstor.org/stable/4615733

25. Geifman, Y.; El-Yaniv, R. Selective classification for deep neural networks. Adv. Neural Inf. Process. Syst. 2017, 30. https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html
