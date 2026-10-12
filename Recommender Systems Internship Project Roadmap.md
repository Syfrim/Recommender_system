Build a multimodal search and recommendation engine that recommends fashion products, retrieves visually similar items and answers product searches. Then adapt its data and evaluation components to a short-video ranking experiment. This is designed for your goal of a For You Page recommendation internship and uses your existing computer vision experience.

The intended outcome is that you can take an unfamiliar recommendation problem, define its objective, inspect its data, build a baseline, train retrieval and ranking models, find failures, test an improvement and explain the trade-offs. Company-specific infrastructure, logging policies and product objectives still require onboarding and supervision. Completing this project gives you practical preparation for that work.

Plan for approximately 16 weeks at 12–16 hours per week. This is a planning estimate, not a deadline; allow extra time if PyTorch training or SQL is new to you. Weeks 1–12 form a working portfolio system with clearly labeled validation results. Weeks 13–16 extend it toward your specific feed recommendation goal and finish the evidence for applications. If time is limited, ship the core before adding models.

All model sizes, cutoffs and time budgets below are starting design choices. Record the actual dataset size, hardware, training time and measured results. No improvement is assumed in advance.

## What you will build

| Surface | Input | Output | What it teaches |
| - | - | - | - |
| Personalized product feed | User history and request time | Ten ranked products | User modeling, candidate retrieval and ranking |
| Similar items | A product or permitted query image | Visually similar products | Image embeddings, vector search and cold-start features |
| Product search | A text query and optional user context | Relevant ranked products | Lexical retrieval, semantic retrieval and relevance evaluation |
| Video ranking experiment | Historical context and an exposed video | Long-view and like predictions | Real feed feedback, multiple objectives and exposure bias |

The product feed is a shopping recommender with a feed interface. Its purchase labels do not make it a video engagement benchmark. The KuaiRand extension supplies the separate video experiment. Use shared interfaces and utilities across these tasks, but train appropriate models and evaluate each task against its own labels.

The recommendation path is: construct the request context; retrieve a few hundred candidates; assemble features available at that time; score candidates; apply simple diversity and eligibility rules; return results and log the decision.

Begin with 100 candidates and ten final recommendations. Later test 50, 100 and 300 candidates. These are experimental settings, not claims about a particular company's architecture.

This architecture aligns with the retrieval, ranking, reranking, experimentation and system work described in [TikTok's Core Feed Recommendation role](https://lifeattiktok.com/search/7520493060184066312). [Meta's explanation of Instagram Explore](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/) is a useful example of how those stages work together.

## Concepts to learn through the project

| Concept | Working meaning | Evidence that you understand it |
| - | - | - |
| Implicit feedback | An observed action suggests interest; absence of an action has several explanations | You can explain why an unpurchased item is not a confirmed dislike |
| Collaborative filtering | Learn from patterns across users and items | You implement an item co-occurrence baseline and an ID embedding model |
| Content features | Describe the item using text, images or attributes | You measure whether images help beyond text and interaction history |
| Retrieval | Find a small useful candidate set efficiently | You measure target coverage before ranking |
| Ranking | Order retrieved candidates for a particular request | You train a model on groups of candidates |
| Reranking | Adjust the final list for diversity or other constraints | You show a quality–diversity trade-off |
| Temporal leakage | Future information accidentally helps predict the past | Your features and model snapshots obey time cutoffs |
| Exposure bias | The existing system influences what users have a chance to interact with | You distinguish observed feedback from universal preferences |
| Cold start | A user or item has little useful interaction history | You evaluate sparse users and a disclosed item holdout |
| Online experimentation | Randomly compare policies with real users | You can design an experiment while keeping offline results separate |

For each phase, keep a short learning note containing: the problem, your prediction, the change, the measurement and your explanation. Being able to defend those decisions matters more than listing many libraries.

## Tools and working structure

Use Python and PyTorch as the main implementation language and framework. Use pandas or Polars for smaller tables, DuckDB and Parquet for filtering larger logs, NumPy and SciPy for numerical work, LightGBM for the first learned ranker, and FAISS for vector retrieval. Use a frozen CLIP-compatible model through OpenCLIP for image and text features. FastAPI and a simple Streamlit interface are sufficient for the demo.

Use one environment and pin package versions after a working installation. Start on CPU. A modest GPU is helpful for image encoding and neural training, but keep the image encoder frozen and cache its outputs. Run a small benchmark before renting compute. Do not build distributed training, Kubernetes, a feature-store service or a custom vector database for this project.

A sensible initial development scale is 2,000 users for debugging, followed by roughly 20,000 users selected using training-period information. The number of products and interactions must come from the data audit. Keep the whole eligible catalog where practical; around 100,000 vectors is manageable without distributed infrastructure. Encode only the needed images. A 100,000-by-512 float32 array is about 205 MB before index overhead.

Create these components as you need them:

| Location | Responsibility |
| - | - |
| `src/data/` | Load source data, build schemas, sample users and create splits |
| `src/features/` | Historical aggregates, image features and text features |
| `src/retrieval/` | Popularity, item co-occurrence, embedding retrieval and candidate unions |
| `src/ranking/` | Candidate features, ranker training and reranking |
| `src/evaluation/` | Metrics, temporal replay, slices and uncertainty estimates |
| `src/serving/` | API, model loading, fallbacks and event logging |
| `app/` | Small interactive demonstration |
| `configs/` | Dataset, split, model and experiment settings |
| `tests/` | Metric, time-cutoff and data-contract checks |
| `reports/` | Experiment results, error analysis and final report |

Keep source data and large checkpoints outside public Git history. Publish reproducible download/preprocessing instructions subject to dataset terms. Maintain a small synthetic fixture for tests and CI. Use notebooks to investigate, then move reusable transformations into functions.

## Week 1 Set up and define the problem

**Purpose:** establish enough Python, ML and SQL fluency to understand the rest of the system.

1. Create the repository and environment. Add a README describing the intended recommendation task in five sentences.

2. Check that you can load a CSV, join two tables by ID, group by user and date, and save Parquet. Write a SQL query for each user's number of historical purchases and a second query for weekly product popularity.

3. In PyTorch, train a tiny classifier and confirm that you understand tensors, batches, the forward pass, a loss, gradients, an optimizer, evaluation mode and checkpoint saving. If this is unfamiliar, spend an additional preparation week here.

4. Work through a five-user, eight-item interaction example by hand. Recommend popular items, then items bought by people with overlapping histories. Explain why these two methods produce different lists.

5. Write the primary task: at the start of a week, rank products that a returning user may purchase during the following seven days.

6. Write down the available signals and the desired outcome. A purchase prediction is the target; whether someone clicked or saw a product is unknown in this dataset.

7. Create an experiment log with columns for hypothesis, dataset version, split, candidate policy, model settings, metrics, runtime and decision.

**Deliverables:** repository skeleton, a tiny reproducible PyTorch exercise, two SQL queries and a one-page task specification.

**Pass the gate when:** you can explain the difference between predicting a scalar rating, retrieving candidates and ranking a list. You can run your exercise from a script rather than relying on hidden notebook state.

## Week 2 Acquire and audit the data

**Purpose:** understand what the records mean before using them as supervision.

Use the official [H&M Personalized Fashion Recommendations data](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data). It includes transactions, product metadata and garment images. Check account access and applicable dataset rules first. Start with the CSVs and a small image subset; the full image collection is much larger than the tables. A Kaggle notebook can be useful for computing selected image features without copying the entire collection to your laptop.

1. Inspect `transactions_train.csv`, `articles.csv` and the image directories. Load IDs as strings. Preserve the article ID representation needed to match image filenames.

2. Build an interaction schema with `user_id`, `item_id`, `event_date`, `event_type` and optional historical transaction fields. Here, the event type is purchase.

3. Build an item table with product name, description, product type, category, color and image path. Initially omit customer demographics; the project does not need them.

4. Keep raw rows unchanged. For the ranking target, derive a binary user–item–day table and keep purchase counts separately. Multiple identical transaction rows may represent multiple units; do not silently declare them corrupt.

5. Audit row counts, date range, missing IDs, missing images, unmatched joins, unique users/items, purchases per user, buyers per item and weekly activity.

6. Inspect at least 30 joined examples manually: transaction, product description and image. Confirm that IDs align.

7. Select an initial observation interval of roughly three to six months. Choose debugging and final user cohorts deterministically from the retrieval-training period, such as a seeded hash sample. Keep their histories within the observation interval. Avoid selecting users because they happen to have many future purchases.

8. Retain sparse users and report them separately. If you restrict the main experiment to users with prior history, name it a returning-user benchmark.

9. Convert the filtered data to Parquet and save a manifest containing the sampling seed, source files, observation dates and actual counts.

**Deliverables:** a data card, validated joined tables, a reproducible sampling script and basic distribution plots.

**Pass the gate when:** you can trace a recommendation example to its original transaction and image, explain the sparsity of your data, and state which user population your benchmark covers.

**Source detail:** the [PyTorch Geometric H&M loader](https://pytorch-geometric.readthedocs.io/en/2.6.0/_modules/torch_geometric/datasets/hm.html) provides a useful independent reference for the transaction schema. You do not need to install PyTorch Geometric.

## Week 3 Build evaluation before training complex models

**Purpose:** create a benchmark that measures generalization rather than accidental access to future information.

Use the final twelve complete weeks of your chosen observation interval for an initial chronological design. Verify dates in the actual files before writing them into a configuration.

| Block | Suggested duration | Permitted use |
| - | - | - |
| A | Eight weeks | Fit retrieval models and define the initial catalog/cohort |
| B | Two weeks | Build weekly candidate groups and train the ranker |
| C | One week | Select hyperparameters and policies |
| D | One week | Final held-out evaluation after choices are frozen |

Earlier observations may provide history if you include them explicitly in the configuration. Retrieval models for B must have been trained only on A or earlier. For the first version, freeze those retrieval weights through B, C and D. Historical user features can update at each request using events strictly before it. The ranker is trained on B, selected using C and evaluated on D. This simple design avoids the extra complexity of rebuilding model snapshots for every week.

1. Define a request as `(user_id, week_start)`. Construct history using dates strictly before `week_start`. Its relevant set is the unique items purchased in the following seven days.

2. H&M has daily transaction dates. Treat purchases on the same date as unordered. Do not invent a within-day sequence or use part of the target day as input.

3. Define catalog eligibility. Initially use products observed before the end of A, and keep this catalog fixed across B–D. This is a controlled historical catalog, not verified stock availability. Report how many future purchased items fall outside it.

4. Keep repeat purchases eligible in the main benchmark. If you test suppression later, report its effect and keep the same targets.

5. Build popularity, user summaries and price features using only information available before the request. The target transaction's price and sales channel are not known input features.

6. Avoid mutable customer snapshots whose historical availability is unclear. Treat product descriptions/images as catalog features under an explicit static-metadata assumption.

7. Implement Recall\@100, NDCG\@10 and MRR\@10 on a hand-checkable toy example before evaluating models.

8. Use all eligible catalog items for retrieval evaluation. If compute is limited, reduce the number of evaluation users with a fixed sample; avoid making the task artificially easy by comparing a positive against a few random negatives.

9. Report the fraction of requests with no future positive. Ranking metrics may be calculated on requests with at least one eligible positive, but explicitly state that denominator. Track all-target coverage as well so out-of-catalog purchases remain visible.

10. Write checks that changing an event after a request cannot change that request's historical features. Check that target events never enter their own history. A genuinely earlier purchase of the same item is allowed.

**Metrics in plain language:**

- `Recall@100 = relevant items among the first 100 candidates / all relevant eligible items`. This asks whether retrieval found what the ranker would need.

- With binary relevance, `DCG@10 = sum(relevance_at_rank / log2(rank + 1))`; NDCG divides this by the best possible DCG for that request.

- `MRR@10 = 1 / rank_of_first_relevant_item`, or zero if none appears in the first ten.

- If there is one relevant item and it appears at rank 3, Recall\@10 is 1, MRR\@10 is about 0.333 and NDCG\@10 is 0.5.

- End-to-end NDCG includes retrieval failures. Never insert the true target into evaluation candidates to make the ranker look better.

**Deliverables:** split configuration, request/target tables, metric implementations, tests and an evaluation script that accepts any recommender.

**Pass the gate when:** a hand-calculated example matches your code, future data cannot alter historical features, and every metric has a documented denominator. Read [On Sampled Metrics for Item Recommendation](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/) to understand why sampled-negative rankings can mislead.

## Week 4 Establish strong simple baselines

**Purpose:** learn what personalization contributes and create a comparison that is hard to fool.

1. Give each model the same interface: `retrieve(user_context, cutoff, k)` returns item IDs and scores. Define stable tie-breaking, unique results and a fallback for empty histories.

2. Implement random eligible items as a diagnostic baseline.

3. Implement global popularity and recent popularity, such as counts in the preceding 7 and 28 days. Choose the window using validation.

4. Implement category popularity conditioned on the user's historical category preferences.

5. Build item-to-item co-occurrence from training users. Normalize pair counts so universally popular items do not dominate every similarity. Retrieve neighbors of historical purchases and aggregate their scores.

6. Learn matrix factorization with a small ID embedding model: represent each user and each item with a vector and score their dot product. Start with 32 or 64 dimensions.

7. For an educational BPR version, sample a historical positive and an unobserved item, then minimize `-log(sigmoid(score_positive - score_unobserved))`. Explain that this ranking assumption does not establish a dislike label. Exclude known positives available to that training example from negative sampling.

8. Evaluate every method on C using the evaluator from Week 3. Record target Recall\@100, final NDCG\@10, coverage and runtime. Keep D untouched.

9. Manually inspect ten successes and ten failures of the strongest baseline. Categorize them: missing candidate, wrong category, too popular, repeat behavior or sparse history.

**Deliverables:** a baseline table, saved baseline outputs and an error notebook.

**Pass the gate when:** all methods use the same task, catalog and evaluation users, and you can explain why recent popularity may beat a learned model. A complex model is allowed to lose. Investigate the cause rather than changing the benchmark to manufacture a win.

## Week 5 Add a substantive computer vision component

**Purpose:** test whether visual information improves recommendations.

1. Load a small pretrained CLIP-compatible image/text encoder. Record its checkpoint, source, preprocessing and usage terms. Use the official [OpenCLIP implementation](https://github.com/mlfoundations/open_clip) as a reference.

2. Put the encoder in evaluation mode and freeze it. Encode item images in batches. Save `item_id → embedding`, the preprocessing version and a missing-image flag.

3. Encode product name, category and description with the paired text encoder. Normalize embeddings before using cosine similarity.

4. Build image-to-image nearest-neighbor search. Inspect 20 query products and their neighbors for wrong IDs, duplicates, background effects and category errors.

5. Build a content-based user representation by averaging embeddings of items in their historical purchases. Compare plain averaging with a recency-weighted average.

6. Evaluate image-only, text-only and combined content retrieval on the purchase task. Keep the candidate budget and splits identical.

7. Add graceful missing-content handling: use text when images are absent and a fallback representation when both are missing.

8. Write a hypothesis before inspecting results: for example, visual features may help sparse users or visually distinctive categories more than heavily purchased items.

**Deliverables:** cached features, a visual similarity demonstration and a content-retrieval experiment table.

**Pass the gate when:** images affect actual retrieval or ranking, and you have measured whether that contribution helps. A gallery alone is not evidence of recommendation quality.

Using frozen pretrained features demonstrates integration and evaluation of visual representations. Describe it that way in your CV. The [Sentence Transformers image-search guide](https://sbert.net/examples/sentence_transformer/applications/image-search/README.html) explains the shared text/image embedding idea.

## Weeks 6 and 7 Train retrieval and build the candidate pipeline

**Purpose:** learn how user preferences become an efficient candidate search.

1. Create a training example from a user's history before a purchase date and a product purchased on that date. Treat all purchases on the date as a target set or separate positives with the same historical context. Do not give the model an arbitrary same-day ordering.

2. Start with a small two-tower model. The user tower receives a user ID embedding, pooled historical item features and history length. The item tower receives an item ID embedding and optional text/image features. Small multilayer perceptrons map both to a common space, such as 64 or 128 dimensions.

3. Normalize the final vectors and score their dot product. Keep item features independent of the requesting user so item vectors can be precomputed.

4. First train an ID/history version. Then add text, image and combined content versions. This separates the benefit of model capacity from the benefit of each modality.

5. Use a batch contrastive objective: matched user–item pairs should score above other candidate items. Conceptually, for one positive, `loss = -log(exp(positive_score / temperature) / sum(exp(candidate_score / temperature)))`.

6. Handle repeated item IDs in a batch. Mask false-negative copies or implement a multi-positive loss; do not label the same product positive in one column and negative in another for the same example. Also avoid treating other known positives for that example as negatives.

7. Overfit a tiny batch as a debugging check, then train on A. Track training loss and a small internal chronological validation slice within A for early stopping; reserve C for the complete pipeline's model choices.

8. Retrieve exactly using batched matrix multiplication or a FAISS flat index. Compare target Recall\@100 against the best baseline. Inspect embedding norms, invalid values, ID alignment and missing-content cases.

9. Cache item vectors. Record the encoder version and item-order mapping with the index. Add a test that index IDs map back to the right products.

10. Combine candidate sources: two-tower results, item co-occurrence, content neighbors and recent popularity. Deduplicate and cap the union with a deterministic policy. Record the source(s) for every candidate.

11. Tune source quotas on C. Compare single-source and union retrieval at the same final candidate budget, so a larger candidate set does not explain the entire gain.

12. Only if exact search becomes a bottleneck, add an approximate index such as HNSW. Compare it with exact search on a fixed set of queries. Report latency and overlap with exact neighbors separately from target relevance recall.

**Deliverables:** a reproducible PyTorch training script, checkpoint, item-vector index, candidate-generation report and quality-versus-latency measurements.

**Pass the gate when:** you can retrieve for existing and unknown users, trace every candidate to its source, explain the training negatives, and identify whether retrieval is the main quality bottleneck.

Two different recalls must remain separate: target Recall\@100 measures recovered future purchases; ANN neighbor recall measures agreement with exact vector search. The [TensorFlow Recommenders retrieval tutorial](https://www.tensorflow.org/recommenders/examples/basic_retrieval) teaches the two-stage idea even if you implement it in PyTorch. The [FAISS getting-started guide](https://github.com/facebookresearch/faiss/wiki/Getting-started) covers the index interface.

## Week 8 Build and train a ranking model

**Purpose:** learn to combine several signals for each user–candidate pair.

1. Freeze your A-trained retrieval model. For each weekly request in B, construct context from earlier dates, generate candidates and calculate candidate features.

2. Create one row per request and candidate. Keep `request_id`, `user_id`, `cutoff`, `item_id`, features and the target label separate.

3. Set relevance to 1 if the candidate appears in the next seven days' purchase set and 0 otherwise. Those zeros are unobserved purchases under this task; they are not confirmed negative impressions.

4. Start with a small feature set: retrieval scores, candidate source flags, historical category affinity, historical item popularity, image/text similarity to the user's history, history length and time since the user's last purchase.

5. Optionally add historical price preference using only prices already observed before the request. Include missing-value indicators. Do not use future transaction fields or full-period aggregates.

6. Count groups with no retrieved positive. For the first ranker, train on naturally retrieved groups containing at least one positive and one other item; record how many groups were excluded. This selection affects the training population, so explain it. Preserve every eligible request in end-to-end evaluation.

7. Sort rows so each request's candidates are contiguous. Build a list of group sizes. A request is one user at one time; the same user can have multiple groups.

8. Train `LGBMRanker` with a LambdaMART-style ranking objective and binary relevance. Use C for early stopping and a small, recorded search over tree complexity and learning rate.

9. Compare three orderings on exactly the same validation candidates: retrieval score alone, a simple hand-weighted combination and the learned ranker.

10. Report end-to-end NDCG\@10 and Recall\@10. Optionally show conditional ranking quality where at least one positive was retrieved, clearly labeled as a diagnostic.

11. Examine feature importance and perform removal experiments. Importance alone is not proof that a feature caused an improvement.

**Deliverables:** point-in-time ranking tables, trained ranker, identical-candidate comparison and a record of excluded training groups.

**Pass the gate when:** group sizes sum to the number of rows, no features use future events, and you can explain why a strong ranker cannot recover a relevant item missing from retrieval.

The [official LGBMRanker API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html) documents its group interface. A ranking score from this setup is not a calibrated probability of purchase or click.

## Week 9 Prove or reject the value of visual features

**Purpose:** produce an experiment that distinguishes this project from a tutorial reproduction.

Use this central question: do image features improve recommendation for users or products with limited interaction evidence?

1. Compare four variants: interaction features only; plus text; plus images; plus both. Keep the evaluation requests, temporal splits and candidate budget fixed.

2. Run two separate experiments. First, freeze candidates to isolate ranking-feature changes. Second, allow retrieval to change and measure the full pipeline. State which experiment each table represents.

3. Define activity and popularity slices using history available before each request: short versus long user history, frequently versus rarely purchased products, missing images and product categories.

4. For the leading variants, run several training seeds if runtime permits. Bootstrap evaluation users to estimate uncertainty in the difference. Resample all requests belonging to a user together.

5. Create a separate simulated item cold-start experiment. Select held-out products deterministically from the historical catalog, remove all their interactions from model fitting, popularity statistics, co-occurrence tables and training user histories, and retain their allowed content metadata. Suppress those products' earlier events in every behavioral feature store in this separate benchmark; include their withheld future events only as evaluation targets.

6. Map held-out item IDs to an unknown ID or use a content-only item path. Use ID dropout during training if your hybrid architecture otherwise depends entirely on known item embeddings.

7. Keep these held-out items eligible at evaluation, evaluate their relevant future interactions separately, and call the result simulated cold start. It is not evidence about actual product launch timing or inventory.

8. Inspect 20 wins and 20 losses. Explain whether images captured useful visual preference, merely repeated category information, or confused similar-looking products with different uses.

9. Write the decision: retain, simplify or remove the image component. A negative result with a sound explanation is useful evidence.

**Deliverables:** ablation table, segment analysis, uncertainty estimates and a short case study.

**Pass the gate when:** you can attribute a measured effect to a particular change and explain where it fails. Do not promise a fixed percentage uplift before running the experiment.

## Week 10 Add history sensitivity and list quality

**Purpose:** understand how a list differs from a collection of independently scored items.

1. Compare user representations built from the latest purchase day, all history equally weighted, and recency-weighted history. Purchases within a day remain an unordered set.

2. Inspect whether a long-term average hides a recent category change. Record examples rather than assuming that recency always helps.

3. Add a simple reranker that penalizes visual near-duplicates or excessive repetition of a product family. Preserve important query constraints in the search surface.

4. A starting rule is maximum marginal relevance: repeatedly choose an item with high relevance and low similarity to items already selected. Tune the relevance-versus-diversity weight on C.

5. Measure NDCG\@10 alongside intra-list diversity. Define diversity as the mean of `1 - cosine_similarity` across item pairs using fixed, normalized content vectors.

6. Measure catalog coverage as the number of unique recommended items divided by the eligible catalog size, and report how often recommendations concentrate in the most popular products.

7. Plot a small trade-off curve. More visual diversity can lower purchase relevance; select a point with an explicit rationale.

8. Check fallbacks: no history, missing images, unknown IDs and too few eligible candidates. Fill remaining slots deterministically with an appropriate popularity baseline.

**Deliverables:** a reranking policy, quality/diversity comparisons and a history-sensitivity experiment.

**Pass the gate when:** you can explain a concrete trade-off and show that the final list contains unique, eligible products. Leave GRU and transformer models until the core works.

## Week 11 Add a properly evaluated search surface

**Purpose:** learn the difference between a user–item recommender and a query–item ranker.

1. Create `search(query, user_context=None, k=10)`. A query is now part of the request. Keep this route separate from the purchase-trained scoring objective.

2. Build a lexical baseline over product name, category and description. TF-IDF is a simple first version; BM25 is a useful next comparison.

3. Encode the text query with the text side of your frozen image/text model. Compare query-to-item text retrieval and query-to-image retrieval.

4. Combine lexical and semantic result lists using reciprocal rank fusion. A starting score is the sum of `1 / (constant + rank)` across sources. Tune the constant and source mix only on development queries.

5. Write 40–60 queries covering exact attributes, vague intent, synonyms and combinations, such as “black sleeveless top” and “light summer trousers.” Avoid constraints the catalog cannot support.

6. Split query intent families into development and held-out test groups before tuning. Keep near-duplicate phrasings in the same group.

7. For each query, pool top results from all compared systems and a few random products. Hide the system identity and grade relevance: 0 irrelevant, 1 partially relevant, 2 strongly relevant. Judge every item needed for each system's reported top-ten metric.

8. Save the judgment rubric and labels. If possible, have a second person independently judge a small subset and report disagreement. This remains a small manually assessed benchmark.

9. Compare lexical, semantic and hybrid search using NDCG\@10 and clear examples. Treat unjudged products as unknown; do not claim complete-catalog relevance recall from this small pool.

10. Keep purchase-based personalization secondary to query relevance. Test whether personalization helps only after the nonpersonalized search route behaves sensibly.

**Optional later extension:** [Microsoft's H&M search dataset](https://huggingface.co/datasets/microsoft/hnm-search-data) provides generated queries linked to products and transactions. Use it as synthetic training supervision for a query-aware ranker if you want more search depth. Split duplicate query/product families carefully and retain independent human judgments for evaluation. Generated queries are not observed customer search logs.

**Deliverables:** a search endpoint, judged query set and a comparison of retrieval methods.

**Pass the gate when:** you can demonstrate how a query changes the result list and support a search-quality claim with query relevance labels. Purchase-ranking improvements alone do not establish search improvements.

## Week 12 Serve the complete system and ship the first portfolio version

**Purpose:** turn experiments into a reliable, inspectable application.

1. Load the selected model, item index, ID mappings and feature configuration as one versioned bundle. Reject a mismatched index/model version.

2. Implement `/recommend`, `/similar` and `/search`. Use the same feature transformations as offline training.

3. Make the recommendation response include item IDs, scores, candidate-source information and model version. Keep internal debugging fields out of the ordinary user view.

4. Build a simple UI with a historical user selector, a personalized grid, a similar-items action and a search box. Show the history cutoff clearly in the research view.

5. Demonstrate an unknown user through a popularity fallback, then optionally allow the demo user to select a few example interests. Call those preferences demo input.

6. Compare offline and API rankings on identical fixture requests. Differences should be explained by explicitly configured policies.

7. Log request ID, anonymous user identifier, timestamp, model version, candidate counts and latency by stage. If recording interactions, distinguish items actually displayed from merely retrieved items.

8. Run a local load test with a stated catalog size, hardware, batch size and concurrency. Report warm p50 and p95 latency and startup/index-loading time separately.

9. Use a starting engineering target such as warm p95 under 200 ms at concurrency 1 on your chosen machine. Measure first, then optimize the largest component. This is a project target, not a production service-level claim.

10. Add simple monitoring for missing features, empty recommendations, error rate and stale model/index versions.

11. Verify a fresh checkout can run the small synthetic fixture and start the demo from documented commands.

12. Record a two-minute video showing a recommendation, an explanation of the candidate pipeline, the image-feature experiment and one known failure.

**Deliverables:** runnable API, small UI, reproducibility instructions, benchmark and first portfolio release.

**Pass the gate when:** another person can follow your README, run the small example and see the intended behavior. Do not delay this release to build an elaborate front end.

## Weeks 13 and 14 Transfer the pipeline to actual video feedback

**Purpose:** prepare directly for feed ranking with outcomes that purchase data cannot provide.

Use [KuaiRand](https://github.com/chongminggao/KuaiRand), an academic dataset of short-video recommendation logs. Start with KuaiRand-Pure for manageable exposed-event modeling. Its restricted video pool means that user sequences are incomplete. If full-history sequence modeling becomes your next focus, the 1K version is the more suitable extension.

1. Read the data dictionary and load the standard-policy and randomly exposed interaction files. Keep their policy indicators. Start with a deterministic training-period user subset if needed, then record the actual cohort and event counts.

2. Build a second adapter into the shared event schema. Preserve user ID, video ID, event time, context and outcomes. Keep the H&M and video ID spaces separate.

3. Define two supervised tasks: predict `long_view` and predict `is_like` before exposure. Read the original definition of each field. In particular, `is_click` has different meanings across interfaces, so do not label every such event a click.

4. Use chronological cutoffs. A concrete initial design is to fit using April 8–21 standard logs, select using April 22–28, and hold out April 29–May 8, after verifying the loaded dates. Analyze standard and random-policy test events separately.

5. Construct features from history strictly before the event. Use content/basic metadata only when it would have been available then. Audit snapshot-based user features before using them. Route unseen user/item IDs to unknown embeddings or content-based fallbacks and report their performance separately.

6. Exclude the current event's watch time, like, long-view and subsequent actions from inputs. They are outcomes.

7. Exclude the supplied month-aggregated video statistics from a temporal experiment. Recompute historical counts from earlier events instead. This is a useful leakage case to explain in interviews.

8. Build a baseline using historical user and item engagement rates with smoothing. Then train a small pointwise classifier using embeddings and historical features.

9. Train a small shared neural representation with two output heads, one for each task. Compare it with the two separate models. Loss weights are tunable design choices, not proof of a business optimum.

10. Evaluate log loss, ROC-AUC and PR-AUC for each label; show base rates and calibration plots. PR-AUC is useful when likes are rare. Report both aggregate and per-user diagnostics, stating how users without both classes are handled.

11. Slice results by history length, video duration, context and exposure policy. Differences between standard and random exposures help you reason about selection bias.

12. If demonstrating a ranked list, combine the two predicted scores with an explicitly illustrative weight. Logged event prediction is your measured task: do not invent a request-level slate or imply you observed outcomes for every alternative video.

13. Keep pre-processing and evaluation interfaces reusable, but retrain models for video outcomes. Purchase-trained model weights are not expected to transfer unchanged.

**Deliverables:** a second data adapter, two outcome models, a multi-task comparison and an exposure-bias analysis.

**Pass the gate when:** you can explain why an exposed-but-unliked video differs from an unobserved product, why watch duration needs context, and why better offline prediction does not prove a better live feed.

Random exposure can support more careful studies, but these experiments alone do not establish online policy uplift. Do not apply inverse propensity weighting unless the relevant logging probabilities, support assumptions and estimator are actually specified. The extension requires no raw-video computer vision claim; the H&M module already demonstrates your image work.

## Week 15 Diagnose one bottleneck and design an online experiment

**Purpose:** practise the kind of judgment that creates value during an internship.

Choose one observed problem rather than adding a model because it is fashionable.

| Observed problem | First investigation | Example improvement to test |
| - | - | - |
| Relevant items rarely enter the candidate set | Candidate recall by source and segment | Change source mix, retriever features or candidate budget |
| Candidates are good but final order is poor | Fixed-candidate ranker comparison | Add a justified feature or correct training labels |
| Sparse users receive poor recommendations | History-length slices | Stronger content path and better fallback |
| Recommendations look repetitive | Product-family repetition and coverage | A controlled reranking penalty |
| API is slow | Latency by stage | Cache item features, batch scoring or adjust the index |
| Metrics suddenly look implausibly high | Feature timestamps and target contamination | Remove leakage and rerun the protocol |

1. Write your diagnosis and one testable hypothesis.

2. Implement one change. Keep the evaluation protocol fixed and compare against the strongest relevant baseline.

3. Report the absolute difference, relative difference, uncertainty, segment effects and computational cost.

4. Make a recommendation to keep, reject or investigate the change. A result that improves one segment and harms another needs an explicit product decision.

5. Write an A/B experiment proposal: randomized unit, primary outcome, guardrails, assignment persistence, expected exposure logging and rollback conditions.

6. For a feed, possible guardrails include negative feedback, content repetition, errors and latency. Do not assume that maximizing watch time alone represents user welfare or long-term satisfaction.

7. Explain that sample size and duration depend on the baseline variance, effect size of interest and traffic. A tiny demo with friends establishes functionality, not a reliable business uplift.

8. Add a draft monitoring checklist covering data freshness, feature distributions, candidate counts, outcomes, latency and model/index version compatibility.

**Deliverables:** a concise experiment memo and an online-test design.

**Pass the gate when:** you can explain why your next experiment is more useful than adding another architecture, and what would make you reject your proposed change.

## Week 16 Lock results and prepare the portfolio

**Purpose:** turn the work into verifiable evidence for internship applications.

1. Freeze all model and policy choices. Run the H&M test block D and the held-out search queries. The video extension has its own locked test dates.

2. If a test reveals a software bug, fix it, document the rerun and distinguish it from a new model choice. Do not repeatedly tune against the final test.

3. Report actual users, requests, catalog size, time ranges and target definitions beside the results.

4. Reproduce the main experiment from a clean environment. Record the package lockfile, source commit, sampling seed, training seed, model checkpoint and index version.

5. Publish a README that explains the problem, architecture, data, split, baselines, main result, demo instructions and limitations. Keep the installation path short.

6. Write a two-to-four-page technical report focused on one question, such as the contribution of visual features for sparse interaction histories.

7. Include the negative findings. A carefully analyzed failed hypothesis demonstrates judgment.

8. Record a three-to-five-minute final walkthrough. Spend more time on evaluation and the experiment than on the UI.

9. Prepare CV bullets using only measured numbers. Do not fill placeholders until the work is complete.

10. Practise explaining the project aloud without looking at the code.

**Example CV templates:**

“Built a two-stage multimodal recommender using PyTorch retrieval, FAISS and LightGBM ranking; evaluated on [N] temporally held-out requests over [M] eligible products.”

“Improved offline NDCG\@10 from [A] to [B] through [specific change], with [L] ms warm p95 serving latency on [hardware and concurrency].”

“Evaluated visual features through controlled ablations and simulated item cold start, identifying [specific benefit or failure] for [defined segment].”

If no meaningful quality gain occurs, use: “Benchmarked collaborative and multimodal recommendation methods and quantified [specific] quality–latency or cold-start trade-offs.” This is stronger than an unsupported improvement claim.

**Pass the gate when:** a reviewer can reproduce the small example, understand the main comparison and trace every CV number to a saved experiment.

## Optional sequence modeling after the core works

Because your target is feed recommendation, sequential modeling is a useful next specialization. Schedule another two to four weeks, or replace part of the search extension if feed modeling is your highest priority.

1. Use a dataset with suitable chronological histories. KuaiRand-1K preserves fuller sampled-user histories than Pure, but requires more data processing. Inspect file sizes before downloading.

2. Begin with three baselines: latest interaction, mean historical embedding and recency-weighted pooling.

3. Train a small GRU over the historical prefix. Handle padding and sequence lengths correctly; start with a capped history such as the last 20–50 events.

4. Make a deliberate choice about the task: predicting a later positive item or predicting engagement for the next exposed item. Construct negatives, evaluation and candidate availability consistently with that choice.

5. Compare against the pooling baseline on identical requests. Watch for gains that come only from using a different history window.

6. If the GRU and evaluation are understood, study a small causal self-attention model such as [SASRec](https://github.com/kang205/SASRec). Mask future positions and verify that padding cannot affect predictions.

7. Inspect users whose recent interests changed and those with short histories. Record cases where order helps and cases where the simpler model wins.

The historical paper and reference code teach the modeling idea; an old dependency stack need not become your project's environment. A new architecture is worthwhile only if its behavior and cost are understood.

## What transfers to each internship

| Internship type | Your strongest evidence | Additional company knowledge you will need |
| - | - | - |
| Shopping recommendation | Purchase-based retrieval, ranker, visual features and item cold start | Stock, price, returns, commercial constraints and actual impression logging |
| Feed recommendation | Retrieval/ranking architecture, history experiments and video outcome modeling | Feed objectives, complete sequences, content policy, creator ecosystem and live experimentation |
| Search ranking | Query-aware retrieval, hybrid search and judged relevance | Real query logs, query intent, position bias and online search satisfaction |
| General ML engineering | Data contracts, reproducibility, serving and monitoring | The company's infrastructure, ownership and deployment process |

When joining a team, your first contribution should follow a repeatable process:

1. Identify the user decision and the product's primary outcome.

2. Map the pipeline: candidate generation, ranking, final rules, serving and logging.

3. Verify what each event means and which features are available at prediction time.

4. Reproduce one existing baseline and its evaluation before changing it.

5. Measure where failures occur and which users/items are affected.

6. Propose one feasible experiment with a clear expected mechanism.

7. Evaluate quality, latency and relevant guardrails.

8. Discuss online testing and rollout with the team, then monitor and retain a rollback path.

This is the transferable skill: diagnosing the system and producing measured improvements. Individual model architectures and business targets will change.

## Interview readiness checks

You should be able to answer these using your own implementation and results:

- Why use retrieval before ranking?

- How would you detect that candidate generation is the bottleneck?

- What did popularity outperform, and why?

- Why does an unobserved interaction differ from a negative impression?

- How did you prevent time leakage?

- Why is your ranker's group a request rather than a lifetime user?

- What did image features add beyond category and text?

- How did you handle a new user and a new item?

- How do target recall and approximate-neighbor recall differ?

- Why might NDCG improve while real users become less satisfied?

- Which part of your system dominates latency?

- What would you measure before an online rollout?

- Which result surprised you, and what did you change because of it?

- What would you simplify if the system had half the compute budget?

Readiness is demonstrated by explanations plus evidence. If you can only define a term but cannot point to an experiment or implementation, revisit that phase.

## Completion checklist and evidence

- [ ] A documented prediction task and data card.

- [ ] Reproducible temporal splits with a locked final test.

- [ ] Correctly tested ranking metrics and request-time features.

- [ ] Strong popularity and collaborative baselines.

- [ ] Image/text features that are actually used in recommendation.

- [ ] A trained retriever and measured candidate coverage.

- [ ] A learned ranker compared on identical candidates.

- [ ] A controlled image-feature or cold-start experiment.

- [ ] A measured relevance/diversity or quality/latency trade-off.

- [ ] A search surface with a clearly labeled judged benchmark.

- [ ] An API and demonstration with useful fallbacks.

- [ ] A video-feedback extension for your FYP goal.

- [ ] An honest experiment table with counts, cutoffs and hardware.

- [ ] A clean reproducibility run, short report and demo video.

- [ ] CV bullets backed by completed work.

Use this experiment table template. Blank cells stay blank until measured.

| Model or change | Split and catalog | Target Recall\@100 | End-to-end NDCG\@10 | p95 latency and hardware | Decision |
| - | - | - | - | - | - |
| Recent popularity | To record | To measure | To measure | To measure | Pending |
| Item co-occurrence | Same protocol | To measure | To measure | To measure | Pending |
| Collaborative embedding baseline | Same protocol | To measure | To measure | To measure | Pending |
| Two-tower retrieval | Same protocol | To measure | To measure | To measure | Pending |
| Fixed candidates plus learned ranker | Same protocol | Unchanged candidates | To measure | To measure | Pending |
| Controlled visual-feature change | State whether candidates change | To measure | To measure | To measure | Pending |

A good project can include a model that loses, a feature you remove and an optimization that improves latency while preserving quality. Those outcomes are credible when the comparison is sound.

## Reading order and first three sessions

Read narrowly and apply each idea immediately.

1. During Week 1, use [Google's recommendation-systems introduction](https://developers.google.com/machine-learning/recommendation) for terminology and the distinction between collaborative and content-based approaches.

2. Before the two-tower model, read [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) for candidate generation and ranking. Read for the architecture and design questions; you do not need to reproduce its production scale.

3. Use the [TensorFlow Recommenders retrieval tutorial](https://www.tensorflow.org/recommenders/examples/basic_retrieval) to understand a worked training/evaluation example, then implement the version you understand in PyTorch.

4. Refer to [OpenCLIP](https://github.com/mlfoundations/open_clip), [FAISS](https://github.com/facebookresearch/faiss/wiki/Getting-started) and [LightGBM](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html) when you reach their phases.

5. Before reporting results, revisit [On Sampled Metrics for Item Recommendation](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/).

6. Before the feed extension, read the [KuaiRand data dictionary](https://github.com/chongminggao/KuaiRand) carefully enough to explain every feature and outcome you use.

**Session 1, approximately two hours:** create the repository and environment; write the task definition; check dataset access; inspect the three main tables; verify five item–image matches.

**Session 2, approximately two hours:** load a small transaction sample; perform the item join; calculate user/item activity counts; save a small Parquet file; document missing values and ID handling.

**Session 3, approximately two hours:** create the five-user toy dataset; hand-calculate Recall and NDCG; implement the metric functions; build a popularity recommender.

Continue according to the completion gates. Move forward when the current stage is understood and reproducible, even if its result is modest.
