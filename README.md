# Clust-McMs
Enhancing Event-centric News Cluster Summarization via Data Sharpening and Localization Insights. ACL2025

Contact [Longyin Zhang](zhangly@i2r.a-star.edu.sg) for more info.


# Evaluation Data
Event-centric news clusters as well as the silver summaries can be found at:
  1. `data/mds_ac_dev_final.json`  # validation set of article clustering
  2. `data/mds_ac_test_final.json`  # test set of article clustering
  3. `data/mds_clusters_gpt_event.json`  # test set of clust-mcms pipeline
  4. `data/mds_clusters_gpt_event_dev.json`  # validation set of clust-mcms pipeline
  5. `data/mmds_test_hit_filtered.json`  # GLOBESUMM test set
  6. `data/mmds_valid_hit_filtered.json`  # GLOBESUMM validation set


# Code
train_sft.sh  # LLM SFT
run_me.py  # main event extraction
run_mmds.py  # main code of clust-mcms
eval_faith.py  # evaluation of the entity-level faithfulness of the generated summaries
eval_coverage.py  # evaluation of the event-level coverage of the generated summaries


# SFT Models
`ml_models/`
