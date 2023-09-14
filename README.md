# OCPPM
Object-centric predictive process monitoring research project. The purpose of this project is to handle object-centric event data as naturally as possible for machine learning purposes. This means that graph structures are retained and even learned from.

## Updating ocpa
1. `pip uninstall ocpa`
2. `pip install git+https://github.com/TKForgeron/ocpa`


## TODO
- Overall Thesis
    - [01/09] Drop OFG, retaining flow
    - check figure and table placement, as LaTeX often has trouble handling large ones.
- Abstract
    - Write, just write it.
- 1 Introduction
    - [08/08] Add research goal to introduction
- 2 Theoretical Background
    - [17/08] Update Section 2.2.4, explaining how HeteroGNN combines data of varying dimensionality
    - [31/08] Update Section 2.2.3, increasing clarity about what a feature vector/node feature vector/node vector is
    - [11/09] Mathematically formalize the term flattening in the context of transforming an OCEL to a traditional event log.
- 3 Related Literature
    - [28/08] Update explanation of Adams' EFG by moving parts of 5.1 into Section 3.2.2. 
    - [28/08] Update Section 3.2.2 to use more hedging scrutinizing Adams' framework paper
    - [31/08] Update Sections 3.2 and 3.3, for correctness, clarity, and flow
    - [01/09] Recreate Figure 9 (EFG by Adams with features)
- 4 Addressing the Current Limitations
    - [31/08] Update Fig. 10, Tab. 7
    - [31/08] Update overall flow, and usage of 'flattening', 'aggregates', 'native' and usage of 'criteria', 'characteristics' and 'capabilities'.
- 5 Experimental Setup
    - [28/08] Update Section 5.1 (EFG config)
    - [01/09] Update flow (with OFG removed)
    - [13/09] add preprocessing subsection:
        - [12/09] add preprocessing section for BPI17 and OTC
        - [13/09] update OCEL summary for BPI17
        - [12/09] add data description for OTC
        - [13/09] update OCEL summary and preprocessing/extraction for FI
    - add somewhere that we use PyTorch Geometric
    - update Machine Learning Pipeline section
- 6 Results
    - [01/09] write subsections about the 3 experiments
    - [01/09] Recreate, update, and check figures and tables
    - [14/09] Explicitly note that the scales are not aligned, and argue this is okay as we intend to compare hyperparameter settings within each encoding and dataset, we don't compare encodings yet.
    - [14/09] align violin plots with learning curve plots in terms of size
- 7 Discussion
    - [02/09] Go by each dataset to scrutinize and interpret the findings (across the 3 experiments)
    - [02/09] Discuss information leakage issue, and recommendations and implications of HOEG configuration
    - [02/09] Give synthesis of results (across the datasets)
- 8 Conclusion
    - [02/09] Discuss possible directions future work
    - [02/09] Summarize thesis
    - [02/09] Answer RQs
- Appendix
    - [13/09] Learning Curves of EFG-Based Models (my K-GNN, and Adams' GCN) with Subgraph Sampling on BPI17