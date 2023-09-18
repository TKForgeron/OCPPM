# How Object-Centric is Object-Centric Predictive Process Monitoring?
## Introducing Objects into Object-Centric Predictive Process Monitoring
Object-centric predictive process monitoring research project. The purpose of this project is to handle object-centric event data as naturally as possible for machine learning purposes. In our solution we propose to construct heterogeneous object event graphs per trace to learn how to best include object information into predictions for events (using heterogeneous graph neural network architectures).

## Updating ocpa
1. `pip uninstall ocpa`
2. `pip install git+https://github.com/TKForgeron/ocpa`


## TODO
- Overall Thesis
    - [01/09] Drop OFG, retaining flow
    - Check figure and table placement (as LaTeX often has trouble handling large ones) and overall style.
    - Check usage of "flattening", "native", and "aggregation" AND "will"
    - Check spelling/grammar/flow of the whole document
- Acknowledgements
    - Express my gratefulness towards those who helped me.
- Abstract
    - [14/09 ] Write, just write it.
- 1 Introduction
    - [08/08] Add research objective (RO) to introduction
    - [15/09] Process feedback from draft 2
        - Add figure in example 1, to further explicate and clarify the example
        - Swap order of RO and RQ
        - Refer to the definition of "object graph"
        - Make easier and more readable the use of object perspective and event perspective.
    - [15/09] Update RO to be more specific. "Flattening" now always takes the definition given in Def. 2.4, and negative implications due to the many-to-many relationships between EVENT and OBJECT (see Def. 2.1, Figure 2) are reffered to as "aggregations" (on objects). Aggregations performed on objects to gain event features are exemplified by Example 5 (features from "Object-centric process predictive analytics" by Galanti et al. [2023]) and Objects-perspective features from "A Framework for Extracting and Encoding Features from Object-Centric Event Data" by Adams et al. (2022).
    - [18/09] Extend example to explain limitations of aggregating on objects
- 2 Theoretical Background
    - [17/08] Update Section 2.2.4, explaining how HeteroGNN combines data of varying dimensionality
    - [31/08] Update Section 2.2.3, increasing clarity about what a feature vector/node feature vector/node vector is
    - [11/09] Mathematically formalize the term flattening in the context of transforming an OCEL to a traditional event log (Def. 2.4).
- 3 Related Literature
    - [28/08] Update explanation of Adams' EFG by moving parts of 5.1 into Section 3.2.2. 
    - [28/08] Update Section 3.2.2 to use more hedging scrutinizing Adams' framework paper
    - [31/08] Update Sections 3.2 and 3.3, for correctness, clarity, and flow
    - [01/09] Recreate Figure 9 (EFG by Adams with features)
    - [15/09] Update last paragraph to be more explicit in what we aim to solve.
- 4 Addressing the Current Limitations
    - [31/08] Update Fig. 10, Tab. 7
    - [31/08] Update overall flow, and usage of 'flattening', 'aggregates', 'native' and usage of 'criteria', 'characteristics' and 'capabilities'.
    - Change the chapter title
- 5 Experimental Setup
    - [28/08] Update Section 5.1 (EFG config)
    - [01/09] Update flow (with OFG removed)
    - [13/09] Add preprocessing subsection:
        - [12/09] add preprocessing section for BPI17 and OTC
        - [13/09] update OCEL summary for BPI17
        - [12/09] add data description for OTC
        - [13/09] update OCEL summary and preprocessing/extraction for FI
    - [15/09] Update Machine Learning Pipeline section
        - add somewhere that we use PyTorch Geometric
    - [15/09] Update chapter introduction
    - [18/09] Restructure and update Section 5.3.1, to more clearly list the 3 baselines and discuss the configurations of them (mainly added HP config of GCN)
- 6 Results
    - [01/09] write subsections about the 3 experiments
    - [01/09] Recreate, update, and check figures and tables
    - [14/09] Explicitly note that the scales are not aligned, and argue this is okay as we intend to compare hyperparameter settings within each encoding and dataset, we don't compare encodings yet.
    - [14/09] Align violin plots with learning curve plots in terms of size
    - [18/09] Update baseline tables to exclude Adams reference and include GCN HP config and subgraph sampling information
- 7 Discussion
    - [02/09] Go by each dataset to scrutinize and interpret the findings (across the 3 experiments)
    - [02/09] Discuss information leakage issue, and recommendations and implications of HOEG configuration
    - [02/09] Give synthesis of results (across the datasets)
    - Revise language to be more formal (as it was written very quickly)
- 8 Conclusion
    - [02/09] Discuss possible directions future work
    - [02/09] Summarize thesis
    - [02/09] Answer RQs
- Appendix
    - [13/09] Learning Curves of EFG-Based Models (my K-GNN, and Adams' GCN) with Subgraph Sampling on BPI17

- Presentation
    - Make figure that shows: systems are designed with multiple entities which have complex relations (m-m, 1-m, 1-1). When preparing system data for process mining, we have to join the different tables (=entities) into one table which represents the event log. Since the original systems have complex relationships, we experience divergence, convergence, and decifiency. The object-centric paradigm suggests to not make a join, but to leave the complex relationships in the data. For predictive process monitoring then, the SOTA still makes a (kind of) join which again leads to data issues (data is bent/flattened to fit ML models)! We propose, again, not to make the join, but to leave the data intact by encoding it in a heterogeneous graph structure.