# Biological validation gates for MVP v0

Reference:
Scanpy CPU pipeline is the biological reference for PBMC 68k.

Primary clustering gate:
ARI vs Scanpy CPU reference >= 0.90.
NMI is reported as a secondary diagnostic metric.

Marker preservation:
Top 20 marker genes per cluster overlap >= 80%.
Top 50 marker overlap may be reported, but is not blocking in MVP v0.

Cluster count drift:
±1 cluster = pass.
±2 clusters = warning and requires science review.
>2 clusters = fail.

Neighborhood preservation:
Required as diagnostic metric in MVP v0, especially for KNN/backend comparisons.
Not blocking until threshold is confirmed by science lead.

iLISI / batch mixing:
Not required for single-donor PBMC MVP.
Required later for datasets with batch structure or batch integration step.

Final biological validity:
Automated gates produce pass/warning/fail.
Final decision is made by the science lead