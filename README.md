# Semantic-Aware Image Retargeting

**Course:** cv5561-f25 (Fall 2025)
**Project Repository:** `[https://github.com/sri299792458/cv5561-f25-team-asa]`

## Project Description

This project aims to improve upon traditional seam carving for image retargeting by integrating modern, semantic-aware models. The original approach, while clever, relies on simple gradient-based energy, which can distort important objects or create artifacts.

We plan to leverage contemporary vision models, such as Segment Anything 2 (for object-level understanding) and Depth Anything V2 (for spatial/foreground understanding), to create a more robust energy function. This "semantic-aware" approach should better preserve key objects and more intelligently handle repairs, resulting in higher-quality retargeted images.

## Team Members & Roles

| Name | Email | Role |
| :--- | :--- | :--- |
| [Srinivas Kantha Reddy] | [kanth042@umn.edu] | Coordinator |
| [Ayaan Mohammed] | [moha2747@umn.edu] | [Role TBD] |
| [Apurv Kushwaha] | [kushw022@umn.edu] | [Role TBD] |

## Core Papers

* **PruneRepaint (NeurIPS 2024):** [https://arxiv.org/abs/2410.22865](https://arxiv.org/abs/2410.22865)
* **Segment Anything 2:** [https://arxiv.org/abs/2408.00714](https://arxiv.org/abs/2408.00714)
* **Depth Anything V2:** [https://arxiv.org/abs/2406.09414](https://arxiv.org/abs/2406.09414)

## Evaluation

* **Dataset:** RetargetMe benchmark (80 images)
* **Metric:** Better object preservation and fewer artifacts compared to traditional seam carving, evaluated both quantitatively (if possible) and qualitatively.
