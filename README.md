# Developing and benchmarking imputation tools for deep mutational scanning (DMS) data

### Abstract

A key aim of large-scale mutagenesis studies is to depict a full functional genomics landscape of variants. Most recent deep mutational scanning (DMS) studies are designed to evaluate the effects of all possible single amino acid substitutions within the protein region of interest. However, due to technical issues, most DMS datasets published so far are incomplete, leading to loss of information.

In this study, I aim to leverage the knowledge of recommender systems to develop novel DMS imputation methods. Here, I first review the scale and pattern of the missing values in DMS data. Then, I develop two DMS imputation methods inspired by the recommender system algorithms. After that, these methods are benchmarked with previously published protein-feature-dependent imputation approaches. Finally, I gather the output from each of these methods and build an ensemble DMS imputation method.

### Setup & usage

1. Create a virtual environment with Python 3.6.13.
2. Install Jupyter Notebook and other required pacakges according to `requirements.txt`.
3. Download [feature data](http://impute.varianteffect.org/downloads/database.tar.gz) for [Wu et al., 2019](http://impute.varianteffect.org/) and save the downloaded files `/humandb/dms/features/*` to `./src/wu2019/database/humandb/dms/features/*`
4. Follow the code and instructions in the notebooks (`./jupyter_code/`).

### Notebook content
* `P1_Build_and_benchmark_imputation_models`: 
	* In the first section, the imputation methods (`AALasso`, `FactoriseDMS`, `Envision`, `Wu et al., 2019`, `Residue-mean` and `Ensemble`) are benchmarked using all DMS datasets I collected.
	* In the second section, the above methods are built with different proportion of training data to evaluate their performance on DMS data with distinct completeness
	* __Notice:__ These sections are computational expensive, so pre-computed results are provided.
* `P2_Analyse_model_performance`: Run all analysis and create all figures mentioned in the text.

### Data content
* `dms_envfeat_data.csv` contains all DMS data and [Envision features](https://envision.gs.washington.edu/shiny/envision_new/) to be used in this study.
* `dms_info.csv` contains information for the DMS data used in this study.
* `predictor_results.csv` contains predicted variant effects from SIFT, PROVEAN and GEMME for proteins also have DMS data available.
* `blosum80.csv` saves BLOSUM80 scores.
* `codon.txt` saves genetic codons.
* Folder `uniprot_fasta` contains protein sequence in FASTA format downloaded from [UniProt](https://www.uniprot.org/).

### Local setup of Wu et al., 2019
Following procedures and changes were made to Wu et al., 2019 method, so that it can be run locally with high efficiency:

1. The code were downloaded from [their GitHub](https://github.com/joewuca/imputation).
2. The method was set up following [their instructions](https://github.com/joewuca/imputation/blob/master/installation_guide.pdf).
3. Some functions were re-written and saved in `./src/wu2019/wu_modify.py`, so the method can be run locally with the pre-processing and refinement processes truncated. Detailed changes were recorded in the documentation of individual function.