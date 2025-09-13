# LLM-FK: Automating Foreign Key Reconstruction

# in Large-Scale Complex Databases

### the method and dataset of LLM-FK，the detailed directory structure as followed

    1. 📊 dataset/                    # Datasets and metadata
       1.1 NorthWind/               
       1.2 WideWorldImporters/       
       1.3 Adventureworks/           
       1.4 Musicbrainz/              
       
    2. 📈 results/                   # Experimental evaluation results
       2.1 Effectiveness Evaluation/    # Detailed FK reconstruction results for RQ1
       2.2 Ablation Analysis/            # Detailed FK reconstruction results for RQ2
       2.3 Pruning Effectiveness Evaluation/              # Prune results for RQ3
       2.4 Robustness Evaluation            # Detailed FK reconstruction results for RQ4
       
    3. 📁 scripts/                   # Core implementation of the LLM-FK method
       3.1 prune/                    # Phase 1: Candidate foreign key pair pruning
          3.1.1 ...                   # All modules and code for this phase
       3.2 local_identify/           # Phase 2: Local FK identification
          3.2.1 ...                   # All modules and code for this phase
       3.3 global_conflict_resolve/  # Phase 3: Global conflict resolution
          3.3.1 ...                   # All modules and code for this phase
       3.4 🚀 LLM_FK.py                 # Main entry point script
       
    4. 🛠️ util/                      # Common utility code
       
    5. ⚙️ config.yaml                # Main project configuration file
       # Contains parameters for:
       #   - LLM (API keys, base URL, model type, etc.)
       #   - Method (strategy settings for each phase)xxxxxxxxxx #   
