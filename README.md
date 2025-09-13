# LLM-FK: Automating Foreign Key Reconstruction

# in Large-Scale Complex Databases

### the method and dataset of LLM-FK，the detailed directory structure as followed

.
├── 📊 dataset/                    # Datasets and metadata
│   ├── NorthWind/               
│   ├── WideWorldImporters/       
│   ├── Adventureworks/           
│   └── Musicbrainz/              
│
├── 📈 results/                   # Experimental evaluation results
│   ├── Effectiveness Evaluation/    # Detailed FK reconstruction results for RQ1
│   ├── Ablation Analysis/            # Detailed FK reconstruction results for RQ2
│   ├── Pruning Effectiveness Evaluation/              # Prune results for RQ3
│   └── Robustness Evaluation            # Detailed FK reconstruction results for RQ4
│
├── 📁 scripts/                   # Core implementation of the LLM-FK method
│   ├── prune/                    # Phase 1: Candidate foreign key pair pruning
│   │   └── ...                   # All modules and code for this phase
│   ├── local_identify/           # Phase 2: Local FK identification
│   │   └── ...                   # All modules and code for this phase
│   ├── global_conflict_resolve/  # Phase 3: Global conflict resolution
│   │   └── ...                   # All modules and code for this phase
│   └── 🚀 LLM_FK.py                 # Main entry point script
│
├── 🛠️ util/                      # Common utility code
│   
├── ⚙️ config.yaml                # Main project configuration file
│   # Contains parameters for:
│   #   - LLM (API keys, base URL, model type, etc.)
│   #   - Method (strategy settings for each phase)
└── 

    #   
