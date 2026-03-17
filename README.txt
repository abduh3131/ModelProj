SOFE 4820 - Modeling & Simulation Final Project
Group 2: Abdulkarim Noorzaie (100748590), Abdullah Hanoosh (100749026)

Disease Transmission & Intervention Simulation
Multi-Group SEIR Model with Monte Carlo Analysis

================================================================================
REQUIREMENTS
================================================================================
- Python 3.10 or higher
- Required packages: numpy, matplotlib, scipy, pandas

Install dependencies:
    pip install -r requirements.txt

================================================================================
HOW TO RUN
================================================================================
1. Open a terminal/command prompt
2. Navigate to the src directory:
       cd src
3. Run the main simulation:
       python main.py

The simulation will:
- Run 300 Monte Carlo simulations for each of the 6 intervention scenarios
- Generate all plots in the output/ folder
- Print summary tables to the console
- Run model validation against COVID-19 benchmarks

Expected runtime: approximately 2-3 minutes

================================================================================
PROJECT STRUCTURE
================================================================================
ModelProject/
    src/
        seir_model.py      - Core multi-group SEIR simulation engine
        monte_carlo.py     - Monte Carlo runner and statistical aggregation
        interventions.py   - 6 intervention scenario definitions
        analysis.py        - Severity analysis, R0 sensitivity, delay comparison
        validation.py      - Model validation against real COVID-19 data
        main.py            - Main entry point, runs everything
    output/                - Generated plots (created automatically)
    requirements.txt       - Python dependencies
    README.txt             - This file

================================================================================
INTERVENTION SCENARIOS
================================================================================
1. No Intervention (Baseline) - uncontrolled spread
2. School Closures           - 70% reduction in children contacts
3. Workplace Restrictions    - 50% reduction in adult contacts
4. Elderly Isolation          - 60% reduction in elderly contacts
5. Combined Moderate         - partial reduction across all groups
6. Full Lockdown             - 75% reduction in all contacts

All interventions include a 14-day detection delay before they take effect.

================================================================================
OUTPUT PLOTS
================================================================================
- seir_*.png                      - SEIR curves per group for each scenario
- scenario_comparison.png         - Total infections bar chart across scenarios
- infection_curves_overlay.png    - Overlaid infection curves for all scenarios
- group_breakdown.png             - Per-group infection breakdown
- infection_timeline.png          - Start/peak/end timeline per group
- severity_comparison.png         - Estimated deaths and hospitalizations
- detection_delay_comparison.png  - Impact of detection delay on outcomes
- r0_sensitivity.png              - R0 sensitivity analysis
- mc_distribution_*.png           - Monte Carlo distribution histogram
- validation_curve_comparison.png - Comparison with Ontario COVID-19 data

================================================================================
DATA SOURCES
================================================================================
- COVID-19 epidemiological parameters: Li et al. (2020), NEJM
- Contact matrix: Mossong et al. (2008), POLYMOD study, PLOS Medicine
- Contact matrix projections: Prem et al. (2017), PLOS Computational Biology
- R0 estimates: Alimohamadi et al. (2020), meta-analysis
- COVID-19 case data: Johns Hopkins CSSE archived repository
  https://github.com/CSSEGISandData/COVID-19
- Infection fatality rates: Levin et al. (2020), European Journal of Epidemiology
