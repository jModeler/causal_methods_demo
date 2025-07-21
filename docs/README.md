# Documentation Index

Welcome to the Causal Methods Demo documentation! This directory contains comprehensive guides and references for all causal inference methods implemented in this project.

## 📚 Available Documentation

### 🎯 **Core Guides**

#### **[Configuration Parameters](configuration_parameters.md)**
Comprehensive guide to all simulation parameters used in the synthetic data generation system. Covers demographics, treatment assignment, outcome modeling, and scenario configurations.

**Contents:**
- Parameter types and default values
- Business rationale and usage examples
- Scenario design principles
- YAML configuration structure

### 🔬 **Method Documentation**

#### **Propensity Score Matching (PSM)** 
- **Comprehensive Guide**: **[PSM Method Guide](psm_guide.md)** - Theory, implementation, and best practices
- **Notebook**: [`notebooks/02_psm_tax.ipynb`](../notebooks/02_psm_tax.ipynb)
- **Implementation**: [`src/causal_methods/psm.py`](../src/causal_methods/psm.py)

**What you'll learn:**
- PSM theory and methodology
- Step-by-step implementation guide
- Covariate balance assessment
- Treatment effect estimation
- Statistical significance testing
- Comparison with naive estimates
- Best practices and limitations

**Key Features Covered:**
- Multiple matching algorithms (nearest neighbor, caliper)
- Automated balance diagnostics
- Rich visualizations (propensity distributions, balance plots)
- Proper statistical tests for binary outcomes
- Sensitivity analysis and robustness checks

#### **Double Machine Learning (DML)** ✨ *Enhanced with Information Criteria!*
- **Comprehensive Guide**: **[DML Method Guide](dml_guide.md)** - Advanced causal ML with model selection
- **Notebook**: [`notebooks/03_dml_tax.ipynb`](../notebooks/03_dml_tax.ipynb)
- **Implementation**: [`src/causal_methods/dml.py`](../src/causal_methods/dml.py)

**What you'll learn:**
- DML theory and cross-fitting methodology
- Information criteria for model selection (AIC, BIC)
- Multiple ML algorithms integration
- Robust inference and statistical testing
- Model performance diagnostics
- Business impact assessment

**Key Features Covered:**
- Cross-fitting with K-fold sample splitting
- Multiple ML models (Random Forest, Gradient Boosting, Linear)
- Information criteria-based model selection
- Influence function-based standard errors
- Comprehensive diagnostic visualizations
- Treatment effect heterogeneity analysis

#### **CUPED (Controlled-experiment Using Pre-Experiment Data)** ✨ *Variance Reduction for A/B Tests*
- **Comprehensive Guide**: **[CUPED Method Guide](cuped_guide.md)** - Precision improvement for experiments
- **Notebook**: [`notebooks/04_cuped_tax.ipynb`](../notebooks/04_cuped_tax.ipynb)
- **Implementation**: [`src/causal_methods/cuped.py`](../src/causal_methods/cuped.py)

**What you'll learn:**
- CUPED theory and variance reduction principles
- Optimal adjustment coefficient estimation
- Pre-experiment covariate selection
- Statistical power improvement
- A/B testing enhancement strategies
- Business impact measurement

**Key Features Covered:**
- Multiple regression adjustment methods (OLS, Ridge, Lasso)
- Covariate balance checking for randomized experiments
- Comprehensive comparison visualizations
- Statistical power and precision analysis
- Cost-effective experiment design
- Integration with A/B testing platforms

#### **Difference-in-Differences (DiD)** ✨ *New!*
- **Comprehensive Guide**: **[DiD Method Guide](did_guide.md)** - Panel data causal inference
- **Notebook**: [`notebooks/01_did_tax.ipynb`](../notebooks/01_did_tax.ipynb)
- **Implementation**: [`src/causal_methods/did.py`](../src/causal_methods/did.py)

**What you'll learn:**
- DiD theory and parallel trends assumption
- Panel data preparation and reshaping
- Treatment timing exploitation
- Parallel trends testing and validation
- Heterogeneous treatment effects
- Policy evaluation applications

**Key Features Covered:**
- Automatic panel data preparation
- Robust standard errors with clustering
- Parallel trends visualization and testing
- Subgroup effect analysis
- Missing data handling
- Business impact translation

### 🛠️ **Technical References**

#### **API Documentation**
- **[PSM API](../src/causal_methods/psm.py)**: Complete PropensityScoreMatching class reference
- **[DML API](../src/causal_methods/dml.py)**: DoubleMachineLearning class reference
- **[CUPED API](../src/causal_methods/cuped.py)**: CUPED class reference
- **[DiD API](../src/causal_methods/did.py)**: DifferenceInDifferences class reference  
- **[Data Simulation API](../src/data_simulation.py)**: TaxSoftwareDataSimulator class reference

#### **Testing Documentation**
- **Test Coverage**: 90%+ overall with comprehensive test suite
- **Test Structure**: Unit tests, integration tests, edge case coverage
- **Running Tests**: `uv run pytest tests/ --cov=src --cov-report=term-missing`

### 📊 **Data and Configuration**

#### **Scenario Explanations**
Understanding the different data generation scenarios:

1. **Baseline Scenario** (`config/simulation_config.yaml`)
   - Standard treatment adoption rates (60-70%)
   - Moderate treatment effects
   - Balanced user demographics

2. **High Treatment Scenario** (`config/scenario_high_treatment.yaml`)
   - Enhanced treatment effects
   - Higher smart assistant impact
   - Optimistic business case

3. **Low Adoption Scenario** (`config/scenario_low_adoption.yaml`)
   - Reduced treatment adoption (40-50%)
   - Lower engagement rates
   - Conservative business case

### 🎓 **Learning Path**

#### **For Beginners**
1. Start with **[Configuration Parameters](configuration_parameters.md)** to understand the data
2. Choose your causal inference method:
   - **Randomized experiments?** → Start with **[CUPED Guide](cuped_guide.md)**
   - **Observational data?** → Start with **[PSM Guide](psm_guide.md)**
   - **Panel/time series data?** → Start with **[DiD Guide](did_guide.md)**
   - **Complex/High-dimensional?** → Start with **[DML Guide](dml_guide.md)**
3. Run the corresponding notebook for hands-on practice
4. Experiment with different scenarios in the config files

#### **For Advanced Users**
1. Review all **method guides** for comprehensive understanding
2. Examine the **API documentation** for method customization
3. Study the **test suite** for edge case handling and best practices
4. Integrate multiple methods for robust causal inference
5. Extend the implementations with custom functionality

#### **Method Selection Guide**
Choose the right method for your use case:

| **Data Type** | **Best Method** | **Alternative** | **Use When** |
|---------------|-----------------|-----------------|---------------|
| Randomized Experiment | **CUPED** | DML | Want to reduce variance and increase precision |
| Cross-sectional Observational | **PSM** | DML | Need to match similar units |
| Panel/Longitudinal | **DiD** | DML | Have before/after periods |
| Complex/High-dimensional | **DML** | PSM + DiD | Many covariates or non-linear relationships |

### 🔧 **Development Guide**

#### **Adding New Documentation**
1. Create markdown files in this `docs/` directory
2. Update this index with links and descriptions
3. Include practical examples and code snippets
4. Cross-reference with relevant notebooks and implementations

#### **Documentation Standards**
- Use clear headings and structure
- Include practical examples and business context
- Reference theoretical background
- Provide troubleshooting guidance
- Link to relevant code sections

### 📈 **Recent Updates**

#### **Version 3.0** (Current)
- ✅ **Complete Method Coverage**: PSM, DML, CUPED, and DiD implementations
- ✅ **Enhanced Documentation**: Comprehensive guides for all methods
- ✅ **Information Criteria Integration**: AIC/BIC model selection for DML
- ✅ **Variance Reduction**: CUPED for experiment precision improvement
- ✅ **Panel Data Analysis**: DiD for temporal causal inference
- ✅ **Business Translation**: Practical impact assessment tools
- ✅ **Test Coverage**: Robust testing across all methods

#### **Key Innovations**
- **Information Criteria Model Selection**: Move beyond R² to principled model choice
- **Integrated Workflow**: Seamless combination of multiple causal methods
- **Business Focus**: Translate statistical results to actionable insights
- **Robustness**: Comprehensive error handling and edge case management

### 🤝 **Contributing to Documentation**

We welcome contributions to improve and expand this documentation:

1. **Fix Issues**: Correct errors or clarify confusing sections
2. **Add Examples**: Provide more practical use cases and business applications
3. **Expand Coverage**: Document advanced techniques or new methods
4. **Improve Structure**: Better organization and cross-referencing

### 📞 **Getting Help**

If you need assistance:
1. Check the relevant **method guide** for comprehensive theory and examples
2. Run the corresponding **notebook** for step-by-step demonstrations
3. Review the **API documentation** for method parameters and options
4. Examine the **test files** for usage patterns and edge cases
5. Create an issue for bugs or feature requests

### 🏆 **Best Practices Summary**

#### **For Reliable Causal Inference:**
1. **Understand your data structure** - Use the configuration guide
2. **Choose the right method** - Follow the method selection guide
3. **Validate assumptions** - Use diagnostic tools in each method
4. **Check robustness** - Try multiple methods when possible
5. **Interpret carefully** - Use business translation tools
6. **Document thoroughly** - Record your analysis decisions

#### **For Business Impact:**
1. **Define clear outcomes** - Measure what matters to business
2. **Quantify uncertainty** - Report confidence intervals, not just point estimates
3. **Translate to business metrics** - Connect statistical results to ROI
4. **Consider practical significance** - Statistical ≠ business significance
5. **Plan for action** - Design analysis to inform decisions

---

**Happy Learning!** 🎉

This documentation is designed to help you master causal inference methods and apply them effectively to real-world business problems. Each method has its strengths - choose the right tool for your specific use case and data structure. 