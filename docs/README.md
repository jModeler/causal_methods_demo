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

#### **Propensity Score Matching (PSM)** ✨ *New!*
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

#### **Difference-in-Differences (DiD)**
- **Notebook**: [`notebooks/01_did_tax.ipynb`](../notebooks/01_did_tax.ipynb)
- **Implementation**: [`src/causal_methods/did.py`](../src/causal_methods/did.py)

**What you'll learn:**
- DiD theory and assumptions
- Panel data preparation
- Parallel trends testing
- Treatment effect estimation
- Visualization techniques

### 🛠️ **Technical References**

#### **API Documentation**
- **[PSM API](../src/causal_methods/psm.py)**: Complete PropensityScoreMatching class reference
- **[DiD API](../src/causal_methods/did.py)**: DifferenceInDifferences class reference  
- **[Data Simulation API](../src/data_simulation.py)**: TaxSoftwareDataSimulator class reference

#### **Testing Documentation**
- **Test Coverage**: 84% overall with 97 tests
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
2. Run the **[PSM Notebook](../notebooks/02_psm_tax.ipynb)** for hands-on learning
3. Explore the **[DiD Notebook](../notebooks/01_did_tax.ipynb)** for temporal analysis
4. Experiment with different scenarios in the config files

#### **For Advanced Users**
1. Review the **API documentation** for method customization
2. Examine the **test suite** for edge case handling
3. Extend the methods with custom functionality
4. Contribute new causal inference methods

### 🔧 **Development Guide**

#### **Adding New Documentation**
1. Create markdown files in this `docs/` directory
2. Update this index with links and descriptions
3. Include code examples and practical applications
4. Cross-reference with relevant notebooks and implementations

#### **Documentation Standards**
- Use clear headings and structure
- Include practical examples
- Reference theoretical background
- Provide troubleshooting guidance
- Link to relevant code sections

### 📈 **Recent Updates**

#### **Version 2.0** (Current)
- ✅ **Added PSM Implementation**: Complete propensity score matching with multiple algorithms
- ✅ **Enhanced Statistical Testing**: Proper handling of binary outcomes with proportion tests
- ✅ **Improved Documentation**: Comprehensive guides and API references
- ✅ **Test Coverage**: Expanded to 97 tests with 84% coverage
- ✅ **Code Quality**: Ruff linting, consistent formatting, type hints

#### **Planned Additions**
- [ ] Double Machine Learning (DML) documentation
- [ ] Instrumental Variables (IV) guide
- [ ] CUPED methodology explanation
- [ ] Causal Forests implementation guide
- [ ] Sensitivity analysis documentation

### 🤝 **Contributing to Documentation**

We welcome contributions to improve and expand this documentation:

1. **Fix Issues**: Correct errors or clarify confusing sections
2. **Add Examples**: Provide more practical use cases
3. **Expand Coverage**: Document new methods or advanced techniques
4. **Improve Structure**: Better organization and navigation

### 📞 **Getting Help**

If you need assistance:
1. Check the relevant notebook for step-by-step examples
2. Review the API documentation for method parameters
3. Examine the test files for usage patterns
4. Create an issue for bugs or feature requests

---

**Happy Learning!** 🎉

This documentation is designed to help you master causal inference methods and apply them effectively to real-world business problems. 