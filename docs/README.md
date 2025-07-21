# Causal Methods Documentation

Comprehensive guides for understanding and applying causal inference methods in business contexts.

## 📚 **Method Guides**

### 🎯 **[Propensity Score Matching (PSM)](psm_guide.md)**
- **Use Case**: Observational studies with rich covariates
- **Strengths**: Balances observed confounders, intuitive matching process
- **Business Application**: User segmentation, feature impact analysis
- **Key Features**: Multiple matching algorithms, automated balance assessment

### 🤖 **[Double Machine Learning (DML)](dml_guide.md)**
- **Use Case**: High-dimensional data with complex relationships
- **Strengths**: Robust to model misspecification, handles many variables
- **Business Application**: A/B test enhancement, customer lifetime value analysis
- **Key Features**: Cross-fitting, multiple ML algorithms, information criteria

### 📈 **[CUPED (Controlled-experiment Using Pre-Experiment Data)](cuped_guide.md)**
- **Use Case**: Randomized experiments with pre-treatment data
- **Strengths**: Variance reduction while preserving unbiasedness
- **Business Application**: A/B test enhancement, experimental power improvement
- **Key Features**: Optimal adjustment coefficients, substantial precision gains

### 📊 **[Difference-in-Differences (DiD)](did_guide.md)**
- **Use Case**: Policy evaluation with before/after structure
- **Strengths**: Controls for time-invariant confounders
- **Business Application**: Feature rollouts, policy changes, natural experiments
- **Key Features**: Parallel trends testing, panel data handling

### ⚖️ **[Synthetic Control](synthetic_control_guide.md)**
- **Use Case**: Individual-level effects with rich pre-treatment data
- **Strengths**: Transparent matching, no parametric assumptions
- **Business Application**: Targeted interventions, segment-specific analysis
- **Key Features**: Individual treatment effects, quality diagnostics, placebo testing

### 🌲 **[Causal Forest](causal_forest_guide.md)**
- **Use Case**: Heterogeneous treatment effects with rich covariates
- **Strengths**: Personalized effects, feature importance, segment discovery
- **Business Application**: Personalized targeting, customer segmentation, ROI optimization
- **Key Features**: Individual treatment effects, automatic covariate selection, business segment analysis

## 🔍 **Method Selection Framework**

### **Data-Driven Method Selection**

| Data Characteristics | Recommended Method | Alternative Options |
|---------------------|-------------------|-------------------|
| **Randomized Experiment** | CUPED | DML (if high-dimensional) |
| **Observational + Rich Covariates** | PSM → DML | Synthetic Control |
| **Before/After + Panel Data** | DiD | Synthetic Control |
| **High-Dimensional Covariates** | DML | PSM with ML propensity scores |
| **Individual-Level Effects Needed** | Synthetic Control | Causal Forest |
| **Heterogeneous Effects Needed** | Causal Forest | Synthetic Control |
| **Limited Pre-treatment Data** | PSM | DiD (if temporal structure) |

### **Business Context Method Selection**

| Business Question | Primary Method | Secondary Method | Rationale |
|------------------|----------------|------------------|-----------|
| **"Does our A/B test show real impact?"** | CUPED | DML | Reduce variance in experiments |
| **"What's the effect on different user segments?"** | Causal Forest | Synthetic Control | Personalized treatment effects |
| **"Did our feature rollout work?"** | DiD | Synthetic Control | Natural experiment design |
| **"How do we control for many confounders?"** | DML | PSM | Handle high-dimensional data |
| **"Which users benefit most from treatment?"** | Causal Forest | Synthetic Control | Heterogeneity and feature importance |
| **"Should we personalize our intervention?"** | Causal Forest | DML | Segment-specific targeting |

### **Sample Size Considerations**

| Sample Size | Recommended Methods | Avoid |
|-------------|-------------------|-------|
| **Small (< 500)** | PSM, DiD | DML (needs cross-fitting) |
| **Medium (500-5000)** | All methods suitable | None |
| **Large (> 5000)** | All methods excel | None (all benefit from larger samples) |

## 📖 **Learning Path Recommendations**

### **🔰 Beginner Path: Foundations First**
1. **Start**: [DiD Guide](did_guide.md) - Learn causal thinking fundamentals
2. **Next**: [PSM Guide](psm_guide.md) - Understand confounders and matching
3. **Then**: [CUPED Guide](cuped_guide.md) - See how to improve experiments

### **📊 Intermediate Path: Method Mastery**
4. **Advanced**: [DML Guide](dml_guide.md) - Master ML for causal inference
5. **Specialized**: [Synthetic Control Guide](synthetic_control_guide.md) - Individual-level effects

### **🎯 Advanced Path: Heterogeneous Effects**
6. **Personalization**: [Causal Forest Guide](causal_forest_guide.md) - Heterogeneous treatment effects

### **🏢 Business Analyst Path**
1. **Quick Start**: [CUPED Guide](cuped_guide.md) - Immediate A/B test improvements
2. **Core Method**: [PSM Guide](psm_guide.md) - Most commonly applicable
3. **Advanced**: [DML Guide](dml_guide.md) - Handle complex business data
4. **Personalization**: [Causal Forest Guide](causal_forest_guide.md) - Segment-specific insights

### **🎓 Data Scientist Path**
1. **Foundation**: [DiD Guide](did_guide.md) - Causal inference principles
2. **ML Integration**: [DML Guide](dml_guide.md) - Combine ML with causal inference
3. **Individual Effects**: [Synthetic Control Guide](synthetic_control_guide.md) - Advanced heterogeneity analysis
4. **Heterogeneous Effects**: [Causal Forest Guide](causal_forest_guide.md) - Personalized treatment effects

## 🔬 **Method Comparison Matrix**

| Aspect | PSM | DML | CUPED | DiD | Synthetic Control | Causal Forest |
|--------|-----|-----|-------|-----|------------------|---------------|
| **Complexity** | Low | High | Medium | Medium | Medium | High |
| **Assumptions** | Strong | Medium | Minimal | Strong | Medium | Medium |
| **Data Requirements** | Moderate | High | Minimal | Panel | Rich Pre-data | High |
| **Individual Effects** | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Heterogeneous Effects** | ❌ | ⚠️ | ❌ | ❌ | ✅ | ✅ |
| **Transparency** | High | Low | High | High | High | Medium |
| **Business Interpretability** | High | Medium | High | High | High | High |

### **Assumption Strength Guide**

- **Minimal**: Few strong assumptions (CUPED)
- **Medium**: Moderate assumptions, testable (DML, Causal Forest, Synthetic Control)
- **Strong**: Several critical assumptions (PSM, DiD)

### **Data Requirement Details**

- **Minimal**: Can work with basic experimental data (CUPED)
- **Moderate**: Needs good set of observed confounders (PSM)
- **High**: Requires many variables or specific structure (DML, Causal Forest)
- **Panel**: Needs multiple time periods (DiD)
- **Rich Pre-data**: Needs multiple relevant pre-treatment predictors (Synthetic Control)

## 🎯 **Quick Reference Cards**

### **Method Selection Decision Tree**

```
Is this a randomized experiment?
├─ YES → CUPED (enhance precision)
└─ NO → Do you need heterogeneous effects?
    ├─ YES → Do you have rich covariates?
    │   ├─ YES → Causal Forest
    │   └─ NO → Synthetic Control (if rich pre-treatment data)
    └─ NO → Do you have rich pre-treatment data?
        ├─ YES → Do you need individual effects?
        │   ├─ YES → Synthetic Control
        │   └─ NO → Do you have high-dimensional data?
        │       ├─ YES → DML
        │       └─ NO → PSM
        └─ NO → Do you have before/after data?
            ├─ YES → DiD
            └─ NO → PSM (with available confounders)
```

### **Quality Assessment Checklist**

#### **For All Methods:**
- [ ] **Assumption plausibility**: Are key assumptions reasonable?
- [ ] **Balance/fit quality**: Do treated and control groups match well?
- [ ] **Statistical significance**: Is the effect statistically reliable?
- [ ] **Business significance**: Is the effect size meaningful?
- [ ] **Robustness**: Consistent across different specifications?

#### **Method-Specific Checks:**
- **PSM**: [ ] Propensity score overlap, [ ] Covariate balance
- **DML**: [ ] Model performance, [ ] Cross-fitting convergence
- **CUPED**: [ ] Covariate correlation, [ ] Variance reduction achieved
- **DiD**: [ ] Parallel trends, [ ] No anticipation effects
- **Synthetic Control**: [ ] Pre-treatment fit, [ ] Weight concentration
- **Causal Forest**: [ ] Feature importance validity, [ ] Effect heterogeneity, [ ] Model stability

## 💼 **Business Implementation Guidelines**

### **Stakeholder Communication**

| Method | Executive Summary | Technical Detail Level |
|--------|------------------|----------------------|
| **PSM** | "Matched similar users for fair comparison" | Medium - intuitive matching |
| **DML** | "Used AI to control for complex factors" | High - requires ML explanation |
| **CUPED** | "Improved A/B test precision using history" | Low - straightforward concept |
| **DiD** | "Compared before/after trends" | Medium - requires trend explanation |
| **Synthetic Control** | "Built artificial control groups" | Medium - intuitive but detailed |
| **Causal Forest** | "Personalized effects for different user types" | Medium - requires heterogeneity explanation |

### **Resource Requirements**

| Method | Data Collection | Analysis Time | Expertise Level |
|--------|----------------|--------------|----------------|
| **PSM** | Moderate | Fast | Intermediate |
| **DML** | High | Medium | Advanced |
| **CUPED** | Low | Fast | Beginner |
| **DiD** | Medium | Medium | Intermediate |
| **Synthetic Control** | High | Medium | Intermediate |
| **Causal Forest** | High | Medium | Advanced |

### **Implementation Timeline**

- **Immediate (1-2 weeks)**: CUPED for existing A/B tests
- **Short-term (1 month)**: PSM for observational analysis
- **Medium-term (2-3 months)**: DiD for policy evaluations
- **Long-term (3-6 months)**: DML, Synthetic Control, and Causal Forest for advanced analysis

## 🎓 **Additional Resources**

### **Academic References**
- **PSM**: Rosenbaum & Rubin (1983), Austin (2011)
- **DML**: Chernozhukov et al. (2018), Athey & Imbens (2019)
- **CUPED**: Deng et al. (2013), Zhao et al. (2019)
- **DiD**: Angrist & Pischke (2009), Cunningham (2021)
- **Synthetic Control**: Abadie et al. (2010), Arkhangelsky et al. (2021)
- **Causal Forest**: Wager & Athey (2018), Athey & Wager (2019)

### **Software Ecosystems**
- **Python**: This repository, EconML (Causal Forest), DoWhy
- **R**: MatchIt, grf (Causal Forest), did, Synth
- **Commercial**: Stata, SAS

### **Professional Development**
- **Courses**: MIT 14.387, Stanford CS229T, Coursera Causal Inference
- **Books**: "Causal Inference: The Mixtape", "Mostly Harmless Econometrics"
- **Conferences**: NBER Methods, ICML Causal Inference Workshop

---

## 🚀 **Getting Started**

1. **Choose your learning path** based on your background and needs
2. **Read the relevant method guide** for detailed implementation
3. **Try the Jupyter notebooks** for hands-on experience
4. **Apply to your business problem** with appropriate method selection
5. **Validate results** using quality assessment checklists

**📧 Questions?** Check the individual method guides for detailed implementation examples and troubleshooting tips.

---

**🎯 This documentation provides everything you need to implement rigorous causal inference methods for business decision-making.** 