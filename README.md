*goglm* supports estimation of generalized linear models in Go.

The most basic usage is as follows:

```
glm := NewGLM(family, data)
result := glm.Fit()
```

where `family` is a GLM family provided by this package,
e.g. `Binomial`, and `data` is a "DataProvider" as defined in
[statmodel](http://github.com/kshedden/statmodel).


Supported features
------------------

* Estimation via IRLS and [gonum](http://github.com/gonum) optimizers

* Most of the more popular families, links and variance functions

* Unit tests covering all families with their default links and
  variance functions, and some of the more common non-canonical links


Missing features
----------------

* Regularized (ridge/LASSO) estimation

* Model diagnostics

* Less-common GLM families (e.g. Tweedie)

* Inference for survey data

* Marginalization

* Missing data handling

* GEE