# This repository is no longer being actively developed.  See [here](https://github.com/kshedden/statmodel/tree/master/glm) for the new version of this package.

__goglm__ supports estimation of generalized linear models in Go.

A basic usage example is as follows:

```
fam := goglm.NewFamily(goglm.BinomialFamily)
// data is a dstream
glm := goglm.NewGLM(data, "Y").Family(fam).Done()
rslt := glm.Fit()
print(rslt.Summary().String())
```

`NewFamily` returns a GLM family (here it is the `Binomial` family),
and `data` is a
"Dstream" as defined in the [dstream](http://github.com/kshedden/dstream)
package.  The Dstream is used to feed data to the GLM in chunks
using a column-oriented storage layout.  A more extensive illustration
can be found in the "examples" directory.


Supported features
------------------

* Estimation via IRLS and [gonum](http://github.com/gonum) optimizers

* Supports many GLM families, links and variance functions

* Supports estimation for case-weighted datasets

* Models can be specified using formulas

* Regularized (ridge/LASSO/elastic net) estimation

* Offsets

* Unit tests covering all families with their default links and
  variance functions, and some of the more common non-canonical links


Missing features
----------------

* Performance assessments

* Model diagnostics

* Less-common GLM families (e.g. Tweedie)

* Marginalization

* Missing data handling

* GEE

* Inference for survey data
