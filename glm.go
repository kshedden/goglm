package goglm

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/gonum/floats"
	"github.com/kshedden/statmodel"
	"github.com/kshedden/statmodel/dataprovider"
)

// GLM describes a generalized linear model.
type GLM struct {
	statmodel.IndRegModel

	// The GLM family
	Fam *Family

	// The GLM link function
	Link *Link

	// The GLM variance function
	Var *Variance

	// Either IRLS (default) or Gradient
	FitMethod string

	// Starting values, optional
	Start []float64

	// L1 (lasso) penalty weight.  FitMethod is ignored if
	// non-zero.
	L1wgt []float64

	// L2 (ridge) penalty weights, optional.  Must fit using
	// Gradient method if present.
	L2wgt []float64

	// Additional information that is model-specific
	Aux interface{}
}

// GLMResults describes the results of a fitted generalized linear model.
type GLMResults struct {
	statmodel.BaseResults

	scale float64
}

// Scale returns the estimated scale parameter.
func (rslt *GLMResults) Scale() float64 {
	return rslt.scale
}

// NewGLM creates a new GLM object for the given family, using its
// default link and variance functions.
func NewGLM(fam *Family, data dataprovider.Reg) *GLM {

	var link *Link
	var vaf *Variance

	fname := strings.ToLower(fam.Name)
	switch fname {
	case "binomial":
		link = NewLink("logit")
		vaf = NewVariance("binomial")
	case "poisson":
		link = NewLink("log")
		vaf = NewVariance("identity")
	case "quasipoisson":
		link = NewLink("log")
		vaf = NewVariance("identity")
	case "gaussian":
		link = NewLink("identity")
		vaf = NewVariance("const")
	case "gamma":
		link = NewLink("recip")
		vaf = NewVariance("squared")
	case "invgaussian":
		link = NewLink("recipsquared")
		vaf = NewVariance("cubed")
	case "negbinom":
		alpha := fam.Aux.(NegBinomAux).Alpha
		return NewNegBinomGLM(alpha, data)
	default:
		msg := fmt.Sprintf("Unknown GLM family: %s\n", fam.Name)
		panic(msg)
	}

	return &GLM{
		IndRegModel: statmodel.IndRegModel{Data: data},
		Fam:         fam,
		Link:        link,
		Var:         vaf,
		FitMethod:   "IRLS",
	}
}

// NegBinomAux contains information specific to a negative binomial GLM.
type NegBinomAux struct {
	Alpha float64
}

// NewNegBinomGLM creates a GLM object with a negative binomial family
// type, using the given parameter alpha to determine the
// mean/variance relationship.  The variance corresponding to mean m
// is m + alpha*m^2.
func NewNegBinomGLM(alpha float64, data dataprovider.Reg) *GLM {

	fam := NewNegBinomFamily(alpha, NewLink("log"))
	vaf := NewNegBinomVariance(alpha)

	return &GLM{
		IndRegModel: statmodel.IndRegModel{Data: data},
		Fam:         fam,
		Link:        NewLink("log"),
		Var:         vaf,
		FitMethod:   "IRLS",
		Aux:         NegBinomAux{Alpha: alpha},
	}
}

// SetLink sets the link function safely (restricting to the valid
// links for the GLM family).  It is also usually possible to set the
// Link field directly, but don't do this with the negative binomial
// family.
func (glm *GLM) SetLink(link *Link) {

	if !glm.Fam.IsValidLink(link) {
		panic("Invalid link")
	}

	if strings.ToLower(glm.Fam.Name) == "negbinom" {
		// Need to reset the family when the link changes
		alpha := glm.Aux.(NegBinomAux).Alpha
		fam := NewNegBinomFamily(alpha, NewLink("log"))
		glm.Fam = fam
	}
	glm.Link = link
}

// LogLike returns the log-likelihood value for the generalized linear
// model at the given parameter values.
func (glm *GLM) LogLike(params []float64, scale float64) float64 {

	var loglike float64
	var linpred []float64
	var mn []float64

	nvar := glm.Data.Nvar()
	glm.Data.Reset()

	for glm.Data.Next() {

		yda := glm.Data.YData()
		wgts := glm.Data.Weights()
		off := glm.Data.Offset()
		n := len(yda)

		// Adjust the allocations
		linpred = resize(linpred, n)
		mn = resize(mn, n)

		// Update the linear predictor
		zero(linpred)
		for j := 0; j < nvar; j++ {
			xda := glm.Data.XData(j)
			floats.AddScaled(linpred, params[j], xda)
		}
		if off != nil {
			floats.Add(linpred, off)
		}

		// Update the log likelihood value
		glm.Link.InvLink(linpred, mn)
		loglike += glm.Fam.LogLike(yda, mn, wgts, scale)
	}

	// Account for the L2 penalty
	if glm.L2wgt != nil {
		nobs := float64(glm.Data.Nobs())
		for j, v := range glm.L2wgt {
			loglike -= nobs * v * params[j] * params[j] / 2
		}
	}

	return loglike
}

func scoreFactor(yda, mn, deriv, va, sfac []float64) {
	for i, y := range yda {
		sfac[i] = (y - mn[i]) / (deriv[i] * va[i])
	}
}

// Score returns the score vector for the generalized linear model at
// the given parameter values.
func (glm *GLM) Score(params []float64, scale float64, score []float64) {

	var linpred []float64
	var mn []float64
	var deriv []float64
	var va []float64
	var fac []float64
	var facw []float64

	nvar := glm.Data.Nvar()
	glm.Data.Reset()
	zero(score)

	for glm.Data.Next() {

		yda := glm.Data.YData()
		wgts := glm.Data.Weights()
		off := glm.Data.Offset()
		n := len(yda)

		// Adjust the allocations
		linpred = resize(linpred, n)
		mn = resize(mn, n)
		deriv = resize(deriv, n)
		va = resize(va, n)
		fac = resize(fac, n)
		facw = resize(facw, n)

		// Update the linear predictor
		zero(linpred)
		for j := 0; j < nvar; j++ {
			xda := glm.Data.XData(j)
			floats.AddScaled(linpred, params[j], xda)
		}
		if off != nil {
			floats.Add(linpred, off)
		}

		glm.Link.InvLink(linpred, mn)
		glm.Link.Deriv(mn, deriv)
		glm.Var.Var(mn, va)

		scoreFactor(yda, mn, deriv, va, fac)

		for j := 0; j < nvar; j++ {
			xda := glm.Data.XData(j)

			if wgts == nil {
				score[j] += floats.Dot(fac, xda)
			} else {
				floats.MulTo(facw, fac, wgts)
				score[j] += floats.Dot(facw, xda)
			}
		}
	}

	// Account for the L2 penalty
	if glm.L2wgt != nil {
		nobs := float64(glm.Data.Nobs())
		for j, v := range glm.L2wgt {
			score[j] -= nobs * v * params[j]
		}
	}
}

// Hessian returns the Hessian matrix for the generalized linear model
// at the given parameter values.  The Hessian is returned as a
// one-dimensional array, which is the vectorized form of the Hessian
// matrix.  Either the observed or expected Hessian can be calculated.
func (glm *GLM) Hessian(params []float64, scale float64, ht statmodel.HessType, hess []float64) {

	var linpred []float64
	var mn []float64
	var lderiv []float64
	var lderiv2 []float64
	var va []float64
	var vad []float64
	var fac []float64
	var sfac []float64

	nvar := glm.Data.Nvar()
	glm.Data.Reset()
	zero(hess)

	for glm.Data.Next() {

		yda := glm.Data.YData()
		wgts := glm.Data.Weights()
		off := glm.Data.Offset()
		n := len(yda)

		// Adjust the allocations
		linpred = resize(linpred, n)
		mn = resize(mn, n)
		lderiv = resize(lderiv, n)
		va = resize(va, n)
		fac = resize(fac, n)
		sfac = resize(sfac, n)

		// Update the linear predictor
		zero(linpred)
		for j := 0; j < nvar; j++ {
			xda := glm.Data.XData(j)
			floats.AddScaled(linpred, params[j], xda)
		}
		if off != nil {
			floats.Add(linpred, off)
		}

		// The mean response
		glm.Link.InvLink(linpred, mn)

		glm.Link.Deriv(mn, lderiv)
		glm.Var.Var(mn, va)

		// Factor for the expected Hessian
		for i := 0; i < len(lderiv); i++ {
			fac[i] = 1 / (lderiv[i] * lderiv[i] * va[i])
		}

		// Adjust the factor for the observed Hessian
		if ht == statmodel.ObsHess {
			vad = resize(vad, n)
			lderiv2 = resize(lderiv2, n)
			glm.Link.Deriv2(mn, lderiv2)
			glm.Var.Deriv(mn, vad)
			scoreFactor(yda, mn, lderiv, va, sfac)

			for i, _ := range fac {
				h := va[i]*lderiv2[i] + lderiv[i]*vad[i]
				h *= sfac[i] * fac[i]
				if wgts != nil {
					h *= wgts[i]
				}
				fac[i] *= 1 + h
			}
		}

		// Update the Hessian matrix
		// TODO: use blas/floats
		for j1 := 0; j1 < nvar; j1++ {
			x1 := glm.Data.XData(j1)
			for j2 := 0; j2 < nvar; j2++ {
				x2 := glm.Data.XData(j2)
				if wgts == nil {
					for i, _ := range x1 {
						hess[j1*nvar+j2] -= fac[i] * x1[i] * x2[i]
					}
				} else {
					for i, _ := range x1 {
						hess[j1*nvar+j2] -= wgts[i] * fac[i] * x1[i] * x2[i]
					}
				}
			}
		}
	}

	// Account for the L2 penalty
	if glm.L2wgt != nil {
		nobs := float64(glm.Data.Nobs())
		for j, v := range glm.L2wgt {
			hess[j*nvar+j] -= nobs * v
		}
	}
}

// Fit estimates the parameters of the GLM and returns a results
// object.  Methods of statmodel.BaseResults can be used to obtain
// many attributes of the fitted model.
func (glm *GLM) Fit() GLMResults {

	if glm.L1wgt != nil {
		return glm.fitL1Reg()
	}

	nvar := glm.Data.Nvar()
	maxiter := 20

	var start []float64
	if glm.Start != nil {
		start = glm.Start
	} else {
		start = make([]float64, nvar)
	}

	if glm.L2wgt != nil {
		glm.FitMethod = "gradient"
	}

	var params []float64

	if strings.ToLower(glm.FitMethod) == "gradient" {
		params, _ = statmodel.FitParams(glm, start)
	} else {
		params = glm.fitIRLS(start, maxiter)
	}

	scale := glm.EstimateScale(params)

	vcov := statmodel.GetVcov(glm, params)
	floats.Scale(scale, vcov)
	ll := glm.LogLike(params, scale)
	xnames := glm.IndRegModel.Data.XNames()

	results := GLMResults{
		BaseResults: statmodel.NewBaseResults(glm,
			ll, params, xnames, vcov),
		scale: scale,
	}

	return results
}

// EstimateScale returns an estimate of the GLM scale parameter at the
// given parameter values.
func (glm *GLM) EstimateScale(params []float64) float64 {

	name := strings.ToLower(glm.Fam.Name)
	if name == "binomial" || name == "poisson" {
		return 1
	}

	nvar := glm.Data.Nvar()
	var linpred []float64
	var mn []float64
	var va []float64
	var ws float64

	glm.Data.Reset()
	var scale float64
	for glm.Data.Next() {

		yda := glm.Data.YData()
		wgt := glm.Data.Weights()
		off := glm.Data.Offset()
		n := len(yda)

		linpred = resize(linpred, n)
		mn = resize(mn, n)
		va = resize(va, n)

		zero(linpred)
		for j := 0; j < nvar; j++ {
			xda := glm.Data.XData(j)
			for i, x := range xda {
				linpred[i] += params[j] * x
			}
		}
		if off != nil {
			floats.AddTo(linpred, linpred, off)
		}

		// The mean response and variance
		glm.Link.InvLink(linpred, mn)
		glm.Var.Var(mn, va)

		for i, y := range yda {
			r := y - mn[i]
			if wgt == nil {
				scale += r * r / va[i]
				ws += 1
			} else {
				scale += wgt[i] * r * r / va[i]
				ws += wgt[i]
			}
		}
	}

	scale /= (ws - float64(nvar))

	return scale
}

func (results *GLMResults) Summary() string {

	glm := results.Model().(*GLM)

	s := results.BaseResults.Summary()
	tw := 80

	var buf bytes.Buffer

	buf.Write([]byte(strings.Repeat("-", tw)))
	buf.Write([]byte("\n"))
	buf.Write([]byte("                        Generalized Linear Model results\n"))
	buf.Write([]byte(strings.Repeat("-", tw)))
	buf.Write([]byte("\n"))

	// Must have even length, add "" to end if needed.
	top := []string{fmt.Sprintf("Family:   %s", glm.Fam.Name),
		fmt.Sprintf("Link:     %s", glm.Link.Name),
		fmt.Sprintf("Variance: %s", glm.Var.Name),
		fmt.Sprintf("Num obs:  %d", glm.DataProps().Nobs),
		fmt.Sprintf("Scale:    %f", results.scale),
		"",
	}

	c := fmt.Sprintf("%%-%ds", tw/2)
	for j, v := range top {
		u := fmt.Sprintf(c, v)
		buf.Write([]byte(u))
		if j%2 == 1 {
			buf.Write([]byte("\n"))
		}
	}

	return buf.String() + s
}

// resize returns a float64 slice of length n, using the initial
// subslice of x if it is big enough.
func resize(x []float64, n int) []float64 {
	if cap(x) >= n {
		return x[0:n]
	}
	return make([]float64, n)
}

// zero sets all elements of the slice to 0
func zero(x []float64) {
	for i, _ := range x {
		x[i] = 0
	}
}

// one sets all elements of the slice to 1
func one(x []float64) {
	for i, _ := range x {
		x[i] = 1
	}
}
