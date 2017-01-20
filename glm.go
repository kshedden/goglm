package goglm

import (
	"github.com/gonum/floats"
	"github.com/kshedden/statmodel"
)

// GLM describes a generalized linear model.
type GLM struct {
	statmodel.IndRegModel

	Fam  Family
	Link Link
	Var  Variance

	FitMethod FitMethodType
	Start     []float64

	// Additional information that is model-specific
	Aux interface{}
}

// FitMethodType defines a numerical algorithm for fitting a GLM.
type FitMethodType int

const (
	GradientFit = iota
	IRLSFit
)

// GLMResults describes the results of a fitted generalized linear model.
type GLMResults struct {
	statmodel.BaseResults

	scale float64
}

func (rslt *GLMResults) Scale() float64 {
	return rslt.scale
}

type GLMFamily int

const (
	BinomialFamily = iota
	GaussianFamily
	PoissonFamily
	NegBinomialFamily
	GammaFamily
	InvGaussianFamily
	QuasiPoissonFamily
)

// NewGLM creates a new GLM object for the given family, using its
// default link and variance functions.  The link and variance
// functions can be changed directly, but to safely change the link,
// use the SetLink method.
func NewGLM(fam Family, data statmodel.DataProvider) *GLM {

	var link Link
	var vaf Variance

	switch fam.FamType {
	case BinomialFamily:
		link = LogitLink
		vaf = BinomVar
	case PoissonFamily:
		link = LogLink
		vaf = IdentVar
	case QuasiPoissonFamily:
		link = LogLink
		vaf = IdentVar
	case GaussianFamily:
		link = IdLink
		vaf = ConstVar
	case GammaFamily:
		link = ReciprocalLink
		vaf = SquaredVar
	case InvGaussianFamily:
		link = ReciprocalSquaredLink
		vaf = CubedVar
	case NegBinomialFamily:
		alpha := fam.Aux.(NegBinomAux).Alpha
		return NewNegBinomialGLM(alpha, data)
	default:
		panic("Unknown GLM family type")
	}

	return &GLM{
		IndRegModel: statmodel.IndRegModel{Data: data},
		Fam:         fam,
		Link:        link,
		Var:         vaf,
		FitMethod:   IRLSFit,
	}
}

// NegBinomAux contains information specific to a negative binomial GLM.
type NegBinomAux struct {
	Alpha float64
}

func NewNegBinomialGLM(alpha float64, data statmodel.DataProvider) *GLM {

	fam := GenNegBinomialFamily(alpha, LogLink)
	vaf := GenNegBinomialVariance(alpha)

	return &GLM{
		IndRegModel: statmodel.IndRegModel{Data: data},
		Fam:         fam,
		Link:        LogLink,
		Var:         vaf,
		FitMethod:   IRLSFit,
		Aux:         NegBinomAux{Alpha: alpha},
	}
}

// SetLink sets the link function safely (restricting to the valid
// links for the GLM family).  It is also usually possible to set the
// Link field directly, but don't do this with the negative binomial
// family.
func (glm *GLM) SetLink(link Link) {

	if !glm.Fam.IsValidLink(link.LinkType) {
		panic("Invalid link")
	}

	if glm.Fam.FamType == NegBinomialFamily {
		// Need to reset the family when the link changes
		alpha := glm.Aux.(NegBinomAux).Alpha
		fam := GenNegBinomialFamily(alpha, LogLink)
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
		n := len(yda)
		wgts := glm.Data.Weights()

		// Adjust the allocations
		linpred = resize(linpred, n)
		mn = resize(mn, n)

		// Update the linear predictor
		zero(linpred)
		for j := 0; j < nvar; j++ {
			xda := glm.Data.XData(j)
			for i, x := range xda {
				linpred[i] += params[j] * x
			}
		}

		// Update the log likelihood value
		glm.Link.InvLink(linpred, mn)
		loglike += glm.Fam.LogLike(yda, mn, wgts, scale)
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

	nvar := glm.Data.Nvar()
	glm.Data.Reset()
	zero(score)

	for glm.Data.Next() {

		yda := glm.Data.YData()
		n := len(yda)
		wgts := glm.Data.Weights()

		// Adjust the allocations
		linpred = resize(linpred, n)
		mn = resize(mn, n)
		deriv = resize(deriv, n)
		va = resize(va, n)
		fac = resize(fac, n)

		// Update the linear predictor
		zero(linpred)
		for j := 0; j < nvar; j++ {
			xda := glm.Data.XData(j)
			for i, x := range xda {
				linpred[i] += params[j] * x
			}
		}

		glm.Link.InvLink(linpred, mn)
		glm.Link.Deriv(mn, deriv)
		glm.Var.Var(mn, va)

		scoreFactor(yda, mn, deriv, va, fac)

		for j := 0; j < nvar; j++ {
			xda := glm.Data.XData(j)
			if wgts == nil {
				for i, x := range xda {
					score[j] += fac[i] * x
				}
			} else {
				for i, x := range xda {
					score[j] += wgts[i] * fac[i] * x
				}
			}
		}
	}
}

// Hessian returns the Hessian matrix for the generalized linear model
// at the given parameter values.
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
		n := len(yda)
		wgts := glm.Data.Weights()

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
			for i, x := range xda {
				linpred[i] += params[j] * x
			}
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
}

// Fit estimates the parameters of the GLM and returns a results object.
func (glm *GLM) Fit() GLMResults {

	nvar := glm.Data.Nvar()
	maxiter := 20

	var start []float64
	if glm.Start != nil {
		start = glm.Start
	} else {
		start = make([]float64, nvar)
	}

	var params []float64

	if glm.FitMethod == GradientFit {
		params, _ = statmodel.FitParams(glm, start)
	} else {
		params = glm.fitIRLS(start, maxiter)
	}

	scale := glm.EstimateScale(params)

	vcov := statmodel.GetVcov(glm, params)
	floats.Scale(scale, vcov)
	ll := glm.LogLike(params, scale)

	results := GLMResults{
		BaseResults: statmodel.NewBaseResults(glm,
			ll, params, vcov),
		scale: scale,
	}

	return results
}

// EstimateScale returns an estimate of the GLM scale parameter at the
// given parameter values.
func (glm *GLM) EstimateScale(params []float64) float64 {

	if glm.Fam.FamType == BinomialFamily || glm.Fam.FamType == PoissonFamily {
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
		n := len(yda)
		linpred = resize(linpred, n)
		mn = resize(mn, n)
		va = resize(va, n)
		wgt := glm.Data.Weights()

		zero(linpred)
		for j := 0; j < nvar; j++ {
			xda := glm.Data.XData(j)
			for i, x := range xda {
				linpred[i] += params[j] * x
			}
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
