package goglm

import (
	"bytes"
	"fmt"
	"strings"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/optimize"

	"github.com/kshedden/dstream/dstream"
	"github.com/kshedden/statmodel"
)

// GLM describes a generalized linear model.
type GLM struct {
	Data dstream.Dstream

	// Positions of the covariates
	xpos []int

	// Name and position of the outcome variable
	yname string
	ypos  int

	// Name and position of the offset variable, if present.
	offsetname string
	offsetpos  int

	// Name and position of the weight variable, if present.
	weightname string
	weightpos  int

	// The GLM family
	fam *Family

	// The GLM link function
	link *Link

	// The GLM variance function
	vari *Variance

	// Either IRLS (default) or Gradient
	fitMethod string

	// Starting values, optional
	start []float64

	// L1 (lasso) penalty weight.  FitMethod is ignored if
	// non-zero.
	l1wgt []float64

	// L2 (ridge) penalty weights, optional.  Must fit using
	// Gradient method if present.
	l2wgt []float64
}

type GLMParams struct {
	params []float64
	scale  float64
}

func (glm *GLM) NumParams() int {
	return len(glm.xpos)
}

func (glm *GLM) Xpos() []int {
	return glm.xpos
}

func (glm *GLM) DataSet() dstream.Dstream {
	return glm.Data
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
func NewGLM(data dstream.Dstream, yname string) *GLM {

	return &GLM{
		Data:      data,
		yname:     yname,
		fitMethod: "IRLS",
	}
}

// FitMethod sets the fitting method, either IRLS or gradient.
func (glm *GLM) FitMethod(method string) *GLM {
	lmethod := strings.ToLower(method)
	if lmethod != "irls" && lmethod != "gradient" {
		msg := fmt.Sprintf("GLM fitting method %s not allowed.\n", method)
		panic(msg)
	}
	glm.fitMethod = method
	return glm
}

// Offset sets the name of the offset variable
func (glm *GLM) Offset(name string) *GLM {
	glm.offsetname = name
	return glm
}

// Weight sets the name of the weight variable.
func (glm *GLM) Weight(name string) *GLM {
	glm.weightname = name
	return glm
}

// Family sets the name of the GLM family variable.
func (glm *GLM) Family(fam *Family) *GLM {
	glm.fam = fam
	return glm
}

// L2Weight set the L2 weights.
func (glm *GLM) L2Weight(l2wgt []float64) *GLM {
	glm.l2wgt = l2wgt
	return glm
}

// Start sets starting values for the fitting algorithm.
func (glm *GLM) Start(start []float64) *GLM {
	glm.start = start
	return glm
}

// Link sets the link function.
func (glm *GLM) Link(link *Link) *GLM {

	if glm.fam == nil {
		panic("Must set family before setting link.\n")
	}
	if !glm.fam.IsValidLink(link) {
		panic("Invalid link")
	}
	glm.link = link

	if strings.ToLower(glm.fam.Name) == "negbinom" {
		// Need to reset the family when the link changes
		glm.fam = NewNegBinomFamily(glm.fam.alpha, link)
	}

	return glm
}

// VarFunc sets the GLM variance function.
func (glm *GLM) VarFunc(va *Variance) *GLM {
	glm.vari = va
	return glm
}

func (glm *GLM) findvars() {

	glm.offsetpos = -1
	glm.weightpos = -1
	glm.ypos = -1

	for k, na := range glm.Data.Names() {
		switch na {
		case glm.yname:
			glm.ypos = k
		case glm.weightname:
			glm.weightpos = k
		case glm.offsetname:
			glm.offsetpos = k
		default:
			glm.xpos = append(glm.xpos, k)
		}
	}

	if glm.ypos == -1 {
		msg := fmt.Sprintf("Outcome variable '%s' not found.", glm.yname)
		panic(msg)
	}
	if glm.weightpos == -1 && glm.weightname != "" {
		msg := fmt.Sprintf("Weight variable '%s' not found.", glm.weightname)
		panic(msg)
	}
	if glm.offsetpos == -1 && glm.offsetname != "" {
		msg := fmt.Sprintf("Offset variable '%s' not found.", glm.offsetname)
		panic(msg)
	}
}

func (glm *GLM) setup() {

	if glm.link == nil {
		glm.link = NewLink(glm.fam.validLinks[0])
	}

	if glm.vari == nil {
		name := strings.ToLower(glm.fam.Name)
		switch name {
		case "binomial":
			glm.vari = NewVariance("binomial")
		case "poisson":
			glm.vari = NewVariance("identity")
		case "quasipoisson":
			glm.vari = NewVariance("identity")
		case "gaussian":
			glm.vari = NewVariance("const")
		case "gamma":
			glm.vari = NewVariance("squared")
		case "invgaussian":
			glm.vari = NewVariance("cubed")
		case "negbinom":
			glm.vari = NewNegBinomVariance(glm.fam.alpha)
		default:
			msg := fmt.Sprintf("Unknown GLM family: %s\n", glm.fam.Name)
			panic(msg)
		}
	}
}

// Done completes definition of a GLM.  After calling Done the GLM can
// be fit by calling the Fit method.
func (glm *GLM) Done() *GLM {

	if glm.fam == nil {
		msg := "GLM: the family must be defined before calling Done.\n"
		panic(msg)
	}

	glm.findvars()

	glm.setup()

	if len(glm.start) == 0 {
		glm.start = make([]float64, glm.NumParams())
	}

	return glm
}

// SetFamily is a convenience method that sets the family, link, and
// variance function based on the given family name.  The link and
// variance functions are set to their canonical values.
func (glm *GLM) SetFamily(name string) *GLM {

	lname := strings.ToLower(name)
	switch lname {
	case "binomial":
		glm.fam = &binomial
		glm.link = NewLink("logit")
		glm.vari = NewVariance("binomial")
	case "poisson":
		glm.fam = &poisson
		glm.link = NewLink("log")
		glm.vari = NewVariance("identity")
	case "quasipoisson":
		glm.fam = &quasiPoisson
		glm.link = NewLink("log")
		glm.vari = NewVariance("identity")
	case "gaussian":
		glm.fam = &gaussian
		glm.link = NewLink("identity")
		glm.vari = NewVariance("const")
	case "gamma":
		glm.fam = &gamma
		glm.link = NewLink("recip")
		glm.vari = NewVariance("squared")
	case "invgaussian":
		glm.fam = &invGaussian
		glm.link = NewLink("recipsquared")
		glm.vari = NewVariance("cubed")
	case "negbinom":
		panic("GLM: can't set family to NegBinom using SetFamily")
	default:
		msg := fmt.Sprintf("Unknown GLM family: %s\n", name)
		panic(msg)
	}

	return glm
}

// LogLike returns the log-likelihood value for the generalized linear
// model at the given parameter values.
func (glm *GLM) LogLike(param statmodel.Parameter) float64 {

	par := param.(*GLMParams)
	params := par.params
	scale := par.scale

	var loglike float64
	var linpred []float64
	var mn []float64

	glm.Data.Reset()

	for glm.Data.Next() {

		var yda, wgts, off []float64

		yda = glm.Data.GetPos(glm.ypos).([]float64)
		n := len(yda)

		if glm.weightpos != -1 {
			wgts = glm.Data.GetPos(glm.weightpos).([]float64)
		}
		if glm.offsetpos != -1 {
			off = glm.Data.GetPos(glm.offsetpos).([]float64)
		}

		// Adjust the allocations
		linpred = resize(linpred, n)
		mn = resize(mn, n)

		// Update the linear predictor
		zero(linpred)
		for j, k := range glm.xpos {
			xda := glm.Data.GetPos(k).([]float64)
			floats.AddScaled(linpred, params[j], xda)
		}
		if off != nil {
			floats.Add(linpred, off)
		}

		// Update the log likelihood value
		glm.link.InvLink(linpred, mn)
		loglike += glm.fam.LogLike(yda, mn, wgts, scale)
	}

	// Account for the L2 penalty
	if glm.l2wgt != nil {
		nobs := float64(glm.Data.NumObs())
		for j, v := range glm.l2wgt {
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
func (glm *GLM) Score(param statmodel.Parameter, score []float64) {

	par := param.(*GLMParams)
	params := par.params

	var linpred []float64
	var mn []float64
	var deriv []float64
	var va []float64
	var fac []float64
	var facw []float64

	glm.Data.Reset()
	zero(score)

	for glm.Data.Next() {

		var yda, wgts, off []float64

		yda = glm.Data.GetPos(glm.ypos).([]float64)
		n := len(yda)

		if glm.weightpos != -1 {
			wgts = glm.Data.GetPos(glm.weightpos).([]float64)
		}
		if glm.offsetpos != -1 {
			off = glm.Data.GetPos(glm.offsetpos).([]float64)
		}

		// Adjust the allocations
		linpred = resize(linpred, n)
		mn = resize(mn, n)
		deriv = resize(deriv, n)
		va = resize(va, n)
		fac = resize(fac, n)
		facw = resize(facw, n)

		// Update the linear predictor
		zero(linpred)
		for j, k := range glm.xpos {
			xda := glm.Data.GetPos(k).([]float64)
			floats.AddScaled(linpred, params[j], xda)
		}
		if off != nil {
			floats.Add(linpred, off)
		}

		glm.link.InvLink(linpred, mn)
		glm.link.Deriv(mn, deriv)
		glm.vari.Var(mn, va)

		scoreFactor(yda, mn, deriv, va, fac)

		for j, k := range glm.xpos {

			xda := glm.Data.GetPos(k).([]float64)

			if wgts == nil {
				score[j] += floats.Dot(fac, xda)
			} else {
				floats.MulTo(facw, fac, wgts)
				score[j] += floats.Dot(facw, xda)
			}
		}
	}

	// Account for the L2 penalty
	if glm.l2wgt != nil {
		nobs := float64(glm.Data.NumObs())
		for j, v := range glm.l2wgt {
			score[j] -= nobs * v * params[j]
		}
	}
}

// Hessian returns the Hessian matrix for the generalized linear model
// at the given parameter values.  The Hessian is returned as a
// one-dimensional array, which is the vectorized form of the Hessian
// matrix.  Either the observed or expected Hessian can be calculated.
func (glm *GLM) Hessian(param statmodel.Parameter, ht statmodel.HessType, hess []float64) {

	par := param.(*GLMParams)
	params := par.params

	var linpred []float64
	var mn []float64
	var lderiv []float64
	var lderiv2 []float64
	var va []float64
	var vad []float64
	var fac []float64
	var sfac []float64

	nvar := glm.NumParams()
	glm.Data.Reset()
	zero(hess)

	for glm.Data.Next() {

		var yda, wgts, off []float64

		yda = glm.Data.GetPos(glm.ypos).([]float64)
		n := len(yda)

		if glm.weightpos != -1 {
			wgts = glm.Data.GetPos(glm.weightpos).([]float64)
		}
		if glm.offsetpos != -1 {
			off = glm.Data.GetPos(glm.offsetpos).([]float64)
		}

		// Adjust the allocations
		linpred = resize(linpred, n)
		mn = resize(mn, n)
		lderiv = resize(lderiv, n)
		va = resize(va, n)
		fac = resize(fac, n)
		sfac = resize(sfac, n)

		// Update the linear predictor
		zero(linpred)
		for j, k := range glm.xpos {
			xda := glm.Data.GetPos(k).([]float64)
			floats.AddScaled(linpred, params[j], xda)
		}
		if off != nil {
			floats.Add(linpred, off)
		}

		// The mean response
		glm.link.InvLink(linpred, mn)

		glm.link.Deriv(mn, lderiv)
		glm.vari.Var(mn, va)

		// Factor for the expected Hessian
		for i := 0; i < len(lderiv); i++ {
			fac[i] = 1 / (lderiv[i] * lderiv[i] * va[i])
		}

		// Adjust the factor for the observed Hessian
		if ht == statmodel.ObsHess {
			vad = resize(vad, n)
			lderiv2 = resize(lderiv2, n)
			glm.link.Deriv2(mn, lderiv2)
			glm.vari.Deriv(mn, vad)
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
		for j1, k1 := range glm.xpos {
			x1 := glm.Data.GetPos(k1).([]float64)
			for j2, k2 := range glm.xpos {
				x2 := glm.Data.GetPos(k2).([]float64)
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
	if glm.l2wgt != nil {
		nobs := float64(glm.Data.NumObs())
		for j, v := range glm.l2wgt {
			hess[j*nvar+j] -= nobs * v
		}
	}
}

// Fit estimates the parameters of the GLM and returns a results
// object.  Methods of statmodel.BaseResults can be used to obtain
// many attributes of the fitted model.
func (glm *GLM) Fit() GLMResults {

	/*
		if glm.L1wgt != nil {
			return glm.fitL1Reg()
		}
	*/

	nvar := glm.NumParams()
	maxiter := 20

	var start []float64
	if glm.Start != nil {
		start = glm.start
	} else {
		start = make([]float64, nvar)
	}

	if glm.l2wgt != nil {
		glm.fitMethod = "gradient"
	}

	var params []float64

	if strings.ToLower(glm.fitMethod) == "gradient" {
		params, _ = glm.fitGradient(start)
	} else {
		params = glm.fitIRLS(start, maxiter)
	}

	scale := glm.EstimateScale(params)

	vcov, _ := statmodel.GetVcov(glm, &GLMParams{params, scale})
	floats.Scale(scale, vcov)
	ll := glm.LogLike(&GLMParams{params, scale})

	var xn []string
	na := glm.Data.Names()
	for _, j := range glm.xpos {
		xn = append(xn, na[j])
	}

	results := GLMResults{
		BaseResults: statmodel.NewBaseResults(glm,
			ll, params, xn, vcov),
		scale: scale,
	}

	return results
}

func (glm *GLM) fitGradient(start []float64) ([]float64, float64) {

	p := optimize.Problem{
		Func: func(x []float64) float64 {
			return -glm.LogLike(&GLMParams{x, 1})
		},
		Grad: func(grad, x []float64) {
			glm.Score(&GLMParams{x, 1}, grad)
			floats.Scale(-1, grad)
		},
	}

	settings := optimize.DefaultSettings()
	settings.Recorder = nil
	settings.GradientThreshold = 1e-8
	settings.FunctionConverge = &optimize.FunctionConverge{
		Absolute:   0,
		Relative:   0,
		Iterations: 200,
	}

	optrslt, err := optimize.Local(p, start, settings, &optimize.BFGS{})
	if err != nil {
		panic(err)
	}
	if err = optrslt.Status.Err(); err != nil {
		panic(err)
	}

	params := optrslt.X
	fvalue := -optrslt.F

	return params, fvalue
}

// EstimateScale returns an estimate of the GLM scale parameter at the
// given parameter values.
func (glm *GLM) EstimateScale(params []float64) float64 {

	name := strings.ToLower(glm.fam.Name)
	if name == "binomial" || name == "poisson" {
		return 1
	}

	nvar := glm.NumParams()
	var linpred []float64
	var mn []float64
	var va []float64
	var ws float64

	glm.Data.Reset()
	var scale float64
	for glm.Data.Next() {

		var yda, wgt, off []float64

		yda = glm.Data.GetPos(glm.ypos).([]float64)
		n := len(yda)

		if glm.weightpos != -1 {
			wgt = glm.Data.GetPos(glm.weightpos).([]float64)
		}
		if glm.offsetpos != -1 {
			off = glm.Data.GetPos(glm.offsetpos).([]float64)
		}

		linpred = resize(linpred, n)
		mn = resize(mn, n)
		va = resize(va, n)

		zero(linpred)
		for j, k := range glm.xpos {
			xda := glm.Data.GetPos(k).([]float64)
			for i, x := range xda {
				linpred[i] += params[j] * x
			}
		}
		if off != nil {
			floats.AddTo(linpred, linpred, off)
		}

		// The mean response and variance
		glm.link.InvLink(linpred, mn)
		glm.vari.Var(mn, va)

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
	top := []string{fmt.Sprintf("Family:   %s", glm.fam.Name),
		fmt.Sprintf("Link:     %s", glm.link.Name),
		fmt.Sprintf("Variance: %s", glm.vari.Name),
		fmt.Sprintf("Num obs:  %d", glm.DataSet().NumObs()),
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
