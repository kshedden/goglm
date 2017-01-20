package goglm

import "math"

type v3Func func([]float64, []float64, []float64, float64) float64

type Family struct {
	FamType    GLMFamily
	LogLike    v3Func
	Deviance   v3Func
	validLinks []LinkType
	link       Link // Used only for negative binomial
	Aux        interface{}
}

var Poisson = Family{
	FamType:    PoissonFamily,
	LogLike:    poissonLogLike,
	Deviance:   poissonDeviance,
	validLinks: []LinkType{LogLinkType, IdLinkType},
}

// QuasiPoisson is the same as Poisson, except that the scale parameter is estimated.
var QuasiPoisson = Family{
	FamType:    QuasiPoissonFamily,
	LogLike:    poissonLogLike,
	Deviance:   poissonDeviance,
	validLinks: []LinkType{LogLinkType, IdLinkType},
}

var Binomial = Family{
	FamType:    BinomialFamily,
	LogLike:    binomialLogLike,
	Deviance:   binomialDeviance,
	validLinks: []LinkType{LogitLinkType, LogLinkType, IdLinkType},
}

var Gaussian = Family{
	FamType:    GaussianFamily,
	LogLike:    gaussianLogLike,
	Deviance:   gaussianDeviance,
	validLinks: []LinkType{LogLinkType, IdLinkType, ReciprocalLinkType},
}

var Gamma = Family{
	FamType:    GammaFamily,
	LogLike:    gammaLogLike,
	Deviance:   gammaDeviance,
	validLinks: []LinkType{LogLinkType, IdLinkType, ReciprocalLinkType},
}

var InvGaussian = Family{
	FamType:    InvGaussianFamily,
	LogLike:    invGaussLogLike,
	Deviance:   invGaussianDeviance,
	validLinks: []LinkType{ReciprocalSquaredLinkType, ReciprocalLinkType, LogLinkType, IdLinkType},
}

func (fam *Family) IsValidLink(lt LinkType) bool {

	for _, q := range fam.validLinks {
		if lt == q {
			return true
		}
	}

	return false
}

func poissonLogLike(y, mn, wt []float64, scale float64) float64 {
	var ll float64
	var w float64 = 1
	for i := 0; i < len(y); i++ {
		g, _ := math.Lgamma(y[i] + 1)
		if wt != nil {
			w = wt[i]
		}
		ll += w * (y[i]*math.Log(mn[i]) - mn[i] - g)
	}
	return ll
}

func binomialLogLike(y, mn, wt []float64, scale float64) float64 {
	var ll float64
	var w float64 = 1
	for i := 0; i < len(y); i++ {
		if wt != nil {
			w = wt[i]
		}
		r := mn[i]/(1-mn[i]) + 1e-200
		ll += w * (y[i]*math.Log(r) + math.Log(1-mn[i]))
	}
	return ll
}

func gaussianLogLike(y, mn, wt []float64, scale float64) float64 {
	var ll float64
	var w float64 = 1
	var ws float64
	for i := 0; i < len(y); i++ {
		if wt != nil {
			w = wt[i]
		}
		r := y[i] - mn[i]
		ll -= w * r * r / (2 * scale)
		ws += w
	}
	ll -= ws * math.Log(2*math.Pi*scale) / 2
	return ll
}

func gammaLogLike(y, mn, wt []float64, scale float64) float64 {

	var ll float64
	var w float64 = 1
	for i := 0; i < len(y); i++ {
		if wt != nil {
			w = wt[i]
		}

		g, _ := math.Lgamma(1 / scale)
		v := y[i]/mn[i] + math.Log(mn[i]) + (scale-1)*math.Log(y[i])
		v += math.Log(scale) + scale*g
		ll -= w * v / scale
	}

	return ll
}

func invGaussLogLike(y, mn, wt []float64, scale float64) float64 {

	var ll float64
	var w float64 = 1
	var ws float64
	for i := 0; i < len(y); i++ {
		if wt != nil {
			w = wt[i]
		}

		r := y[i] - mn[i]
		v := r * r / (y[i] * mn[i] * mn[i] * scale)
		v += math.Log(scale * y[i] * y[i] * y[i])

		ll -= 0.5 * w * v
		ws += w
	}

	ll -= 0.5 * ws * math.Log(2*math.Pi)
	return ll
}

func poissonDeviance(y, mn, wgt []float64, scale float64) float64 {

	var dev float64
	var w float64 = 1

	for i := 0; i < len(y); i++ {
		if wgt != nil {
			w = wgt[i]
		}

		if y[i] > 0 {
			dev += 2 * w * y[i] * math.Log(y[i]/mn[i])
		}
	}
	dev /= scale

	return dev
}

func binomialDeviance(y, mn, wgt []float64, scale float64) float64 {

	var dev float64
	var w float64 = 1

	for i := 0; i < len(y); i++ {
		if wgt != nil {
			w = wgt[i]
		}

		dev -= 2 * w * (y[i]*math.Log(mn[i]) + (1-y[i])*math.Log(1-mn[i]))
	}

	return dev
}

func gammaDeviance(y, mn, wgt []float64, scale float64) float64 {

	var dev float64
	var w float64 = 1

	for i := 0; i < len(y); i++ {
		if wgt != nil {
			w = wgt[i]
		}

		dev += 2 * w * ((y[i]-mn[i])/mn[i] - math.Log(y[i]/mn[i]))
	}

	return dev
}

func invGaussianDeviance(y, mn, wgt []float64, scale float64) float64 {

	var dev float64
	var w float64 = 1

	for i := 0; i < len(y); i++ {
		if wgt != nil {
			w = wgt[i]
		}

		r := y[i] - mn[i]
		dev += w * (r * r / (y[i] * mn[i] * mn[i]))
	}
	dev /= scale

	return dev
}

func gaussianDeviance(y, mn, wgt []float64, scale float64) float64 {

	var dev float64
	var w float64 = 1

	for i := 0; i < len(y); i++ {
		if wgt != nil {
			w = wgt[i]
		}

		r := y[i] - mn[i]
		dev += w * r * r
	}
	dev /= scale

	return dev
}

func GenNegBinomialFamily(alpha float64, link Link) Family {

	loglike := func(y, mn, wt []float64, scale float64) float64 {

		var ll float64
		var w float64 = 1
		var lp []float64

		lp = resize(lp, len(y))
		link.Link(mn, lp)
		c3, _ := math.Lgamma(1 / alpha)

		for i := 0; i < len(y); i++ {
			if wt != nil {
				w = wt[i]
			}

			elp := math.Exp(lp[i])

			c1, _ := math.Lgamma(y[i] + 1/alpha)
			c2, _ := math.Lgamma(y[i] + 1)
			c := c1 - c2 - c3

			v := y[i] * math.Log(alpha*elp/(1+alpha*elp))
			v -= math.Log(1+alpha*elp) / alpha

			ll += w * (v + c)
		}

		return ll
	}

	deviance := func(y, mn, wt []float64, scale float64) float64 {

		var dev float64
		var w float64 = 1
		var lp []float64

		lp = resize(lp, len(y))
		link.Link(mn, lp)

		for i := 0; i < len(y); i++ {
			if wt != nil {
				w = wt[i]
			}

			if y[i] == 1 {
				z1 := y[i] * math.Log(y[i]/mn[i])
				z2 := (1 + alpha*y[i]) / alpha
				z2 *= math.Log((1 + alpha*y[i]) / (1 + alpha*mn[i]))
				dev += w * (z1 - z2)
			} else {
				dev += 2 * w * math.Log(1+alpha*mn[i]) / alpha
			}
		}
		dev /= scale

		return dev
	}

	return Family{
		FamType:    NegBinomialFamily,
		LogLike:    loglike,
		Deviance:   deviance,
		Aux:        NegBinomAux{Alpha: alpha},
		validLinks: []LinkType{LogLinkType, IdLinkType},
		link:       link,
	}
}
