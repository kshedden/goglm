package goglm

import "math"

type vecFunc func([]float64, []float64)

type Link struct {
	name string

	// Link calculates the link function value (usually mapping
	// the mean to the linear predictor).
	link vecFunc

	// InvLink calculates the inverse value of the link function
	// (usually mapping the linear preditor to the mean).
	invLink vecFunc

	// Deriv calculates the derivative of the link function.
	deriv vecFunc

	// Deriv2 calculates the second derivative of the link function.
	deriv2 vecFunc
}

var LogLink = Link{
	name:    "Log",
	link:    logFunc,
	invLink: expFunc,
	deriv:   logDerivFunc,
	deriv2:  logDeriv2Func,
}

var IdLink = Link{
	name:    "Identity",
	link:    idFunc,
	invLink: idFunc,
	deriv:   idDerivFunc,
	deriv2:  idDeriv2Func,
}

var CLogLogLink = Link{
	name:    "CLogLog",
	link:    cloglogFunc,
	invLink: cloglogInvFunc,
	deriv:   cloglogDerivFunc,
	deriv2:  cloglogDeriv2Func,
}

var LogitLink = Link{
	name:    "Logit",
	link:    logitFunc,
	invLink: expitFunc,
	deriv:   logitDerivFunc,
	deriv2:  logitDeriv2Func,
}

var RecipLink = Link{
	name:    "Reciprocal",
	link:    genPowFunc(-1, 1),
	invLink: genPowFunc(-1, 1),
	deriv:   genPowFunc(-2, -1),
	deriv2:  genPowFunc(-3, 2),
}

var RecipSquaredLink = Link{
	name:    "ReciprocalSquared",
	link:    genPowFunc(-2, 1),
	invLink: genPowFunc(-0.5, 1),
	deriv:   genPowFunc(-3, -2),
	deriv2:  genPowFunc(-4, 6),
}

func logFunc(x []float64, y []float64) {
	for i := 0; i < len(x); i++ {
		y[i] = math.Log(x[i])
	}
}

func logDerivFunc(x []float64, y []float64) {
	for i := 0; i < len(x); i++ {
		y[i] = 1 / x[i]
	}
}

func logDeriv2Func(x []float64, y []float64) {
	for i := 0; i < len(x); i++ {
		y[i] = -1 / (x[i] * x[i])
	}
}

func expFunc(x []float64, y []float64) {
	for i := 0; i < len(x); i++ {
		y[i] = math.Exp(x[i])
	}
}

func logitFunc(x []float64, y []float64) {
	for i := 0; i < len(x); i++ {
		r := x[i] / (1 - x[i])
		y[i] = math.Log(r)
	}
}

func logitDerivFunc(x []float64, y []float64) {
	for i := 0; i < len(x); i++ {
		y[i] = 1 / (x[i] * (1 - x[i]))
	}
}

func logitDeriv2Func(x []float64, y []float64) {
	for i := 0; i < len(x); i++ {
		v := x[i] * (1 - x[i])
		y[i] = (2*x[i] - 1) / (v * v)
	}
}

func expitFunc(x []float64, y []float64) {
	for i := 0; i < len(x); i++ {
		y[i] = 1 / (1 + math.Exp(-x[i]))
	}
}

func idFunc(x []float64, y []float64) {
	copy(y, x)
}

func idDerivFunc(x []float64, y []float64) {
	one(y)
}

func idDeriv2Func(x []float64, y []float64) {
	zero(y)
}

func cloglogFunc(x []float64, y []float64) {
	for i, v := range x {
		y[i] = math.Log(-math.Log(1 - v))
	}
}

func cloglogDerivFunc(x []float64, y []float64) {
	for i, v := range x {
		y[i] = 1 / ((v - 1) * math.Log(1-v))
	}
}

func cloglogDeriv2Func(x []float64, y []float64) {
	for i, v := range x {
		f := math.Log(1 - v)
		r := -1 / ((1 - v) * (1 - v) * f)
		y[i] = r * (1 + 1/f)
	}
}

func cloglogInvFunc(x []float64, y []float64) {
	for i, v := range x {
		y[i] = 1 - math.Exp(-math.Exp(v))
	}
}

func genPowFunc(p float64, s float64) vecFunc {
	return func(x []float64, y []float64) {
		for i, _ := range x {
			y[i] = s * math.Pow(x[i], p)
		}
	}
}
