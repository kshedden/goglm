package goglm

import (
	"fmt"
	"math"
	"strings"
)

// VecFunc is a function with two float64 array arguments.
type VecFunc func([]float64, []float64)

type Link struct {
	Name string

	// Link calculates the link function (usually mapping the mean
	// value to the linear predictor).
	Link VecFunc

	// InvLink calculates the inverse of the link function
	// (usually mapping the linear predictor to the mean value).
	InvLink VecFunc

	// Deriv calculates the derivative of the link function.
	Deriv VecFunc

	// Deriv2 calculates the second derivative of the link function.
	Deriv2 VecFunc
}

// NewLink returns a link function object corresponding to the given
// name.  Supported values are log, identity, cloglog, logit, recip,
// and recipsquared.
func NewLink(name string) *Link {

	name = strings.ToLower(name)

	switch name {
	case "log":
		return &logLink
	case "identity":
		return &idLink
	case "cloglog":
		return &cLogLogLink
	case "logit":
		return &logitLink
	case "recip":
		return &recipLink
	case "recipsquared":
		return &recipSquaredLink
	default:
		msg := fmt.Sprintf("Link name unknown: %s\n", name)
		panic(msg)
	}
}

var logLink = Link{
	Name:    "Log",
	Link:    logFunc,
	InvLink: expFunc,
	Deriv:   logDerivFunc,
	Deriv2:  logDeriv2Func,
}

var idLink = Link{
	Name:    "Identity",
	Link:    idFunc,
	InvLink: idFunc,
	Deriv:   idDerivFunc,
	Deriv2:  idDeriv2Func,
}

var cLogLogLink = Link{
	Name:    "CLogLog",
	Link:    cloglogFunc,
	InvLink: cloglogInvFunc,
	Deriv:   cloglogDerivFunc,
	Deriv2:  cloglogDeriv2Func,
}

var logitLink = Link{
	Name:    "Logit",
	Link:    logitFunc,
	InvLink: expitFunc,
	Deriv:   logitDerivFunc,
	Deriv2:  logitDeriv2Func,
}

var recipLink = Link{
	Name:    "Recip",
	Link:    genPowFunc(-1, 1),
	InvLink: genPowFunc(-1, 1),
	Deriv:   genPowFunc(-2, -1),
	Deriv2:  genPowFunc(-3, 2),
}

var recipSquaredLink = Link{
	Name:    "RecipSquared",
	Link:    genPowFunc(-2, 1),
	InvLink: genPowFunc(-0.5, 1),
	Deriv:   genPowFunc(-3, -2),
	Deriv2:  genPowFunc(-4, 6),
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

func genPowFunc(p float64, s float64) VecFunc {
	return func(x []float64, y []float64) {
		for i, _ := range x {
			y[i] = s * math.Pow(x[i], p)
		}
	}
}
