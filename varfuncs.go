package goglm

import (
	"fmt"
	"strings"
)

// NewVariance returns a new variance function object corresponding to
// the given name.  Supported names are binomial, const, cubed,
// identity, and, squared.
func NewVariance(name string) *Variance {

	name = strings.ToLower(name)
	switch name {

	case "binomial":
		return &binomVariance
	case "identity":
		return &identVariance
	case "const":
		return &constVariance
	case "squared":
		return &squaredVariance
	case "cubed":
		return &cubedVariance
	default:
		msg := fmt.Sprintf("Unknown variance function: %s\n", name)
		panic(msg)
	}
}

// Variance represents a GLM variance function.
type Variance struct {
	Var   VecFunc
	Deriv VecFunc
}

var binomVariance = Variance{
	Var:   binomVar,
	Deriv: binomVarDeriv,
}

var identVariance = Variance{
	Var:   identVar,
	Deriv: identVarDeriv,
}

var constVariance = Variance{
	Var:   constVar,
	Deriv: constVarDeriv,
}

var squaredVariance = Variance{
	Var:   squaredVar,
	Deriv: squaredVarDeriv,
}

var cubedVariance = Variance{
	Var:   cubedVar,
	Deriv: cubedVarDeriv,
}

func binomVar(mn []float64, v []float64) {
	for i, p := range mn {
		v[i] = p * (1 - p)
	}
}

func binomVarDeriv(mn []float64, dv []float64) {
	for i, p := range mn {
		dv[i] = 1 - 2*p
	}
}

func identVar(mn []float64, v []float64) {
	copy(v, mn)
}

func identVarDeriv(mn []float64, v []float64) {
	one(v)
}

func constVar(mn []float64, v []float64) {
	one(v)
}

func constVarDeriv(mn []float64, v []float64) {
	zero(v)
}

func squaredVar(mn []float64, v []float64) {
	for i, m := range mn {
		v[i] = m * m
	}
}

func squaredVarDeriv(mn []float64, v []float64) {
	for i, m := range mn {
		v[i] = 2 * m
	}
}

func cubedVar(mn []float64, v []float64) {
	for i, m := range mn {
		v[i] = m * m * m
	}
}

func cubedVarDeriv(mn []float64, v []float64) {
	for i, m := range mn {
		v[i] = 3 * m * m
	}
}

// NewNegBinomialVariance returns a variance function for the negative
// binomial family, using the given parameter alpha to determine the
// mean/variance relationship.  The variance for mean m is m +
// alpha*m^2.
func NewNegBinomialVariance(alpha float64) *Variance {

	vaf := func(mn []float64, v []float64) {
		for i, m := range mn {
			v[i] = m + alpha*m*m
		}
	}

	vad := func(mn []float64, v []float64) {
		for i, m := range mn {
			v[i] = 1 + 2*alpha*m
		}
	}

	return &Variance{
		Var:   vaf,
		Deriv: vad,
	}
}
