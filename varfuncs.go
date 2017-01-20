package goglm

type Variance struct {
	Var   VecFunc
	Deriv VecFunc
}

var BinomVar = Variance{
	Var:   binomVar,
	Deriv: binomVarDeriv,
}

var IdentVar = Variance{
	Var:   identVar,
	Deriv: identVarDeriv,
}

var ConstVar = Variance{
	Var:   constVar,
	Deriv: constVarDeriv,
}

var SquaredVar = Variance{
	Var:   squaredVar,
	Deriv: squaredVarDeriv,
}

var CubedVar = Variance{
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

func GenNegBinomialVariance(alpha float64) Variance {

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

	return Variance{
		Var:   vaf,
		Deriv: vad,
	}
}
