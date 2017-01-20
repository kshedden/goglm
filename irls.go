package goglm

import (
	"math"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

func (glm *GLM) fitIRLS(start []float64, maxiter int) []float64 {

	dtol := 1e-8

	var linpred []float64
	var mn []float64
	var va []float64
	var lderiv []float64
	var irlsw []float64
	var adjy []float64
	var nparam mat64.Vector

	nvar := glm.Data.Nvar()

	xty := make([]float64, nvar)
	xtx := make([]float64, nvar*nvar)

	var params []float64
	if start == nil {
		params = make([]float64, nvar)
	} else {
		params = start
	}

	var dev []float64

	for iter := 0; iter < maxiter; iter++ {
		glm.Data.Reset()
		zero(xtx)
		zero(xty)
		var devi float64
		for glm.Data.Next() {

			yda := glm.Data.YData()
			wgt := glm.Data.Weights()
			n := len(yda)

			// Allocations
			linpred = resize(linpred, n)
			mn = resize(mn, n)
			va = resize(va, n)
			lderiv = resize(lderiv, n)
			irlsw = resize(irlsw, n)
			adjy = resize(adjy, n)

			zero(linpred)
			for j := 0; j < nvar; j++ {
				xda := glm.Data.XData(j)
				for i, x := range xda {
					linpred[i] += params[j] * x
				}
			}

			if iter == 0 {
				glm.startingMu(yda, mn)
			} else {
				glm.Link.invLink(linpred, mn)
			}

			glm.Link.deriv(mn, lderiv)
			glm.Var.Var(mn, va)

			devi += glm.Fam.Deviance(yda, mn, wgt, 1)

			// Create weights for WLS
			if wgt != nil {
				for i, _ := range yda {
					irlsw[i] = wgt[i] / (lderiv[i] * lderiv[i] * va[i])
				}
			} else {
				for i, _ := range yda {
					irlsw[i] = 1 / (lderiv[i] * lderiv[i] * va[i])
				}
			}

			// Create an adjusted response for WLS
			for i, _ := range yda {
				adjy[i] = linpred[i] + lderiv[i]*(yda[i]-mn[i])
			}

			// Update the weighted moment matrices
			for j := 0; j < nvar; j++ {

				// Update x' w^-1 ya
				xda := glm.Data.XData(j)
				for i, y := range adjy {
					xty[j] += y * xda[i] * irlsw[i]
				}

				// Update x' w^-1 x
				for k := 0; k < nvar; k++ {
					xdb := glm.Data.XData(k)
					for i, _ := range xda {
						xtx[j*nvar+k] += xda[i] * xdb[i] * irlsw[i]
					}
				}
			}
		}

		// Update the parameters
		xtxm := mat64.NewDense(nvar, nvar, xtx)
		xtyv := mat64.NewVector(nvar, xty)
		err := nparam.SolveVec(xtxm, xtyv)
		if err != nil {
			panic(err)
		}
		params = nparam.RawVector().Data

		dev = append(dev, devi)
		if len(dev) > 3 && math.Abs(dev[len(dev)-1]-dev[len(dev)-2]) < dtol {
			break
		}
	}

	return params
}

func (glm *GLM) startingMu(y []float64, mn []float64) {

	var q float64
	if glm.Fam.FamType == BinomialFamily {
		q = 0.5
	} else {
		q = floats.Sum(y) / float64(len(y))
	}
	for i, _ := range mn {
		mn[i] = (y[i] + q) / 2
		if mn[i] < 0.1 {
			mn[i] = 0.1
		}
	}
}
